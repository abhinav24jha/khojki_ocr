"""Adapt to real print with synthetic replay and fully resumable checkpoints."""

from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import time

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--device", default="mps")
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--output", default="adapt-v1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--synthetic-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--annotations", type=Path, default=ROOT / "data/annotations.json")
    p.add_argument("--layout", choices=["legacy", "robust"], default="legacy")
    p.add_argument("--word-fraction", type=float, default=0)
    p.add_argument("--word-geometry", choices=["legacy", "preserve"], default="legacy")
    p.add_argument("--bold-replay", action="store_true")
    p.add_argument("--legacy-rgb-word-padding", action="store_true")
    p.add_argument("--freeze-bn", action="store_true")
    p.add_argument("--freeze-rnn", action="store_true")
    p.add_argument("--packed", action="store_true")
    p.add_argument("--height", type=int, choices=[48, 64], default=48)
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error("--steps must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("--lr must be finite and positive")
    if (
        not 0 <= args.synthetic_fraction <= 1
        or not 0 <= args.word_fraction <= 1
        or args.word_fraction + args.synthetic_fraction > 1
    ):
        p.error("Mixture fractions must be nonnegative and sum to at most one")
    if args.train_limit is not None and args.train_limit < 1:
        p.error("--train-limit must be positive")
    if not 0 <= args.seed < 2**32:
        p.error("--seed must be in [0, 2**32)")
    if (
        not args.output
        or Path(args.output).name != args.output
        or args.output in {".", ".."}
    ):
        p.error("--output must be a simple experiment name")
    if not Path(args.checkpoint).is_file():
        p.error("Checkpoint does not exist")
    if not args.annotations.is_file():
        p.error("Annotations do not exist")
    return args


def prepare_data(args, rows):
    """Build training-only line and word banks and retain dev rows for scoring."""
    from PIL import Image
    from khojki_ocr.recognize import split_columns, preprocess

    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if args.train_limit:
        train = train[: args.train_limit]
    if not train:
        raise ValueError("Training split is empty")
    if not dev or not any(r["text"] for r in dev):
        raise ValueError("Development split must contain reference text")
    samples = []
    for r in train:
        with Image.open(ROOT / "data" / r["image"]) as source_image:
            im = source_image.convert("L")
        chunks = split_columns(im, args.layout)
        texts = r["text"].split(" ")
        if len(chunks) == len(texts):
            samples.extend((preprocess(c), t) for c, t in zip(chunks, texts))
        else:
            samples.append((preprocess(im), r["text"]))
    print("real lines", len(train), "training chunks", len(samples), flush=True)
    wordbank = []
    if args.word_fraction:
        from khojki_ocr.word_augmentation import training_words

        for r in train:
            with Image.open(ROOT / "data" / r["image"]) as source_image:
                wordbank.extend(
                    (preprocess(im), txt)
                    for im, txt in training_words(
                        source_image,
                        r["text"],
                        preserve_height=args.word_geometry == "preserve",
                        white_padding=not args.legacy_rgb_word_padding,
                    )
                )
        if not wordbank:
            raise ValueError("No verified training word crops available")
        print("word crops from training only", len(wordbank), flush=True)

    return train, dev, samples, wordbank


def build_replay(args):
    """Render the same seeded 1,000-line synthetic replay bank on every resume."""
    from khojki_ocr.render import Renderer

    renderers = [
        Renderer(ROOT / f"assets/Noto{x}Khojki-Regular.ttf", 36)
        for x in ["Sans", "Serif"]
    ]
    if args.bold_replay:
        renderers.append(Renderer(ROOT / "assets/NotoSerifKhojki-Bold.ttf", 36))
    base = [chr(c) for c in range(0x11200, 0x1122C) if c != 0x11212]
    cons = base[8:]
    signs = [""] * 7 + [chr(c) for c in range(0x1122C, 0x11235)]
    synthetic = []
    for i in range(1000):
        s = ""
        for j in range(random.randint(3, 18)):
            c = random.choice(base)
            s += c
            if c in cons:
                s += random.choice(signs)
            if random.random() < 0.2:
                s += "\U0001123a"
        if i % 5 == 0:
            s = "".join(
                random.choices(
                    [chr(c) for c in range(0xAE6, 0xAF0)], k=random.randint(1, 3)
                )
            )
        synthetic.append((renderers[i % len(renderers)].render(s), s))

    return synthetic


def augment(im, height):
    from PIL import Image, ImageFilter
    from khojki_ocr.model import image_tensor

    im = im.resize(
        (
            max(12, round(im.width * random.uniform(0.82, 1.18))),
            max(20, round(im.height * random.uniform(0.85, 1.15))),
        ),
        Image.Resampling.BILINEAR,
    )
    if random.random() < 0.4:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.5)))
    im = im.rotate(
        random.uniform(-0.7, 0.7),
        resample=Image.Resampling.BILINEAR,
        expand=True,
        fillcolor=255,
    )
    return image_tensor(im, height)


def evaluate(model, dev, args):
    """Score raw codepoints on development lines with the selected layout."""
    import torch
    from PIL import Image
    from khojki_ocr.model import collate, image_tensor, decode
    from khojki_ocr.recognize import split_columns, preprocess
    from khojki_ocr.metrics import score

    model.eval()
    results = []
    with torch.no_grad():
        for r in dev:
            with Image.open(ROOT / "data" / r["image"]) as source_image:
                chunks = split_columns(source_image.copy(), args.layout)
            x, _, lengths, _, _ = collate(
                [(image_tensor(preprocess(im), args.height), "") for im in chunks]
            )
            pred = " ".join(
                t.strip()
                for t in decode(model(x.to(args.device), lengths), lengths)
                if t.strip()
            )
            results.append(
                {
                    "id": r["id"],
                    "reference": r["text"],
                    "prediction": pred,
                    **score(r["text"], pred),
                }
            )
    return results


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from khojki_ocr.model import CRNN, VOCAB, collate
    from khojki_ocr.checkpoints import load_single_checkpoint, save_checkpoint
    from khojki_ocr.word_augmentation import concatenate_words

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    annotations = args.annotations.read_bytes()
    rows = json.loads(annotations)
    train, dev, samples, wordbank = prepare_data(args, rows)
    # Rebuild the seeded replay bank before restoring saved RNG state. Reversing
    # this order changes both the bank and the resumed augmentation sequence.
    synthetic = build_replay(args)
    saved = load_single_checkpoint(args.checkpoint)
    model = CRNN(packed=args.packed)
    model.load_state_dict(saved["model"])
    model.to(args.device)
    if args.freeze_rnn:
        for param in model.rnn.parameters():
            param.requires_grad_(False)
        # cuDNN backward through frozen recurrent weights still needs training mode.
        # Disable recurrent dropout without switching the layer to eval.
        model.rnn.dropout = 0.0
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0001,
    )
    start_step = 0
    if args.resume:
        if saved["annotation_sha256"] != hashlib.sha256(annotations).hexdigest():
            raise ValueError(
                "References changed; start a new named experiment rather than resuming."
            )
        for key in [
            "seed",
            "train_limit",
            "synthetic_fraction",
            "word_fraction",
            "bold_replay",
            "layout",
            "height",
            "packed",
            "word_geometry",
            "legacy_rgb_word_padding",
            "freeze_bn",
            "freeze_rnn",
            "lr",
        ]:
            if key in saved.get("training_config", {}) and saved["training_config"][
                key
            ] != getattr(args, key):
                raise ValueError(
                    f"Resume requires original {key}; warm start a new experiment to change it."
                )
        opt.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if args.device == "mps" and "mps_rng" in saved:
            torch.mps.set_rng_state(saved["mps_rng"])
        if args.device.startswith("cuda") and "cuda_rng" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if start_step >= args.steps:
        raise ValueError(
            "Requested step target must exceed the resumed checkpoint step"
        )
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    out = ROOT / "experiments" / args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / "config.json").write_text(
        json.dumps(
            config
            | {
                "annotation_sha256": hashlib.sha256(annotations).hexdigest(),
                "train_line_ids": [r["id"] for r in train],
                "validation_line_ids": [r["id"] for r in dev],
                "test_used": False,
                "word_bank_size": len(wordbank),
            },
            indent=2,
        )
    )
    history = saved.get("history", []) if args.resume else []
    best = saved.get("best_validation_cer", 999) if args.resume else 999
    start = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        batch = []
        for k in range(12):
            u = random.random()
            if u < args.synthetic_fraction:
                im, txt = random.choice(synthetic)
            elif u < args.synthetic_fraction + args.word_fraction:
                im, txt = concatenate_words(
                    random.choices(wordbank, k=random.choice([1, 2, 3, 4, 5, 6])),
                    gap=random.randint(1, 7),
                    preserve_height=args.word_geometry == "preserve",
                )
            else:
                im, txt = random.choice(samples)
            batch.append((augment(im, args.height), txt))
        x, y, xlen, ylen, txt = collate(batch)
        model.train()
        if args.freeze_bn:
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.eval()
        opt.zero_grad()
        logp = model(x.to(args.device), xlen)
        loss = loss_fn(logp.cpu(), y, xlen, ylen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        if step % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
        if step % 50 == 0 or step == args.steps:
            results = evaluate(model, dev, args)
            n = sum(r["reference_characters"] for r in results)
            errors = sum(r["errors"] for r in results)
            cer = errors / n
            record = {
                "step": step,
                "loss": loss.item(),
                "validation_cer": cer,
                "errors": errors,
                "characters": n,
                "seconds": time.perf_counter() - start,
                "lines": results,
            }
            history.append(record)
            (out / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2)
            )
            print(
                json.dumps({k: v for k, v in record.items() if k != "lines"}),
                flush=True,
            )
            checkpoint = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(),
                "vocab": VOCAB,
                "step": step,
                "validation_cer": cer,
                "best_validation_cer": min(best, cer),
                "history": history,
                "training_config": config,
                "height": args.height,
                "packed": args.packed,
                "layout": args.layout,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "annotation_sha256": hashlib.sha256(annotations).hexdigest(),
            }
            if args.device == "mps":
                checkpoint["mps_rng"] = torch.mps.get_rng_state()
            if args.device.startswith("cuda"):
                checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
            save_checkpoint(checkpoint, out / "last.pt")
            if cer < best:
                best = cer
                save_checkpoint(checkpoint, out / "best.pt")


if __name__ == "__main__":
    main()
