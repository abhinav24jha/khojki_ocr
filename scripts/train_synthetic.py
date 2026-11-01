"""Train a compact CTC line recognizer with shaped, augmented synthetic text.

The --resume option is a weights-only warm start, not an exact optimizer resume.
"""

from pathlib import Path
from collections import Counter
import argparse
import json
import random
import time

ROOT = Path(__file__).resolve().parents[1]
base = [chr(c) for c in range(0x11200, 0x1122C) if c != 0x11212]
cons = [chr(c) for c in range(0x11208, 0x1122C) if c != 0x11212]
signs = [""] * 7 + [chr(c) for c in range(0x1122C, 0x11235)]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--output", default="synthetic-v1")
    p.add_argument(
        "--resume", help="Warm-start weights; this is not an exact optimizer resume."
    )
    p.add_argument("--repertoire", choices=["legacy", "expanded"], default="legacy")
    args = p.parse_args(argv)
    if args.steps < 1 or args.batch < 1:
        p.error("--steps and --batch must be positive")
    if (
        not args.output
        or Path(args.output).name != args.output
        or args.output in {".", ".."}
    ):
        p.error("--output must be a simple experiment name")
    if args.resume and not Path(args.resume).is_file():
        p.error("Warm-start checkpoint does not exist")
    return args


def sentence(rng, repertoire="legacy"):
    s = ""
    for i in range(rng.randint(5, 23)):
        c = rng.choice(
            base + (["\U0001123f", "\U00011240"] if repertoire == "expanded" else [])
        )
        s += c
        if c in cons or c == "\U0001123f":
            if repertoire == "expanded" and rng.random() < 0.04:
                s += "\U00011236"
            s += rng.choice(signs)
            if repertoire == "expanded" and rng.random() < 0.08:
                s += rng.choice(["\U00011234", "\U00011237"])
        if rng.random() < 0.24:
            s += (
                rng.choice(
                    ["\U0001123a"] * 12
                    + [
                        "\U00011238",
                        "\U00011239",
                        "\U0001123b",
                        "\U0001123c",
                        "\U0001123d",
                        " ",
                    ]
                )
                if repertoire == "expanded"
                else "\U0001123a"
            )
    if repertoire == "expanded" and rng.random() < 0.1:
        s += rng.choice(
            [
                "\U00011208\U00011235\U00011229\U00011236",
                "\U00011210\U00011235\U00011213",
                "\U00011219\U00011235\U00011226",
                "\U0001121b\U00011235\U00011226",
                "\U00011208\U00011241",
                "\U00011208\U0001123e",
            ]
        )
    if rng.random() < 0.25:
        s += " " + "".join(
            rng.choices([chr(c) for c in range(0xAE6, 0xAF0)], k=rng.randint(1, 2))
        )
    return s


class Synthetic:
    """Small eager image bank with the map-style Dataset protocol.

    Rendering uses an isolated RNG; augmentation uses the seeded training RNG.
    A torch base class is unnecessary for DataLoader's indexed access.
    """

    def __init__(self, n, seed, augment, fonts, repertoire="legacy"):
        self.items = []
        self.augment = augment
        rng = random.Random(seed)
        for i in range(n):
            text = sentence(rng, repertoire)
            self.items.append((fonts[i % 2].render(text, pad=5), text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image, ImageFilter
        from khojki_ocr.model import image_tensor

        im, text = self.items[i]
        if self.augment:
            im = im.resize(
                (
                    max(12, int(im.width * random.uniform(0.8, 1.2))),
                    random.randint(27, 49),
                ),
                Image.Resampling.BILINEAR,
            )
            if random.random() < 0.6:
                im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.65)))
            if random.random() < 0.3:
                im = im.filter(ImageFilter.MinFilter(3))
            im = im.rotate(
                random.uniform(-1.2, 1.2),
                resample=Image.Resampling.BILINEAR,
                expand=True,
                fillcolor=255,
            )
        return image_tensor(im), text


def evaluate(model, vloader, device):
    """Compute aggregate codepoint CER without changing the training sampler."""
    import torch
    from khojki_ocr.model import decode
    from khojki_ocr.metrics import score

    model.eval()
    errors = n = 0
    examples = []
    with torch.no_grad():
        for vx, vy, vl, vyl, vt in vloader:
            preds = decode(model(vx.to(device)), vl)
            for gt, pred in zip(vt, preds):
                z = score(gt, pred)
                errors += z["errors"]
                n += len(gt)
                if len(examples) < 3:
                    examples.append({"reference": gt, "prediction": pred})
    return errors, n, examples


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from khojki_ocr.render import Renderer
    from khojki_ocr.model import CRNN, VOCAB, collate
    from khojki_ocr.checkpoints import load_single_checkpoint, save_checkpoint

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    fonts = [
        Renderer(ROOT / f"assets/Noto{font}Khojki-Regular.ttf", 36)
        for font in ["Sans", "Serif"]
    ]
    start = time.perf_counter()
    train = Synthetic(3000, 17, True, fonts, args.repertoire)
    val = Synthetic(120, 991, False, fonts, args.repertoire)
    print("generated", time.perf_counter() - start, flush=True)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate)
    vloader = DataLoader(val, batch_size=args.batch, collate_fn=collate)
    model = CRNN().to(args.device)
    if args.resume:
        model.load_state_dict(load_single_checkpoint(args.resume)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    out = ROOT / "experiments" / args.output
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter(c for _, text in train.items for c in text)
    (out / "config.json").write_text(
        json.dumps(
            vars(args)
            | {
                "seed": 17,
                "train_samples": len(train),
                "validation_samples": len(val),
                "vocab": VOCAB,
                "training_codepoint_counts": {
                    f"U+{ord(c):04X}": n for c, n in counts.items()
                },
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    best = 999
    step = 0
    it = iter(loader)
    history = []
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, y, xlen, ylen, txt = batch
        model.train()
        opt.zero_grad()
        logits = model(x.to(args.device))
        # Apple's backend does not implement CTC on all versions; the differentiable
        # CPU loss keeps convolutions and recurrent layers on the GPU.
        loss = loss_fn(logits.cpu(), y, xlen, ylen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        step += 1
        if step % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(loss.item(), 4),
                        "seconds": round(time.perf_counter() - start, 1),
                    }
                ),
                flush=True,
            )
        if step % 100 == 0 or step == args.steps:
            errors, n, examples = evaluate(model, vloader, args.device)
            record = {
                "step": step,
                "loss": loss.item(),
                "validation_cer": errors / n,
                "errors": errors,
                "characters": n,
                "seconds": time.perf_counter() - start,
                "examples": examples,
            }
            history.append(record)
            (out / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2)
            )
            print(
                json.dumps({k: v for k, v in record.items() if k != "examples"}),
                flush=True,
            )
            checkpoint = {
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "vocab": VOCAB,
                "step": step,
                "validation_cer": errors / n,
                "height": 48,
                "training_config": vars(args),
                "warm_start_only": True,
            }
            save_checkpoint(checkpoint, out / "last.pt")
            if errors / n < best:
                best = errors / n
                save_checkpoint(checkpoint, out / "best.pt")
            if best < 0.015 and step >= 500:
                break


if __name__ == "__main__":
    main()
