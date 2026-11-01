"""Run a local Ollama vision baseline; all requests/responses are retained."""

import argparse, base64, json, time, io
from pathlib import Path
import requests
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--model", default="gemma3:4b")
p.add_argument("--images", nargs="+", required=True)
p.add_argument("--reference")
p.add_argument("--output", required=True)
p.add_argument(
    "--hex",
    action="store_true",
    help="Constrain output to explicit codepoint IDs rather than rare Unicode tokens.",
)
p.add_argument("--tokens", type=int, default=700)
p.add_argument("--upscale", type=int, default=1)
a = p.parse_args()
records = []
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
prompt = (
    "Transcribe the historical Khojki text in the first image literally, left to right. "
    "Return the original Khojki Unicode script (U+11200 onward), preserving vowel signs and word separators U+1123A. "
    "Digits may be Gujarati U+0AE6..U+0AEF. Do not transliterate into Gujarati, Devanagari, Latin or Arabic. "
    "Do not translate or explain. Do not invent text. If a character is unreadable use U+FFFD. "
    "Return JSON with exactly one string field: text."
)
if a.reference:
    prompt += " The second image is a visual character reference, with hexadecimal Unicode labels. Use it to identify shapes; do not transcribe the reference itself."
if a.hex:
    prompt = prompt.replace(
        "Return JSON with exactly one string field: text.",
        'Return JSON with a codepoints array of hexadecimal Unicode labels, e.g. ["11208","1122C","1123A"]. Use 0020 for a large column gap. Identify each visual shape from the reference chart. Do not output transliterations.',
    )
for path in a.images:
    if a.upscale > 1:
        im = Image.open(path)
        im = im.resize(
            (im.width * a.upscale, im.height * a.upscale), Image.Resampling.LANCZOS
        )
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        source_bytes = buffer.getvalue()
    else:
        source_bytes = Path(path).read_bytes()
    images = [base64.b64encode(source_bytes).decode()]
    if a.reference:
        images.append(base64.b64encode(Path(a.reference).read_bytes()).decode())
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "options": {
            "temperature": 0,
            "seed": 17,
            "num_predict": a.tokens,
            "num_ctx": 8192,
        },
        "keep_alive": 0,
    }
    if a.hex:
        codes = (
            [f"{c:05X}" for c in range(0x11200, 0x11242) if c != 0x11212]
            + [f"{c:04X}" for c in range(0xAE6, 0xAF0)]
            + ["0020", "FFFD"]
        )
        payload["format"] = {
            "type": "object",
            "properties": {
                "codepoints": {
                    "type": "array",
                    "items": {"type": "string", "enum": codes},
                }
            },
            "required": ["codepoints"],
        }
    if a.model.startswith("gemma4"):
        payload["think"] = False
    t = time.perf_counter()
    try:
        r = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=240)
        r.raise_for_status()
        response = r.json()
        raw = response.get("message", {}).get("content", "")
        parsed = True
        try:
            value = json.loads(raw)
            text = (
                "".join(chr(int(c, 16)) for c in value["codepoints"])
                if a.hex
                else value["text"]
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            text = raw
            parsed = False
        record = {
            "id": Path(path).stem,
            "image": str(path),
            "model": a.model,
            "prompt": prompt,
            "reference": a.reference,
            "text": text,
            "parse_success": parsed,
            "options": payload["options"],
            "upscale": a.upscale,
            "hex_output": a.hex,
            "raw_response": response,
            "wall_seconds": time.perf_counter() - t,
        }
    except Exception as e:
        record = {
            "id": Path(path).stem,
            "error": str(e),
            "wall_seconds": time.perf_counter() - t,
        }
    records.append(record)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k in ["id", "text", "error", "wall_seconds"]
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
