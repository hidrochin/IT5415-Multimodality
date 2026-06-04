"""CLI: download a sample academic PDF to test the pipeline.

Default is "Attention Is All You Need" (arXiv 1706.03762) — it matches the
spec's example question, "What is self-attention?".

    python scripts/fetch_sample.py
"""
import argparse
import pathlib
import urllib.request

DEFAULT_URL = "https://arxiv.org/pdf/1706.03762"
DEFAULT_NAME = "attention_is_all_you_need.pdf"
RAW_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a sample PDF into data/raw/.")
    ap.add_argument("--url", default=DEFAULT_URL, help="PDF URL to download")
    ap.add_argument("--name", default=DEFAULT_NAME, help="Destination filename")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / args.name
    print(f"Downloading {args.url} ...")
    req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0 (mmrag fetch)"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"Saved to {dest}  ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
