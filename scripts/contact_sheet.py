"""Contact sheets of the target-private images, for choosing one-shot images by eye (adapt.fixed_shots).

One PNG per private class of the config's target domain: every image of the class as a thumbnail,
captioned with its file name (the value adapt.fixed_shots expects). Reads the image folders only.
  python scripts/contact_sheet.py configs/office_a2d.yaml --out oneshot_review
  python scripts/contact_sheet.py configs/office_a2d.yaml --out oneshot_review --classes speaker --thumb 320
"""
import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cida.common import load_config  # noqa: E402
from cida.data import resolve_classes  # noqa: E402


def sheet(paths, thumb, cols, title):
    rows = math.ceil(len(paths) / cols)
    cap = 16
    W, H = cols * thumb, rows * (thumb + cap) + 24
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), title, fill="black")
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = (k % cols) * thumb, 24 + (k // cols) * (thumb + cap)
        img.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        draw.text((x + 2, y + thumb + 1), p.name, fill="black")
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="oneshot_review")
    ap.add_argument("--domain", default=None, help="default: the config's target domain")
    ap.add_argument("--classes", nargs="*", default=None, help="class names (default: all private classes)")
    ap.add_argument("--thumb", type=int, default=160)
    ap.add_argument("--data_root", default=None, help="default: the config's data_root")
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, [f"data_root={args.data_root}"] if args.data_root else [])
    dom = args.domain or cfg["tgt_domain"]
    root = Path(cfg["data_root"]) / cfg["domain_paths"][dom]
    names = sorted(d.name for d in root.iterdir() if d.is_dir())  # ImageFolder order
    private = [names[c] for c in resolve_classes(cfg["private"], names)]
    out = Path(args.out) / dom
    out.mkdir(parents=True, exist_ok=True)
    for name in args.classes or private:
        paths = sorted(p for p in (root / name).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        f = out / f"{name}.png"
        sheet(paths, args.thumb, args.cols, f"{dom}/{name}  ({len(paths)} images)").save(f)
        print(f"{f}  {len(paths)} images")


if __name__ == "__main__":
    main()
