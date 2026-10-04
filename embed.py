"""Embed every image in a folder with open_clip (ViT-B-32) and cache the vectors.

Cache layout (cache/):
  vectors.npy  float32 [N, 512], L2-normalised, row i belongs to files[i]
  index.json   {model, pretrained, preprocess, dim, folder, files: [{name, size, mtime_ns}]}

Re-runs only embed new or changed files. If model, pretrained or preprocess in the
index differ from the values below, the whole cache is rebuilt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
PREPROCESS = "letterbox-black"  # bump this string whenever pad_to_square changes
CACHE_DIR = Path(__file__).parent / "cache"
VECTORS_PATH = CACHE_DIR / "vectors.npy"
INDEX_PATH = CACHE_DIR / "index.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BATCH_SIZE = 32


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(MODEL_NAME)


def pad_to_square(img: Image.Image) -> Image.Image:
    """Letterbox onto a black square so the model sees the whole frame."""
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


def list_images(folder: Path, recursive: bool) -> list[Path]:
    paths = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def file_key(path: Path, folder: Path) -> dict:
    st = path.stat()
    return {
        "name": path.relative_to(folder).as_posix(),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def load_cache() -> tuple[dict | None, np.ndarray | None]:
    if not (INDEX_PATH.exists() and VECTORS_PATH.exists()):
        return None, None
    try:
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        vectors = np.load(VECTORS_PATH)
    except (OSError, ValueError):
        return None, None
    if vectors.shape[0] != len(index.get("files", [])):
        return None, None
    return index, vectors


@torch.inference_mode()
def embed_paths(paths: list[Path], model, preprocess, device: str) -> tuple[list[int], np.ndarray]:
    """Return (indices of paths that loaded, their normalised vectors)."""
    ok: list[int] = []
    chunks: list[np.ndarray] = []
    for start in range(0, len(paths), BATCH_SIZE):
        tensors, batch_ok = [], []
        for i in range(start, min(start + BATCH_SIZE, len(paths))):
            try:
                with Image.open(paths[i]) as img:
                    tensors.append(preprocess(pad_to_square(img.convert("RGB"))))
                batch_ok.append(i)
            except Exception as exc:  # corrupt or unreadable image
                print(f"  skipped {paths[i].name}: {exc}", file=sys.stderr)
        if not tensors:
            continue
        feats = model.encode_image(torch.stack(tensors).to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        chunks.append(feats.float().cpu().numpy())
        ok.extend(batch_ok)
        print(f"  embedded {len(ok)}/{len(paths)}")
    dim = chunks[0].shape[1] if chunks else 0
    return ok, (np.concatenate(chunks) if chunks else np.zeros((0, dim), np.float32))


def build_cache(folder: Path, recursive: bool = False) -> None:
    folder = folder.resolve()
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    paths = list_images(folder, recursive)
    if not paths:
        sys.exit(f"No images ({', '.join(sorted(IMAGE_EXTS))}) in {folder}")

    index, vectors = load_cache()
    settings = {"model": MODEL_NAME, "pretrained": PRETRAINED, "preprocess": PREPROCESS}
    if index is not None:
        stale = [k for k, v in settings.items() if index.get(k) != v]
        if stale:
            print(f"Cache settings changed ({', '.join(stale)}); rebuilding from scratch.")
            index = vectors = None
        elif index.get("folder") != str(folder):
            print("Cache was built for a different folder; rebuilding from scratch.")
            index = vectors = None

    old_rows = {}
    if index is not None:
        old_rows = {(f["name"], f["size"], f["mtime_ns"]): i for i, f in enumerate(index["files"])}

    keys = [file_key(p, folder) for p in paths]
    todo = [i for i, k in enumerate(keys) if (k["name"], k["size"], k["mtime_ns"]) not in old_rows]
    print(f"{len(paths)} images in folder: {len(paths) - len(todo)} cached, {len(todo)} to embed.")

    new_vecs: dict[int, np.ndarray] = {}
    if todo:
        device = pick_device()
        print(f"Loading {MODEL_NAME} ({PRETRAINED}) on {device}...")
        model, preprocess, _ = load_model(device)
        ok, vecs = embed_paths([paths[i] for i in todo], model, preprocess, device)
        new_vecs = {todo[j]: vecs[n] for n, j in enumerate(ok)}

    kept_keys, kept_vecs = [], []
    for i, k in enumerate(keys):
        row = old_rows.get((k["name"], k["size"], k["mtime_ns"]))
        if row is not None:
            kept_keys.append(k)
            kept_vecs.append(vectors[row])
        elif i in new_vecs:
            kept_keys.append(k)
            kept_vecs.append(new_vecs[i])
    if not kept_vecs:
        sys.exit("No images could be embedded.")

    CACHE_DIR.mkdir(exist_ok=True)
    matrix = np.stack(kept_vecs).astype(np.float32)
    np.save(VECTORS_PATH, matrix)
    INDEX_PATH.write_text(
        json.dumps({**settings, "dim": int(matrix.shape[1]), "folder": str(folder), "files": kept_keys}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {matrix.shape[0]} vectors ({matrix.shape[1]}-d) to {CACHE_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True, type=Path, help="folder of images to embed")
    ap.add_argument("--recursive", action="store_true", help="also scan subfolders")
    args = ap.parse_args()
    build_cache(args.folder, args.recursive)
