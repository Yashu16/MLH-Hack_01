"""Query the cached image vectors: text -> top-k images, or image -> similar images."""
import json
from pathlib import Path

import numpy as np
import torch

from embed import (
    INDEX_PATH, MODEL_NAME, PREPROCESS, PRETRAINED, VECTORS_PATH, load_model, pick_device,
)

# Change this one line to alter how text queries are phrased for the model,
# e.g. "a screenshot of {query}". "{query}" is replaced by what the user typed.
PROMPT_TEMPLATE = "{query}"

TOP_K = 12


class ImageSearch:
    def __init__(self) -> None:
        if not (INDEX_PATH.exists() and VECTORS_PATH.exists()):
            raise FileNotFoundError("No cache found. Run: python embed.py --folder <your image folder>")
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        expected = {"model": MODEL_NAME, "pretrained": PRETRAINED, "preprocess": PREPROCESS}
        stale = [k for k, v in expected.items() if index.get(k) != v]
        if stale:
            raise RuntimeError(
                f"Cache was built with different {', '.join(stale)}. Re-run embed.py to rebuild it."
            )
        self.folder = Path(index["folder"])
        self.names: list[str] = [f["name"] for f in index["files"]]
        self.paths: list[Path] = [self.folder / n for n in self.names]
        self.vectors = np.load(VECTORS_PATH)
        self.device = pick_device()
        self.model, _, self.tokenizer = load_model(self.device)

    def _top_k(self, scores: np.ndarray, k: int, exclude: int | None = None) -> list[tuple[Path, float]]:
        if exclude is not None:
            scores = scores.copy()
            scores[exclude] = -np.inf
        k = min(k, int((scores > -np.inf).sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k else []
        top = sorted(top, key=lambda i: -scores[i])
        return [(self.paths[i], float(scores[i])) for i in top]

    @torch.inference_mode()
    def by_text(self, query: str, k: int = TOP_K) -> list[tuple[Path, float]]:
        query = query.strip()
        if not query:
            return []
        tokens = self.tokenizer([PROMPT_TEMPLATE.format(query=query)]).to(self.device)
        vec = self.model.encode_text(tokens)
        vec = (vec / vec.norm(dim=-1, keepdim=True)).float().cpu().numpy()[0]
        return self._top_k(self.vectors @ vec, k)

    def by_image(self, path: str | Path, k: int = TOP_K) -> list[tuple[Path, float]]:
        """Similar images from the stored vector of `path`; the image itself is excluded."""
        path = Path(path)
        try:
            idx = self.paths.index(path)
        except ValueError:
            idx = next((i for i, p in enumerate(self.paths) if p.resolve() == path.resolve()), None)
            if idx is None:
                raise KeyError(f"{path} is not in the cache")
        return self._top_k(self.vectors @ self.vectors[idx], k, exclude=idx)


if __name__ == "__main__":
    import sys

    s = ImageSearch()
    q = " ".join(sys.argv[1:]) or "a character standing near water"
    print(f"text query: {q!r}")
    hits = s.by_text(q)
    for p, sc in hits:
        print(f"  {sc:.3f}  {p.name}")
    print(f"similar to {hits[0][0].name}:")
    for p, sc in s.by_image(hits[0][0]):
        print(f"  {sc:.3f}  {p.name}")
