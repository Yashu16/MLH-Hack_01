# Photo Search

Search a folder of pictures by describing what's in them, or pick one and find the ones that look like it.

I had 220 gameplay screenshots from one game update and no good way to find a specific moment in them without scrolling through every file. This app lets me type something like *"a character standing near water"* and get the 12 closest matches, then press **Find similar** on any result to see look-alikes.

It is retrieval only. There is no text generation and no chatbot.

## Why open-source AI matters here

- **Open weights, open data.** The model is OpenCLIP ViT-B-32 with the `laion2b_s34b_b79k` weights, trained on the publicly documented LAION-2B dataset.
- **Runs on your own machine.** After the one-time weight download, nothing leaves your computer: no API key, no per-query cost, no rate limit. Your screenshots stay private.
- **Swappable.** The model name, weights and preprocessing are constants in `embed.py`. Change one and the cache rebuilds itself.

## How it works

1. `embed.py` embeds every image with OpenCLIP and saves L2-normalised vectors to `cache/vectors.npy`, with a matching `cache/index.json`.
2. `search.py` embeds a text query with the same model and ranks images by cosine similarity. For Find similar, it ranks by the selected image's stored vector and leaves that image out of its own results.
3. `app.py` is a Gradio UI: search box, gallery of the top 12 with scores, a full-size preview of the selected image, a **Find similar** button, and a **Back to text results** button.

Design choices worth knowing:

- **Letterboxing instead of cropping.** CLIP normally center-crops to a square, which drops the left and right edges of a 16:9 screenshot. Images are padded to a square instead, so the model sees the whole frame.
- **Self-invalidating cache.** `index.json` records the model, weights and preprocess name. If any differ from the code, the cache is rebuilt automatically.
- **Incremental updates.** Re-running `embed.py` only embeds new or changed files, keyed by name, size and modification time. Corrupt images are skipped with a warning.
- **Configurable prompt.** `PROMPT_TEMPLATE` in `search.py` controls how text queries are phrased, e.g. `"a screenshot of {query}"`.

## Setup

Requires Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA build of torch (omit --index-url for CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

```powershell
# 1. Embed a folder (first run downloads the model weights, about 600 MB)
python embed.py --folder "C:\path\to\your\images"      # add --recursive for subfolders

# 2. Launch the UI
python app.py                                          # opens on http://127.0.0.1:7860

# Optional: query from the command line
python search.py "a night sky with stars"
```

On an RTX 3050 laptop GPU, embedding 220 1920×1080 PNGs took about 37 seconds, including the first-run weight download. Search over 220 vectors is a single matrix-vector product.

## Limitations

- CLIP understands scenes, not game lore. "Blue sky over the ocean" works; a character's name is unreliable.
- Text-to-image scores are naturally low (around 0.25–0.35 for good matches). Image-to-image scores are much higher (around 0.8).
- Vectors are compared by brute force, which is fine for thousands of images but not millions.

## Ideas for next

- Search by dropping in a new image instead of picking from the gallery.
- A vector index if the collection grows large.

## License

MIT. See [LICENSE](LICENSE).
