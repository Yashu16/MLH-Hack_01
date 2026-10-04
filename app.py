"""Gradio UI: text query -> top 12 similar images, plus 'find similar' on a selected image."""
import hashlib
from pathlib import Path

import gradio as gr
from PIL import Image

from embed import CACHE_DIR
from search import TOP_K, ImageSearch

THUMB_DIR = CACHE_DIR / "thumbs"
THUMB_WIDTH = 480
EXAMPLES = [
    "a character standing near water",
    "a menu screen with text",
    "a night sky with stars",
    "a battle with enemies",
]

engine = ImageSearch()


def thumb_for(path: Path) -> Path:
    """Small JPEG copy of `path`, created once and reused (keyed by name, size, mtime)."""
    st = path.stat()
    key = hashlib.sha1(f"{path.name}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    out = THUMB_DIR / f"{key}.jpg"
    if not out.exists():
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 10))
            img.save(out, "JPEG", quality=85)
    return out


def show(hits):
    """(path, score) hits -> (gallery items, original paths, scores).

    Captions are kept short ("(35) · 0.33") so they aren't truncated on narrow thumbnails;
    the full file name is shown in the status line when an image is selected.
    """
    items = [(str(thumb_for(p)), f"{p.stem.removeprefix('Screenshot ')} · {s:.2f}") for p, s in hits]
    return items, [str(p) for p, _ in hits], [s for _, s in hits]


def run_text(query):
    hits = engine.by_text(query)
    if not hits:
        gr.Warning("Type something to search for.")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    items, paths, scores = show(hits)
    text = {"query": query, "items": items, "paths": paths, "scores": scores}
    state = {"paths": paths, "scores": scores, "selected": None, "text": text}
    return items, state, f"Top {len(items)} for “{query.strip()}”", None


def on_select(evt: gr.SelectData, state):
    path = state["paths"][evt.index]
    score = state["scores"][evt.index]
    state = {**state, "selected": path}
    return path, state, f"Selected **{Path(path).name}** (score {score:.3f}). Press “Find similar”."


def run_similar(state):
    if not state or not state.get("selected"):
        gr.Warning("Click an image in the gallery first.")
        return gr.skip(), gr.skip(), gr.skip()
    chosen = state["selected"]
    items, paths, scores = show(engine.by_image(chosen))
    state = {**state, "paths": paths, "scores": scores, "selected": None}
    return items, state, f"Top {len(items)} similar to {Path(chosen).name}"


def back_to_text(state):
    saved = (state or {}).get("text")
    if not saved:
        gr.Warning("No text results to go back to yet.")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    state = {**state, "paths": saved["paths"], "scores": saved["scores"], "selected": None}
    return saved["items"], state, f"Top {len(saved['items'])} for “{saved['query'].strip()}”", None


with gr.Blocks(title="Photo Search") as demo:
    gr.Markdown(f"# Photo search\nSearch {len(engine.paths)} images by text, or find look-alikes of one.")
    state = gr.State({"paths": [], "scores": [], "selected": None, "text": None})
    with gr.Row():
        query = gr.Textbox(placeholder="Describe what you're looking for…", show_label=False, scale=5)
        search_btn = gr.Button("Search", variant="primary", scale=1)
    status = gr.Markdown()
    gallery = gr.Gallery(label=f"Top {TOP_K}", columns=4, rows=3, height="auto", allow_preview=False, object_fit="contain")
    with gr.Row():
        selected_img = gr.Image(label="Selected (full size)", interactive=False, height=360)
        with gr.Column():
            similar_btn = gr.Button("Find similar", variant="primary")
            back_btn = gr.Button("Back to text results")

    run_outputs = [gallery, state, status, selected_img]
    search_btn.click(run_text, [query], run_outputs, api_name="search")
    query.submit(run_text, [query], run_outputs)
    gr.Examples(EXAMPLES, inputs=query, fn=run_text, outputs=run_outputs, run_on_click=True, cache_examples=False)
    gallery.select(on_select, [state], [selected_img, state, status])
    similar_btn.click(run_similar, [state], [gallery, state, status], api_name="similar")
    back_btn.click(back_to_text, [state], run_outputs)


if __name__ == "__main__":
    print(f"Preparing thumbnails for {len(engine.paths)} images...")
    for p in engine.paths:
        thumb_for(p)
    demo.launch(allowed_paths=[str(engine.folder), str(THUMB_DIR.resolve())])
