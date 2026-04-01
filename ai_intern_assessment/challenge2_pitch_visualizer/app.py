"""
app.py — Pitch Visualizer
--------------------------
Flask web application that turns a narrative text block into a
multi-panel AI-generated storyboard.

Run:
  python app.py
Then open http://localhost:5000
"""

import os
import traceback
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from segmenter import segment_narrative
from prompt_engineer import engineer_prompt
from image_generator import generate_image

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ──────────────────────────────────────────────────────────────────────────────
# Visual style presets (Bonus: User-selectable styles)
# ──────────────────────────────────────────────────────────────────────────────

STYLES = {
    "photorealistic": "cinematic photorealism, golden hour lighting, 8K, ultra-detailed",
    "watercolor":     "delicate watercolor illustration, soft washes, pastel tones, hand-painted",
    "comic_book":     "bold comic book art, thick ink outlines, dynamic perspective, vivid flat colors",
    "scifi":          "sci-fi concept art, dramatic neon lighting, futuristic atmosphere, matte painting",
    "oil_painting":   "impressionist oil painting, rich textured brushstrokes, warm palette, museum quality",
}

STYLE_LABELS = {
    "photorealistic": "📷 Photorealistic",
    "watercolor":     "🎨 Watercolor",
    "comic_book":     "💥 Comic Book",
    "scifi":          "🚀 Sci-Fi Concept Art",
    "oil_painting":   "🖼  Oil Painting",
}


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", styles=STYLE_LABELS)


@app.route("/generate", methods=["POST"])
def generate():
    """
    POST JSON body: { "text": "...", "style": "photorealistic" }
    Returns JSON with panels list: [{ scene, enhanced_prompt, image_url }, ...]
    Streams errors gracefully.
    """
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    style_key = data.get("style", "photorealistic")

    if not text:
        return jsonify({"error": "No text provided."}), 400

    style_suffix = STYLES.get(style_key, STYLES["photorealistic"])
    style_label = STYLE_LABELS.get(style_key, "Photorealistic")

    # --- Step 1: Segment narrative
    scenes = segment_narrative(text)
    if len(scenes) < 1:
        return jsonify({"error": "Could not extract meaningful sentences from the text."}), 400

    # --- Step 2 & 3: Engineer prompts and generate images
    panels = []
    errors = []

    for i, scene in enumerate(scenes):
        panel = {"scene": scene, "panel_num": i + 1}
        try:
            enhanced = engineer_prompt(scene, style_suffix)
            panel["enhanced_prompt"] = enhanced
        except Exception as e:
            panel["enhanced_prompt"] = f"{scene}, {style_suffix}"  # fallback
            errors.append(f"Panel {i+1} prompt error: {e}")

        try:
            img_path = generate_image(panel["enhanced_prompt"])
            # Convert filesystem path to URL path
            panel["image_url"] = "/" + img_path.replace("\\", "/")
        except Exception as e:
            panel["image_url"] = None
            errors.append(f"Panel {i+1} image error: {e}")

        panels.append(panel)

    return jsonify({
        "panels": panels,
        "style_label": style_label,
        "errors": errors,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Serve static images
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/static/images/<filename>")
def serve_image(filename):
    from flask import send_from_directory
    return send_from_directory("static/images", filename)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("static/images", exist_ok=True)
    app.run(debug=True, port=5000)
