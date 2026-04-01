import argparse
import json
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from emotion_detector import EmotionDetector
from tts_engine import TTSEngine

app = FastAPI(title="Empathy Engine", version="1.0")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
_detector: EmotionDetector | None = None
_engine: TTSEngine | None = None


def get_detector() -> EmotionDetector:
    global _detector
    if _detector is None:
        _detector = EmotionDetector()
    return _detector


def get_engine() -> TTSEngine:
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/synthesize")
async def synthesize(request: Request, text: str = Form(...)):
    if not text.strip():
        return JSONResponse({"error": "Text cannot be empty."}, status_code=400)

    detector = get_detector()
    engine = get_engine()

    analysis = detector.detect(text)
    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = str(OUTPUT_DIR / filename)
    engine.synthesize(text, analysis["vocal_params"], output_path)

    return JSONResponse({
        "emotion": analysis["emotion"],
        "confidence": analysis["confidence"],
        "vocal_params": analysis["vocal_params"],
        "all_emotions": analysis["all_emotions"],
        "audio_url": f"/audio/{filename}",
    })


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found."}, status_code=404)
    return FileResponse(str(path), media_type="audio/mpeg")

def run_cli(text: str, output: str = "output.mp3"):
    detector = EmotionDetector()
    engine = TTSEngine()

    print(f"\n{'='*60}")
    print(f"  Input: {text}")
    print(f"{'='*60}")

    analysis = detector.detect(text)

    print(f"\n  Detected Emotion : {analysis['emotion'].upper()}")
    print(f"  Confidence       : {analysis['confidence']*100:.1f}%")
    print(f"\n  Vocal Parameters :")
    for k, v in analysis["vocal_params"].items():
        print(f"    {k:<12} : {v}")
    print(f"\n  All Emotions:")
    for emotion, score in sorted(analysis["all_emotions"].items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 30)
        print(f"    {emotion:<10} {bar} {score*100:.1f}%")

    engine.synthesize(text, analysis["vocal_params"], output)
    print(f"\n  ✅ Audio saved to: {output}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empathy Engine")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--text", type=str, help="Input text (CLI mode)")
    parser.add_argument("--output", type=str, default="output.mp3", help="Output MP3 path")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    args = parser.parse_args()

    if args.cli:
        if not args.text:
            parser.error("--text is required in CLI mode")
        run_cli(args.text, args.output)
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=True)
