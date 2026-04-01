# AI Intern Assessment

This repo contains full solutions for both assessment challenges.

---

## Project Structure

```
ai_intern_assessment/
├── README.md                        ← You are here
├── challenge1_empathy_engine/       ← Challenge 1
│   ├── app.py                       (FastAPI web app + CLI)
│   ├── emotion_detector.py          (Transformer-based emotion classifier)
│   ├── tts_engine.py                (gTTS + pydub audio modulator)
│   ├── requirements.txt
│   └── templates/index.html        (Web UI)
└── challenge2_pitch_visualizer/    ← Challenge 2
    ├── app.py                       (Flask web app)
    ├── segmenter.py                 (NLTK narrative segmenter)
    ├── prompt_engineer.py           (Claude-powered prompt enhancer)
    ├── image_generator.py           (DALL-E 3 image generator)
    ├── requirements.txt
    └── templates/
        ├── index.html               (Input form UI)
        └── storyboard.html          (Final storyboard output)
```

---

## Challenge 1 - The Empathy Engine

### What it does
- Accepts text input via **CLI** or a **FastAPI web UI**
- Classifies emotion using `j-hartmann/emotion-english-distilroberta-base` (7 classes: joy, anger, sadness, fear, surprise, disgust, neutral)
- Modulates **rate**, **pitch**, and **volume** of synthesized speech proportionally to emotion intensity
- Outputs an `.mp3` audio file and plays it in the browser

### Setup

```bash
cd challenge1_empathy_engine
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **First run** will download the ~300MB emotion model automatically.  
> Ensure `ffmpeg` is installed on your system (`brew install ffmpeg` / `sudo apt install ffmpeg`).

### Run (Web UI)

```bash
uvicorn app:app --reload --port 8000
```
Open http://localhost:8000

### Run (CLI)

```bash
python app.py --cli --text "I just got promoted! This is the best day of my life!"
```

The output `.mp3` will be saved to `outputs/` and the emotion analysis printed to console.

### Emotion → Voice Mapping

| Emotion   | Rate   | Pitch (semitones) | Volume (dB) |
|-----------|--------|-------------------|-------------|
| Joy       | ×1.20  | +3                | +3          |
| Anger     | ×1.10  | −2                | +4          |
| Sadness   | ×0.85  | −3                | −3          |
| Fear      | ×1.15  | +2                | +1          |
| Surprise  | ×1.10  | +4                | +3          |
| Disgust   | ×0.90  | −2                | −1          |
| Neutral   | ×1.00  |  0                |  0          |

**Intensity Scaling (Bonus):** All parameters are scaled by the model's confidence score, so "This is good" produces a mild modulation while "This is AMAZING!!!" produces a much stronger effect.

---

## Challenge 2 — The Pitch Visualizer

### What it does
- Accepts a block of narrative text via a **Flask web UI**
- Segments it into scenes using NLTK sentence tokenizer
- Sends each sentence to **Claude (claude-sonnet-4-20250514)** to generate a rich, visually descriptive image prompt
- Calls **DALL-E 3** to generate a unique image per scene
- Renders a polished **HTML storyboard** with images + captions + prompts
- Supports user-selectable visual styles (Bonus)

### Prerequisites — API Keys

You need two API keys. Create a `.env` file in `challenge2_pitch_visualizer/`:

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Setup

```bash
cd challenge2_pitch_visualizer
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Run

```bash
python app.py
```
Open http://localhost:5000

### Design Choices

**Narrative Segmentation:** NLTK `sent_tokenize` splits the input at natural sentence boundaries, ensuring each panel represents a complete narrative beat.

**LLM Prompt Engineering (Bonus):** Each sentence is passed to Claude with a system prompt instructing it to return a cinematographic, visually detailed image generation prompt. This adds lighting, mood, composition, and art style details that DALL-E 3 uses to produce dramatic imagery.

**Visual Consistency (Bonus):** Every engineered prompt is appended with the user-selected style suffix (e.g., `"cinematic photorealism, golden hour lighting, 8K"`) ensuring visual coherence across panels.

**User-Selectable Styles (Bonus):** The UI offers 5 preset styles — Photorealistic, Watercolor, Comic Book, Sci-Fi Concept Art, and Oil Painting.

---

## Requirements Summary

| Challenge | Key Libraries |
|-----------|--------------|
| 1         | `transformers`, `torch`, `gtts`, `pydub`, `fastapi`, `uvicorn` |
| 2         | `flask`, `anthropic`, `openai`, `nltk`, `python-dotenv`, `requests` |
