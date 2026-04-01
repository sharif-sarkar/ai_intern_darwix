"""
emotion_detector.py
-------------------
Detects emotion from input text using a pre-trained transformer model.
Model: j-hartmann/emotion-english-distilroberta-base
Supports 7 classes: joy, anger, sadness, fear, surprise, disgust, neutral

Bonus: Intensity scaling — confidence score scales the modulation strength.
"""

from transformers import pipeline


# --- Emotion → Vocal Parameter Mapping ---
# rate   : playback speed multiplier (1.0 = normal)
# pitch  : semitone shift (positive = higher, negative = lower)
# volume : dB gain (+positive = louder, -negative = quieter)

EMOTION_VOICE_MAP = {
    "joy":      {"rate": 1.20, "pitch":  3.0, "volume_db":  3.0},
    "anger":    {"rate": 1.10, "pitch": -2.0, "volume_db":  4.0},
    "sadness":  {"rate": 0.85, "pitch": -3.0, "volume_db": -3.0},
    "fear":     {"rate": 1.15, "pitch":  2.0, "volume_db":  1.0},
    "surprise": {"rate": 1.10, "pitch":  4.0, "volume_db":  3.0},
    "disgust":  {"rate": 0.90, "pitch": -2.0, "volume_db": -1.0},
    "neutral":  {"rate": 1.00, "pitch":  0.0, "volume_db":  0.0},
}


class EmotionDetector:
    """
    Wraps the HuggingFace emotion classification pipeline and applies
    intensity-scaled vocal parameter modulation.
    """

    def __init__(self):
        print("Loading emotion model (first run may take a moment)...")
        self._classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        print("Emotion model ready.")

    def detect(self, text: str) -> dict:
        """
        Analyse `text` and return a result dict containing:
          - emotion       : top predicted emotion label (str)
          - confidence    : probability of the top emotion (float 0–1)
          - vocal_params  : intensity-scaled voice parameters (dict)
          - all_emotions  : full probability distribution (dict)
        """
        raw = self._classifier(text)[0]  # list of {label, score}
        sorted_results = sorted(raw, key=lambda x: x["score"], reverse=True)

        top = sorted_results[0]
        emotion = top["label"].lower()
        confidence = top["score"]

        # --- Intensity Scaling (Bonus) ---
        # Map confidence [0, 1] → intensity scale [0.4, 1.0]
        # Low-confidence detections produce subtle modulation;
        # high-confidence detections produce full modulation.
        intensity = 0.4 + confidence * 0.6

        base = EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP["neutral"])
        vocal_params = {
            "rate":      1.0 + (base["rate"] - 1.0) * intensity,
            "pitch":     base["pitch"] * intensity,
            "volume_db": base["volume_db"] * intensity,
        }

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "vocal_params": {k: round(v, 3) for k, v in vocal_params.items()},
            "all_emotions": {r["label"].lower(): round(r["score"], 4) for r in sorted_results},
        }


# --- Quick self-test ---
if __name__ == "__main__":
    detector = EmotionDetector()
    samples = [
        "I just got promoted! This is the best day of my life!",
        "I can't believe they lied to me. I'm absolutely furious.",
        "The meeting is scheduled for Tuesday at 3 PM.",
        "I'm so scared. I don't know what's going to happen.",
    ]
    for text in samples:
        result = detector.detect(text)
        print(f"\nText     : {text}")
        print(f"Emotion  : {result['emotion']} ({result['confidence']*100:.1f}% confident)")
        print(f"Params   : {result['vocal_params']}")
