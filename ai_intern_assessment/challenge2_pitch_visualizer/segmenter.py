"""
segmenter.py
------------
Splits a block of narrative text into individual scenes/sentences
using NLTK's Punkt sentence tokenizer.

Run once to download the model:
  python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
"""

import re
import nltk


def segment_narrative(text: str, min_length: int = 10) -> list[str]:
    """
    Split `text` into a list of meaningful scene sentences.

    Args:
        text       : Input narrative paragraph(s).
        min_length : Minimum character count to keep a segment
                     (filters out fragments like "OK." or "Yes.").

    Returns:
        List of sentence strings, each representing one storyboard panel.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    sentences = nltk.sent_tokenize(text)

    # Filter very short fragments
    scenes = [s.strip() for s in sentences if len(s.strip()) >= min_length]

    return scenes


# --- Quick self-test ---
if __name__ == "__main__":
    sample = (
        "Sarah had always dreamed of running her own bakery. "
        "After years of saving, she finally opened 'Sweet Mornings' on a rainy Tuesday. "
        "The line stretched around the block before she even turned the key. "
        "By noon, every croissant was gone and strangers were hugging her. "
        "That evening she sat alone in the empty shop, laughing and crying at the same time."
    )
    scenes = segment_narrative(sample)
    for i, s in enumerate(scenes, 1):
        print(f"  Scene {i}: {s}")
