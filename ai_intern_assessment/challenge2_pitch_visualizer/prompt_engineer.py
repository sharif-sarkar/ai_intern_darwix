"""
prompt_engineer.py
------------------
Uses the Anthropic Claude API to transform a plain narrative sentence
into a rich, cinematographic prompt suitable for DALL-E 3.

This is the "LLM-Powered Prompt Refinement" bonus feature.
"""

import os
import anthropic


SYSTEM_PROMPT = """You are a world-class art director and prompt engineer specialising in
AI image generation. Your task is to transform a plain narrative sentence into a
highly detailed, visually descriptive image-generation prompt.

Rules:
- Describe the scene as a professional photographer or filmmaker would frame it.
- Include: subject, action/pose, environment, lighting, mood, camera angle, depth of field.
- Do NOT include character names — describe appearance instead.
- Do NOT include text, logos, or words within the image.
- Return ONLY the prompt text — no preamble, no quotes, no explanation.
- Keep the prompt under 180 words.
"""


def engineer_prompt(sentence: str, style_suffix: str = "") -> str:
    """
    Send `sentence` to Claude and receive an enhanced image-generation prompt.

    Args:
        sentence     : One scene/sentence from the narrative.
        style_suffix : Optional style keywords appended for visual consistency,
                       e.g. "cinematic photorealism, golden hour lighting, 8K"

    Returns:
        Enhanced prompt string ready for DALL-E 3.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_msg = f'Original scene: "{sentence}"'
    if style_suffix:
        user_msg += f'\n\nAlways append this style to the end of your prompt: "{style_suffix}"'

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    enhanced = message.content[0].text.strip()
    return enhanced


# --- Quick self-test ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    sentence = "After years of saving, she finally opened her bakery on a rainy Tuesday."
    style = "warm watercolor illustration, soft pastels"
    result = engineer_prompt(sentence, style)
    print(f"Original : {sentence}")
    print(f"Enhanced : {result}")
