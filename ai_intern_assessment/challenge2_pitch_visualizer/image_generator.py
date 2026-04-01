"""
image_generator.py
------------------
Calls the OpenAI DALL-E 3 API to generate a 1792×1024 landscape image
from an engineered prompt, downloads it, and saves it locally.
"""

import os
import uuid
import requests
from pathlib import Path

import openai


def generate_image(prompt: str, output_dir: str = "static/images") -> str:
    """
    Generate an image for `prompt` using DALL-E 3.

    Returns the local file path of the saved PNG image.
    Raises RuntimeError on API or download failure.
    """
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1792x1024",   # widescreen — ideal for storyboard panels
        quality="standard", # use "hd" for higher quality (costs more)
        response_format="url",
    )

    image_url = response.data[0].url

    # Download and save locally so the storyboard can serve it without
    # relying on a time-limited OpenAI CDN URL
    img_data = requests.get(image_url, timeout=60).content
    filename = f"{uuid.uuid4().hex}.png"
    local_path = Path(output_dir) / filename
    local_path.write_bytes(img_data)

    return str(local_path)


# --- Quick self-test ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    prompt = (
        "A young woman with flour-dusted apron standing at the door of a cozy bakery "
        "on a rainy cobblestone street, warm amber light spilling from the window, "
        "cinematic photography, shallow depth of field, 8K"
    )
    path = generate_image(prompt)
    print(f"Image saved to: {path}")
