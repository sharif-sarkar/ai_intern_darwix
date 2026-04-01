"""
tts_engine.py
-------------
Generates emotionally-modulated speech audio from text.

Pipeline:
  1. gTTS generates a base MP3 from the input text.
  2. pydub applies:
       - Volume adjustment   (dB gain)
       - Pitch shift         (frame-rate trick, then resample)
       - Rate/tempo change   (speedup / slowdown)
  3. Final audio is exported as MP3.

Why the frame-rate pitch trick?
  pydub does not include a dedicated pitch-shift DSP, but changing the
  underlying sample-rate before resampling to the original rate achieves
  the same effect cheaply and without extra dependencies.
"""

import os
import tempfile
from pathlib import Path

from gtts import gTTS
from pydub import AudioSegment
import imageio_ffmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


class TTSEngine:
    """Converts text to emotionally-modulated speech."""

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        text: str,
        vocal_params: dict,
        output_path: str,
        lang: str = "en",
    ) -> str:
        """
        Generate speech for `text`, apply `vocal_params`, save to `output_path`.

        vocal_params keys:
          rate       (float) – playback speed multiplier, e.g. 1.2 = 20% faster
          pitch      (float) – semitone shift, e.g. +3 or -2
          volume_db  (float) – dB gain, e.g. +3 or -3

        Returns the absolute path to the saved MP3 file.
        """
        # Step 1 – generate base TTS audio via gTTS
        base_audio = self._generate_gtts(text, lang)

        # Step 2 – apply modulations
        audio = self._apply_volume(base_audio, vocal_params.get("volume_db", 0.0))
        audio = self._apply_pitch(audio, vocal_params.get("pitch", 0.0))
        audio = self._apply_rate(audio, vocal_params.get("rate", 1.0))

        # Step 3 – export
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        audio.export(output_path, format="mp3")
        return output_path

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_gtts(text: str, lang: str) -> AudioSegment:
        """Use gTTS to produce a base AudioSegment."""
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            tts.save(tmp_path)
            audio = AudioSegment.from_mp3(tmp_path)
        finally:
            os.unlink(tmp_path)
        return audio

    @staticmethod
    def _apply_volume(audio: AudioSegment, db: float) -> AudioSegment:
        """Adjust loudness by `db` decibels."""
        if abs(db) < 0.01:
            return audio
        return audio + db

    @staticmethod
    def _apply_pitch(audio: AudioSegment, semitones: float) -> AudioSegment:
        """
        Shift pitch by `semitones` via frame-rate resampling.

        Raising the frame rate before spawning, then re-setting to the
        original rate, achieves a pitch shift without altering duration.
        """
        if abs(semitones) < 0.1:
            return audio
        factor = 2 ** (semitones / 12.0)
        new_frame_rate = int(audio.frame_rate * factor)
        shifted = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": new_frame_rate},
        )
        # Resample back to original frame rate so duration is unchanged
        return shifted.set_frame_rate(audio.frame_rate)

    @staticmethod
    def _apply_rate(audio: AudioSegment, rate: float) -> AudioSegment:
        """Speed up or slow down playback by `rate` multiplier."""
        if abs(rate - 1.0) < 0.02:
            return audio
        if rate > 1.0:
            return audio.speedup(playback_speed=rate)
        # For slowing down, use frame-rate trick (inverse of speedup)
        slow_factor = 1.0 / rate
        new_frame_rate = int(audio.frame_rate / slow_factor)
        slowed = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": new_frame_rate},
        )
        return slowed.set_frame_rate(audio.frame_rate)


# --- Quick self-test ---
if __name__ == "__main__":
    engine = TTSEngine()
    params = {"rate": 1.2, "pitch": 3.0, "volume_db": 3.0}
    out = engine.synthesize(
        "This is the best news I have heard all year!",
        params,
        "/tmp/test_happy.mp3",
    )
    print(f"Saved to: {out}")
