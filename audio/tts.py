import os
import time
import warnings
import logging

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import pygame

from audio.gtts_tts import GTTSProvider
from audio.elevenlabs_tts import ElevenLabsProvider

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class TTSHandler:
    def __init__(self):
        # Initialize the Pygame audio mixer
        pygame.mixer.init()
        self.supported_langs = {"en", "hi", "ml"}
        self.provider = os.getenv("TTS_PROVIDER", "gtts").strip().lower()

        self.gtts = GTTSProvider()
        self.elevenlabs = ElevenLabsProvider()

        if self.provider not in {"gtts", "elevenlabs"}:
            logging.warning(f"Unknown TTS_PROVIDER '{self.provider}', defaulting to gtts.")
            self.provider = "gtts"

        if self.provider == "elevenlabs" and not self.elevenlabs.is_available:
            logging.warning("ElevenLabs is selected but not fully configured. Falling back to gTTS.")
            self.provider = "gtts"

    def _play_file(self, filename: str):
        logging.info("Playing response audio...")
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def _synthesize_with_provider(self, text: str, lang: str) -> str:
        if self.provider == "elevenlabs":
            logging.info(f"Generating voice response with ElevenLabs (Language: {lang})...")
            return self.elevenlabs.synthesize_to_file(text, lang)

        logging.info(f"Generating voice response with gTTS (Language: {lang})...")
        return self.gtts.synthesize_to_file(text, lang)

    def speak(self, text: str, lang: str = 'en'):
        """
        Generates TTS audio and plays it back. 
        Accepts a `lang` parameter to match the user's spoken language.
        """
        if not text:
            return

        if lang not in self.supported_langs:
            logging.warning(f"Unsupported TTS language '{lang}', falling back to English.")
            lang = "en"

        filename = None
        try:
            filename = self._synthesize_with_provider(text, lang)
            self._play_file(filename)

        except Exception as e:
            logging.error(f"Primary TTS provider failed: {e}")

            # Automatic fallback path to gTTS if ElevenLabs fails at runtime.
            if self.provider == "elevenlabs":
                try:
                    logging.info("Falling back to gTTS...")
                    filename = self.gtts.synthesize_to_file(text, lang)
                    self._play_file(filename)
                    return
                except Exception as fallback_error:
                    logging.error(f"Fallback gTTS failed: {fallback_error}")
        finally:
            # Unload music so the file is freed and can be deleted on Windows
            if pygame.mixer.get_init():
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass

            # Give audio device extra time to reset after output playback
            time.sleep(0.5)
            if filename and os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    tts = TTSHandler()
    tts.speak("Hello, I am testing the audio system natively.", lang="en")
