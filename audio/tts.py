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
from gtts import gTTS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class TTSHandler:
    def __init__(self):
        # Initialize the Pygame audio mixer
        pygame.mixer.init()
        self.supported_langs = {"en", "hi", "ml"}

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
            
        filename = f"dynamic_resp_{int(time.time())}.mp3"
        try:
            logging.info(f"Generating voice response (Language: {lang})...")
            # Create gTTS object with dynamically passed language
            tts = gTTS(text=text, lang=lang)
            tts.save(filename)
            
            logging.info("Playing response audio...")
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Wait until playback completes seamlessly
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            logging.error(f"TTS Engine Error: {e}")
        finally:
            # Unload music so the file is freed and can be deleted on Windows
            pygame.mixer.music.unload()
            # Give audio device extra time to reset after output playback
            time.sleep(0.5) # Increased from 0.1 to allow full device reset
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    tts = TTSHandler()
    tts.speak("Hello, I am testing the audio system natively.", lang="en")
