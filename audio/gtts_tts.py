import os
import time
import tempfile

from gtts import gTTS


class GTTSProvider:
    def __init__(self):
        self.supported_langs = {"en", "hi", "ml"}

    def synthesize_to_file(self, text: str, lang: str) -> str:
        if lang not in self.supported_langs:
            lang = "en"

        fd, temp_path = tempfile.mkstemp(prefix=f"dynamic_resp_{int(time.time())}_", suffix=".mp3")
        os.close(fd)

        tts = gTTS(text=text, lang=lang)
        tts.save(temp_path)
        return temp_path
