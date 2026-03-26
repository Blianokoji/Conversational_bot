import os
import tempfile
import logging

from elevenlabs.client import ElevenLabs


class ElevenLabsProvider:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        self.model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip()
        self.output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128").strip()

        self.default_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
        self.voice_map = {
            "en": os.getenv("ELEVENLABS_VOICE_ID_EN", "").strip(),
            "hi": os.getenv("ELEVENLABS_VOICE_ID_HI", "").strip(),
            "ml": os.getenv("ELEVENLABS_VOICE_ID_ML", "").strip(),
        }

        self.client = ElevenLabs(api_key=self.api_key) if self.api_key else None

    @property
    def is_available(self) -> bool:
        return bool(self.client and (self.default_voice_id or any(self.voice_map.values())))

    def _voice_id_for_lang(self, lang: str) -> str:
        voice_id = self.voice_map.get(lang) or self.default_voice_id
        if not voice_id:
            raise ValueError("No ElevenLabs voice ID configured. Set ELEVENLABS_VOICE_ID or per-language voice IDs.")
        return voice_id

    def synthesize_to_file(self, text: str, lang: str) -> str:
        if not self.client:
            raise RuntimeError("ELEVENLABS_API_KEY is missing.")

        voice_id = self._voice_id_for_lang(lang)

        audio_stream = self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=self.model_id,
            output_format=self.output_format,
        )

        if isinstance(audio_stream, (bytes, bytearray)):
            audio_bytes = bytes(audio_stream)
        else:
            chunks = []
            for chunk in audio_stream:
                if isinstance(chunk, (bytes, bytearray)):
                    chunks.append(bytes(chunk))
            audio_bytes = b"".join(chunks)

        if not audio_bytes:
            raise RuntimeError("ElevenLabs returned empty audio payload.")

        fd, temp_path = tempfile.mkstemp(prefix="dynamic_resp_", suffix=".mp3")
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        logging.info(f"Generated ElevenLabs audio (lang={lang}, model={self.model_id}).")
        return temp_path
