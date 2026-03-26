import time
import numpy as np
import sounddevice as sd
import whisper
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BufferedSTT:
    def __init__(self, model_size="base", energy_threshold=0.015, silence_chunks=2, chunk_duration=2.0):
        """
        Initializes the Whisper model and pseudo-streaming buffer parameters.
        """
        self.sample_rate = 16000 # Whisper expects 16kHz
        self.chunk_duration = chunk_duration
        self.energy_threshold = energy_threshold
        self.silence_limit = silence_chunks # Number of consecutive silent chunks to trigger transcription
        self.min_voice_chunks = 1 if self.chunk_duration >= 2.0 else 2
        self.allowed_languages = {"en", "hi", "ml"}
        self.domain_prompt = (
            "Transight, telematics, IoT, asset tracking, compliance, "
            "fleet management, cold chain management"
        )
        
        # Calibrate noise floor to adjust threshold dynamically
        self.calibrate_noise_floor()

        logging.info(f"Loading Whisper model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        logging.info("Whisper model loaded.")

    def calibrate_noise_floor(self, duration=3):
        """Measures background noise to set an adaptive energy threshold."""
        print(f"\n🎧 Calibrating microphone for {duration} seconds... (Please remain silent)")
        
        try:
            # Warmup: Do a quick recording to initialize the audio device
            _ = sd.rec(int(0.5 * self.sample_rate), 
                       samplerate=self.sample_rate, 
                       channels=1, 
                       dtype='float32')
            sd.wait()
            time.sleep(0.3)  # Brief pause after warmup
            
            # Record a chunk to measure noise
            recording = sd.rec(int(duration * self.sample_rate), 
                               samplerate=self.sample_rate, 
                               channels=1, 
                               dtype='float32')
            sd.wait()
            recording = recording.flatten()
            
            # Calculate RMS energy of background noise
            noise_floor = np.sqrt(np.mean(recording**2))
            
            # Set threshold to 50% above noise floor, but enforce a sane minimum
            # If noise is 0.001, threshold -> 0.0015
            # If noise is 0.0 (digital silence), threshold -> 0.005 (default min)
            self.energy_threshold = max(noise_floor * 1.5, 0.005)
            
            print(f"✅ Calibration complete. Noise Floor: {noise_floor:.5f}")
            print(f"👉 Adaptive Energy Threshold set to: {self.energy_threshold:.5f}\n")
            
        except Exception as e:
            print(f"⚠️ Calibration failed: {e}")
            print(f"👉 Using default energy threshold: {self.energy_threshold}")

    def record_until_silence(self) -> np.ndarray:
        """
        Records fixed-duration chunks until `silence_limit` consecutive silent chunks are detected.
        Returns the accumulated audio buffer as a single 16kHz float32 numpy array.
        """
        buffer = []
        silent_count = 0
        started_speaking = False
        consecutive_silence = 0
        voice_chunks = 0
        
        print("\n\n" + "="*40)
        print("🎤 Listening... (Speak now)")
        print("="*40 + "\n")
        
        try:
            while True:
                # Record one fixed-duration chunk
                chunk = sd.rec(int(self.chunk_duration * self.sample_rate), 
                               samplerate=self.sample_rate, 
                               channels=1, 
                               dtype='float32')
                sd.wait()
                chunk = chunk.flatten()
                
                # Calculate Root Mean Square (RMS) energy to detect voice activity
                energy = np.sqrt(np.mean(chunk**2))
                print(f"\rEnergy: {energy:.4f} / Threshold: {self.energy_threshold:.4f}", end="", flush=True)
                
                # Sanity check for dead microphone
                if energy == 0.0:
                    consecutive_silence += 1
                    if consecutive_silence > 5:
                        print("\n⚠️ WARNING: Microphone is returning absolute silence (0.0). Check your input device settings!")
                else:
                    consecutive_silence = 0
    
                # Dynamic voice activity detection
                if energy > self.energy_threshold:
                    if not started_speaking:
                        print("\n", end="", flush=True)  # Newline before logging
                        logging.info(f"Voice detected, buffering {self.chunk_duration:.1f}-second chunks...")
                        started_speaking = True
                    silent_count = 0
                    voice_chunks += 1
                    buffer.append(chunk)
                else:
                    if started_speaking:
                        silent_count += 1
                        buffer.append(chunk) # Keep a bit of trailing silence for natural padding
                        if silent_count >= self.silence_limit and voice_chunks >= self.min_voice_chunks:
                            print("\n", end="", flush=True)  # Newline before logging
                            logging.info("Silence detected. Processing the buffered audio...")
                            break
                    # If we haven't started speaking, discard semantic-less silence and keep waiting.
    
            # Concatenate all chunks into a single numpy array for Whisper
            return np.concatenate(buffer)
            
        except Exception as e:
            print(f"\n❌ Error during audio recording: {e}")
            logging.error(f"Recording error: {e}")
            raise

    def _choose_allowed_language(self, audio_data: np.ndarray) -> str:
        """Detect language and constrain it to en/hi/ml with English fallback."""
        try:
            audio = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            # Prefer English unless supported alternatives are clearly more likely.
            hi_prob = probs.get("hi", 0.0)
            ml_prob = probs.get("ml", 0.0)
            en_prob = probs.get("en", 0.0)

            best_alt_code, best_alt_prob = ("hi", hi_prob) if hi_prob >= ml_prob else ("ml", ml_prob)
            if best_alt_prob >= 0.60 and best_alt_prob > en_prob:
                return best_alt_code
            return "en"
        except Exception as e:
            logging.warning(f"Language detection fallback to English: {e}")
        return "en"

    def transcribe(self) -> tuple[str, str, str]:
        """
        Records buffered audio and transcribes it.
        Returns:
        - spoken_text: transcript in the user's spoken language
        - lang_code: detected language code (en/hi/ml)
        - llm_text_en: English text for retrieval/LLM context
        """
        try:
            audio_data = self.record_until_silence()
            lang = self._choose_allowed_language(audio_data)
            
            logging.info("Transcribing audio with Whisper...")
            # Force supported languages and disable fp16 on CPU to prevent warning noise.
            result_native = self.model.transcribe(
                audio_data,
                language=lang,
                task="transcribe",
                fp16=False,
                temperature=0.0,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
                no_speech_threshold=0.45,
                initial_prompt=self.domain_prompt,
            )
            
            spoken_text = result_native.get('text', '').strip()
            spoken_text = self._post_process_transcript(spoken_text)
            lang = result_native.get('language', lang)
            if lang not in self.allowed_languages:
                lang = "en"

            llm_text_en = spoken_text
            if lang != "en" and spoken_text:
                # Translate non-English speech to English before retrieval/embedding.
                result_en = self.model.transcribe(
                    audio_data,
                    language=lang,
                    task="translate",
                    fp16=False,
                    temperature=0.0,
                    beam_size=5,
                    best_of=5,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.45,
                    initial_prompt=self.domain_prompt,
                )
                llm_text_en = result_en.get('text', '').strip()
                llm_text_en = self._post_process_transcript(llm_text_en)
            
            if spoken_text:
                logging.info(f"🗣️ User ({lang}): {spoken_text}")
                if lang != "en" and llm_text_en:
                    logging.info(f"🌐 English for LLM: {llm_text_en}")
            else:
                logging.info("⚠️ No speech detected in audio.")
                
            # Add a small delay to allow microphone device to reset before next recording
            time.sleep(0.5)
            
            return spoken_text, lang, llm_text_en
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            raise

    def _post_process_transcript(self, text: str) -> str:
        """Normalize frequent STT substitutions for key brand and domain terms."""
        replacements = {
            " transit ": " Transight ",
            " transite ": " Transight ",
            " tran site ": " Transight ",
            " climatics ": " telematics ",
            " iot ": " IoT ",
        }

        normalized = f" {text} "
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
            normalized = normalized.replace(old.title(), new)
        return normalized.strip()

if __name__ == "__main__":
    stt = BufferedSTT()
    stt.transcribe()
