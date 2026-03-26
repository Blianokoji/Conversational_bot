import os
import time
import numpy as np
import sounddevice as sd
import whisper
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BufferedSTT:
    def __init__(self, model_size="base", energy_threshold=0.015, silence_chunks=2, chunk_duration=2.0, device_index=None):
        """
        Initializes the Whisper model and pseudo-streaming buffer parameters.
        """
        self.sample_rate = 16000 # Whisper expects 16kHz
        self.chunk_duration = chunk_duration
        self.energy_threshold = energy_threshold
        self.silence_limit = silence_chunks # Number of consecutive silent chunks to trigger transcription
        self.min_voice_chunks = 1 if self.chunk_duration >= 2.0 else 2
        self.allowed_languages = {"en", "hi", "ml"}
        self.device_index = device_index or os.getenv("AUDIO_DEVICE_INDEX")
        if self.device_index:
            try:
                self.device_index = int(self.device_index)
            except (ValueError, TypeError):
                self.device_index = None
        self.domain_prompt = (
            "Transight, telematics, IoT, asset tracking, compliance, fleet management, "
            "cold chain management, vehicle ECU, remote monitoring, GPS tracking, "
            "ടിലിമാറ്റിക്സ്, IoT, അസ്സെറ്റ് ട്രാക്കിംഗ്, കോൾഡ് ചെയിൻ, "
            "टेलीमैटिक्स, IoT, संपत्ति ट്यാัก्किंग"
        )
        self.confidence_threshold = float(os.getenv("WHISPER_CONFIDENCE_THRESHOLD", "0.5"))
        self.no_speech_threshold = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.65"))
        
        # Calibrate noise floor to adjust threshold dynamically
        self.calibrate_noise_floor()

        logging.info(f"Loading Whisper model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        logging.info("Whisper model loaded.")

    def calibrate_noise_floor(self, duration=3):
        """Measures background noise to set an adaptive energy threshold."""
        print(f"\n🎧 Calibrating microphone for {duration} seconds... (Please remain silent)")
        if self.device_index is not None:
            print(f"   Using audio device index: {self.device_index}")
        
        try:
            # Warmup: Do a quick recording to initialize the audio device
            _ = sd.rec(int(0.5 * self.sample_rate), 
                       samplerate=self.sample_rate, 
                       channels=1, 
                       dtype='float32',
                       device=self.device_index)
            sd.wait()
            time.sleep(0.3)  # Brief pause after warmup
            
            # Record a chunk to measure noise
            recording = sd.rec(int(duration * self.sample_rate), 
                               samplerate=self.sample_rate, 
                               channels=1, 
                               dtype='float32',
                               device=self.device_index)
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
            overlap_buffer = None
            while True:
                # Record one fixed-duration chunk
                new_chunk = sd.rec(int(self.chunk_duration * self.sample_rate), 
                               samplerate=self.sample_rate, 
                               channels=1, 
                               dtype='float32',
                               device=self.device_index)
                sd.wait()
                new_chunk = new_chunk.flatten()
                
                # Apply 50% overlap: keep last half of previous chunk to avoid word clipping
                if overlap_buffer is not None:
                    chunk = np.concatenate([overlap_buffer, new_chunk])
                else:
                    chunk = new_chunk
                overlap_buffer = new_chunk[len(new_chunk)//2:]
                
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
            if best_alt_prob >= 0.65 and best_alt_prob > en_prob:
                logging.info(f"Detected language: {best_alt_code} (confidence: {best_alt_prob:.3f})")
                return best_alt_code
            logging.info(f"Detected language: en (confidence: {en_prob:.3f})")
            return "en"
        except Exception as e:
            logging.warning(f"Language detection fallback to English: {e}")
        return "en"

    def _check_transcription_confidence(self, result: dict) -> bool:
        """Gate transcription quality; return False if confidence is below threshold."""
        if not result.get("segments"):
            return False
        
        avg_logprob_sum = 0
        avg_no_speech_sum = 0
        segment_count = len(result["segments"])
        
        for seg in result["segments"]:
            avg_logprob_sum += seg.get("avg_logprob", 0)
            avg_no_speech_sum += seg.get("no_speech_prob", 0)
        
        avg_logprob = avg_logprob_sum / segment_count if segment_count > 0 else -float("inf")
        avg_no_speech = avg_no_speech_sum / segment_count
        compression_ratio = result.get("result", {}).get("compression_ratio", 0)
        
        logprob_ok = avg_logprob > -1.0
        no_speech_ok = avg_no_speech < self.no_speech_threshold
        # Skip compression ratio check for very short utterances (e.g., "hello")—they often have extreme ratios
        compression_ok = compression_ratio == 0.0 or (0.5 < compression_ratio < 2.5)
        
        if not (logprob_ok and no_speech_ok and compression_ok):
            logging.warning(
                f"Low transcription confidence: logprob={avg_logprob:.3f} (ok={logprob_ok}), "
                f"no_speech={avg_no_speech:.3f} (ok={no_speech_ok}), "
                f"compression={compression_ratio:.3f} (ok={compression_ok})"
            )
            return False
        return True

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
            
            logging.info(f"Transcribing audio with Whisper (model=base, lang={lang})...")
            # Disable fp16 on CPU to prevent compatibility issues
            result_native = self.model.transcribe(
                audio_data,
                language=lang,
                task="transcribe",
                fp16=False,
                temperature=0.0,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
                no_speech_threshold=self.no_speech_threshold,
                initial_prompt=self.domain_prompt,
            )
            
            if not self._check_transcription_confidence(result_native):
                logging.info("Transcription confidence too low, requesting repeat.")
                return "", lang, ""
            
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
                    no_speech_threshold=self.no_speech_threshold,
                    initial_prompt=self.domain_prompt,
                )
                
                if self._check_transcription_confidence(result_en):
                    llm_text_en = result_en.get('text', '').strip()
                    llm_text_en = self._post_process_transcript(llm_text_en)
                else:
                    logging.warning("Translation confidence too low; using original for LLM.")
            
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
