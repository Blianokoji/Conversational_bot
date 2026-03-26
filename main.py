import sys
import time
import logging

from audio.stt import BufferedSTT
from audio.tts import TTSHandler
from rag.retriever import Retriever
from conversation.memory import Memory
from rag.generator import Generator

# Suppress noisy HTTP logs from Google SDK
logging.getLogger('httpx').setLevel(logging.WARNING)

def run_agent():
    print("\n" + "="*50)
    print("🚀 Starting Transight AI Conversational Agent...")
    print("="*50)

    try:
        # Initialize components
        print("[1/5] Initializing STT (Whisper buffered chunks)...")
        # Use 2-second chunks for more stable speech capture.
        stt = BufferedSTT(model_size="base", energy_threshold=0.005, chunk_duration=2.0)
        
        print("[2/5] Initializing TTS (gTTS + Pygame)...")
        tts = TTSHandler()
        
        print("[3/5] Initializing Knowledge Base Retriever (FAISS)...")
        retriever = Retriever()
        
        print("[4/5] Initializing Conversational Memory...")
        memory = Memory(max_turns=5)
        
        print("[5/5] Initializing Gemini 2.5 Flash Generator...")
        generator = Generator(retriever, memory)
        
        print("\n✅ System Ready. You can start speaking.")
        print("To quit, say 'Goodbye', 'Quit', or press Ctrl+C.")
        
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        sys.exit(1)

    while True:
        try:
            # 1. Listen natively until silence (fixed-duration chunk accumulation)
            user_text, lang_code, user_text_en = stt.transcribe()
            
            if not user_text:
                continue
                
            # Exit condition
            low_text = user_text.lower().strip(" .!,")
            if low_text in ["goodbye", "quit", "exit", "stop", "bye"]:
                print("👋 Shutting down...")
                tts.speak("Goodbye! Have a great day.", lang="en")
                break
            
            # 2. Use English text for retrieval/LLM because KB vectors are English.
            llm_query = user_text_en if user_text_en else user_text
            response_text = generator.generate_response(
                llm_query,
                response_language=lang_code,
                original_user_query=user_text,
            )
            print(f"\n🤖 Bot ({lang_code}): {response_text}\n")
            
            # 3. Speak the response natively matching the user's language
            tts.speak(response_text, lang=lang_code)
            
            # Add delay to allow audio device to fully reset between output and input
            time.sleep(1.0)
            
        except KeyboardInterrupt:
            print("\n👋 Shutting down via keyboard interruption...")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_agent()
