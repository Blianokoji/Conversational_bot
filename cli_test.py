import sys
import logging
from rag.retriever import Retriever
from conversation.memory import Memory
from rag.generator import Generator

# Suppress noisy HTTP logs from Google SDK
logging.getLogger('httpx').setLevel(logging.WARNING)

def run_cli():
    print("\n" + "="*50)
    print("🚀 Starting Transight AI (Text-Only CLI Mode)...")
    print("="*50)

    try:
        print("Initializing Knowledge Base Retriever (FAISS)...")
        retriever = Retriever()
        
        print("Initializing Conversational Memory...")
        memory = Memory(max_turns=5)
        
        print("Initializing Gemini 1.5 Flash Generator...")
        generator = Generator(retriever, memory)
        
        print("\n✅ System Ready. Type your question.")
        print("Type 'quit' or 'exit' to stop.\n")
        
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["goodbye", "quit", "exit", "stop", "bye"]:
                print("👋 Shutting down...")
                break
            
            # Generate response via RAG + Gemini Flash
            response_text = generator.generate_response(user_input)
            print(f"\n🤖 Bot: {response_text}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Shutting down via keyboard interruption...")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_cli()
