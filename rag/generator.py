import os
import re
import logging
from google import genai
from google.genai import types

from rag.retriever import Retriever
from conversation.memory import Memory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class Generator:
    def __init__(self, retriever: Retriever, memory: Memory):
        self.retriever = retriever
        self.memory = memory
        
        # Pull API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.warning("GEMINI_API_KEY environment variable not set. Google GenAI calls will fail.")
        
        # Initialize Google GenAI client
        self.client = genai.Client() if not api_key else genai.Client(api_key=api_key)
        
        # Safer model choice for MVP
        self.model_name = 'gemini-2.5-flash'
        
        # Strong system instruction for grounding + voice optimization
        self.system_instruction = (
            "You are Transight AI Assistant, a voice-based support agent.\n"
            "Follow these strict rules:\n"
            "- Answer ONLY using the provided context\n"
            "- Do NOT hallucinate or add outside knowledge\n"
            "- If unsure, ask the user to repeat\n"
            "- Keep responses short, clear, and natural for speech\n"
            "- Respond in the same language as the user\n"
        )
        self.language_names = {"en": "English", "hi": "Hindi", "ml": "Malayalam"}
        self.reprompt_text = "Could you say that again?"

    def _normalize_query(self, user_query: str) -> str:
        """Fix common STT substitutions for key domain terms before retrieval."""
        normalized = user_query.strip()
        substitutions = {
            r"\btransit\b": "Transight",
            r"\btransite\b": "Transight",
            r"\btran\s*site\b": "Transight",
            r"\bclimatics\b": "telematics",
            r"\bio t\b": "IoT",
        }
        for pattern, replacement in substitutions.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        return normalized

    def _is_identity_query(self, user_query: str) -> bool:
        q = user_query.lower()
        return any(
            phrase in q
            for phrase in [
                "who are you",
                "what are you",
                "what do you do",
                "about transight",
                "tell me about transight",
                "what is transight",
            ]
        )

    def _is_greeting_query(self, user_query: str) -> bool:
        q = user_query.lower().strip()
        return bool(re.search(r"\b(hello|hi|hey|good\s*morning|good\s*evening|namaste|namaskaram)\b", q))

    def _local_context_fallback(self, user_query: str) -> str:
        """Generate a concise fallback answer directly from retrieved chunks."""
        try:
            results = self.retriever.retrieve(user_query, top_k=3)
            if not results:
                return self.reprompt_text

            best_text = (results[0].get("text") or "").strip()
            if not best_text:
                return self.reprompt_text

            compact = " ".join(best_text.split())
            summary = compact[:320].rstrip(" ,.;:")
            return f"Based on our information, {summary}."
        except Exception:
            return self.reprompt_text

    def generate_response(
        self,
        user_query: str,
        response_language: str = "en",
        original_user_query: str = "",
    ) -> str:
        """
        Retrieves context, formats prompt, injects memory, and calls Gemini Flash.
        """
        user_query = self._normalize_query(user_query)
        response_language = response_language if response_language in self.language_names else "en"
        response_language_name = self.language_names[response_language]

        if self._is_greeting_query(user_query):
            bot_text = "Hello. I am the Transight AI Assistant. How can I help you today?"
            if response_language != "en":
                bot_text = self._translate_with_model(bot_text, response_language_name)
            self.memory.add_interaction(user_query, bot_text)
            return bot_text

        # Handle common identity/company intro queries deterministically.
        if self._is_identity_query(user_query):
            bot_text = (
                "I am the Transight AI Assistant. Transight provides IoT, telematics, "
                "compliance, and remote monitoring solutions across industries like "
                "automotive, logistics, utilities, and manufacturing."
            )
            if response_language != "en":
                # Let model localize deterministic text when non-English output is requested.
                bot_text = self._translate_with_model(bot_text, response_language_name)
            self.memory.add_interaction(user_query, bot_text)
            return bot_text
        
        # 1. Retrieve relevant KB chunks
        kb_context = self.retriever.get_context_string(user_query, top_k=5)
        
        # 2. Get past conversation
        chat_context = self.memory.get_context_string()
        
        # 3. Assemble strong structured prompt
        prompt = f"""
You are Transight AI Assistant.

--- RULES ---
1. Answer ONLY using the provided Context Information.
2. Do NOT use outside knowledge.
3. If the answer is not in the context, say:
    "Could you say that again?"
4. Keep responses concise and spoken-friendly (maximum 3–4 sentences).
5. Do NOT use bullet points, markdown, or formatting.
6. Respond ONLY in {response_language_name}.
7. Do not mix languages in one response.

--- CONTEXT INFORMATION ---
{kb_context}

--- CONVERSATION HISTORY ---
{chat_context}

--- USER QUERY ---
{user_query}

--- ORIGINAL USER SPOKEN QUERY ---
{original_user_query}

--- RESPONSE ---
"""

        try:
            logging.info("Generating response with Gemini Flash...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.3,
                )
            )
            
            bot_text = response.text.strip() if response.text else "I'm sorry, I couldn't generate a response."
            
            # 4. Save to memory
            self.memory.add_interaction(user_query, bot_text)
            
            return bot_text
        
        except Exception as e:
            logging.error(f"Error calling Gemini: {e}")
            error_text = str(e)
            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                return self._local_context_fallback(user_query)
            return self.reprompt_text

    def _translate_with_model(self, english_text: str, target_language_name: str) -> str:
        """Translate fixed deterministic responses when non-English output is requested."""
        try:
            prompt = (
                f"Translate the following text to {target_language_name}. "
                "Keep meaning exact and concise for spoken output.\n\n"
                f"Text: {english_text}"
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return response.text.strip() if response.text else english_text
        except Exception:
            return english_text


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    ret = Retriever()
    mem = Memory()
    gen = Generator(ret, mem)
    
    query = "What exactly does Transight do?"
    print(f"User: {query}")
    
    ans = gen.generate_response(query)
    print(f"Bot: {ans}")