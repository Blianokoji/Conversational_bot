class Memory:
    def __init__(self, max_turns: int = 5):
        """
        Maintains a sliding window of the last `max_turns` interactions.
        """
        self.history = []
        self.max_turns = max_turns

    def add_interaction(self, user_text: str, bot_response: str):
        self.history.append({"user": user_text, "bot": bot_response})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context_string(self) -> str:
        """
        Formats the current memory history into a string for LLM injection.
        """
        if not self.history:
            return ""
        
        lines = ["\n[Recent Conversation History]"]
        for turn in self.history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['bot']}")
            
        return "\n".join(lines) + "\n"
