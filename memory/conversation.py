from utils.logger import get_logger

logger = get_logger(__name__)

MAX_HISTORY = 10  # Keep last 10 exchanges to avoid token limits

class ConversationMemory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> list[dict]:
        return self.messages.copy()

    def _trim(self):
        """Keep only the last MAX_HISTORY messages to avoid token overflow."""
        if len(self.messages) > MAX_HISTORY * 2:
            self.messages = self.messages[-(MAX_HISTORY * 2):]
            logger.info(f"Trimmed conversation history to {len(self.messages)} messages")

    def clear(self):
        self.messages = []
        logger.info("Conversation memory cleared")

    def __len__(self):
        return len(self.messages)
