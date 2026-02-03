"""Chat domain models."""
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """Chat message."""
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ChatHistory:
    """Chat history with limit."""
    messages: list[ChatMessage] = field(default_factory=list)
    max_messages: int = 10

    def add(self, message: ChatMessage) -> None:
        """Add message to history."""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def add_pair(self, user_content: str, assistant_content: str) -> None:
        """Add user/assistant message pair."""
        self.add(ChatMessage(role="user", content=user_content))
        self.add(ChatMessage(role="assistant", content=assistant_content))

    def to_list(self) -> list[dict]:
        """Convert to list of dicts for LLM."""
        return [{"role": m.role, "content": m.content} for m in self.messages]
