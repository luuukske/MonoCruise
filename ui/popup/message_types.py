"""Message type definitions and data structures."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class MessageStyle(Enum):
    """Defines the visual style of a popup message. Value is the type rank (higher = more
    important).     Order: ERROR > WARNING > CHECK > NOTICE."""
    ERROR = 4
    WARNING = 3
    CHECK = 2
    NOTICE = 1


# Char-based message type mapping
MESSAGE_TYPE_MAP: dict[str, MessageStyle] = {
    "e": MessageStyle.ERROR,
    "w": MessageStyle.WARNING,
    "c": MessageStyle.CHECK,
    "n": MessageStyle.NOTICE,
}


@dataclass
class StyleConfig:
    """Visual configuration for a message style."""
    icon_color: str
    border_color: str
    text_color: str
    icon_name: str
    
    
STYLE_CONFIGS: dict[MessageStyle, StyleConfig] = {
    MessageStyle.ERROR: StyleConfig(
        icon_color="#FF4444",
        border_color="#FF4444",
        text_color="#FF4444",
        icon_name="error.svg"
    ),
    MessageStyle.WARNING: StyleConfig(
        icon_color="#FFA500",
        border_color="#FFA500",
        text_color="#FFA500",
        icon_name="warning.svg"
    ),
    MessageStyle.CHECK: StyleConfig(
        icon_color="#44CC44",
        border_color="#44CC44",
        text_color="#44CC44",
        icon_name="check.svg"
    ),
    MessageStyle.NOTICE: StyleConfig(
        icon_color="#FFFFFF",
        border_color="#FFFFFF",
        text_color="#FFFFFF",
        icon_name="info.svg"
    ),
}


@dataclass
class PopupMessage:
    """Represents a popup message. Ordering: 1. Message type rank (error > warning > check >..."""
    priority: int
    title: str
    text: str
    style: MessageStyle
    duration_ms: int
    timestamp: float = field(default_factory=time.time)
    remaining_ms: int = field(default=0)
    
    @property
    def type_rank(self) -> int:
        """The rank derived from the message type (higher = more important)."""
        return self.style.value
    
    def __post_init__(self):
        if self.remaining_ms == 0:
            self.remaining_ms = self.duration_ms
    
    def __lt__(self, other: "PopupMessage") -> bool:
        """Type rank first, then priority within same type, then timestamp."""
        if self.type_rank != other.type_rank:
            return self.type_rank > other.type_rank
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp
    
    def get_style_config(self) -> StyleConfig:
        """Get the visual configuration for this message's style."""
        return STYLE_CONFIGS[self.style]
    
    def __repr__(self) -> str:
        return f"Message(type={self.style.name}, p={self.priority}, text='{self.text[:20]}...')"