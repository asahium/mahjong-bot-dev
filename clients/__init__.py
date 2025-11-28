"""
Mahjong Client Interfaces

Provides interfaces for connecting trained agents to online platforms.
"""

from .tenhou_client import (
    TenhouClientInterface, 
    TenhouGameState,
    TenhouLobby,
    TenhouGameType,
    MockTenhouClient,
)

# WebSocket client (requires websockets library)
try:
    from .tenhou_websocket import TenhouWebSocketClient
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    TenhouWebSocketClient = None

__all__ = [
    "TenhouClientInterface", 
    "TenhouGameState",
    "TenhouLobby",
    "TenhouGameType",
    "MockTenhouClient",
    "TenhouWebSocketClient",
    "WEBSOCKET_AVAILABLE",
]

