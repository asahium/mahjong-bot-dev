#!/usr/bin/env python3
"""
Tenhou WebSocket Client

Real implementation of Tenhou client using WebSocket protocol.
Based on research of Tenhou's communication protocol.

Protocol overview:
- Connection: wss://b-ww.mjv.jp/ (or regional servers)
- Messages: XML-based format
- Tile encoding: 0-135 (4 copies of 34 tile types)

Message types:
- HELO: Authentication response
- JOIN: Join lobby
- GO: Game request/response
- INIT: Round initialization (deal)
- T/U/V/W: Draw tile (for player position 0/1/2/3)
- D/E/F/G: Discard tile (for player position 0/1/2/3)
- N: Call (chi, pon, kan, kita)
- REACH: Riichi declaration
- AGARI: Win declaration
- RYUUKYOKU: Draw game
- DORA: Dora indicator revealed

⚠️ Warning: Using bots on Tenhou may violate their Terms of Service.
This code is for educational and research purposes only.
"""

import asyncio
import hashlib
import logging
import random
import re
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Callable, Any
from urllib.parse import quote

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("Install websockets: pip install websockets")

from clients.tenhou_client import (
    TenhouClientInterface, 
    TenhouGameState,
    TenhouLobby,
    TenhouGameType,
)

logger = logging.getLogger(__name__)


# Tenhou server endpoints
TENHOU_SERVERS = {
    "main": "wss://b-ww.mjv.jp/",
    "test": "wss://b.mjv.jp/",  # Test server
}


class TenhouMeldType(IntEnum):
    """Meld types in Tenhou encoding"""
    CHI = 0
    PON = 1
    KAKAN = 2  # Add to pon
    MINKAN = 3  # Open kan
    ANKAN = 4  # Closed kan


def decode_meld(meld_int: int) -> Tuple[TenhouMeldType, List[int], int]:
    """
    Decode Tenhou meld integer.
    
    Returns:
        (meld_type, tile_ids, from_player_offset)
    """
    from_who = meld_int & 0x3
    
    if meld_int & 0x4:
        # Chi
        t0, t1, t2 = (meld_int >> 3) & 0x3, (meld_int >> 5) & 0x3, (meld_int >> 7) & 0x3
        base_and_called = meld_int >> 10
        base = base_and_called // 3
        base = (base // 7) * 9 + base % 7
        base *= 4
        tiles = [t0 + 4 * (base // 4), t1 + 4 * (base // 4 + 1), t2 + 4 * (base // 4 + 2)]
        return TenhouMeldType.CHI, tiles, from_who
    
    elif meld_int & 0x8:
        # Pon
        t4 = (meld_int >> 5) & 0x3
        t0, t1, t2 = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))[t4]
        base = (meld_int >> 9) // 3
        base *= 4
        tiles = [t0 + base, t1 + base, t2 + base]
        return TenhouMeldType.PON, tiles, from_who
    
    elif meld_int & 0x10:
        # Kakan (add to pon)
        added = (meld_int >> 5) & 0x3
        base = (meld_int >> 9) // 3
        base *= 4
        tiles = [i + base for i in range(4)]
        return TenhouMeldType.KAKAN, tiles, from_who
    
    else:
        # Kan
        hai0 = (meld_int >> 8) // 4
        hai0 *= 4
        tiles = [i + hai0 for i in range(4)]
        
        if from_who == 0:
            return TenhouMeldType.ANKAN, tiles, 0
        else:
            return TenhouMeldType.MINKAN, tiles, from_who
    
    return TenhouMeldType.CHI, [], 0


def encode_discard(tile_id: int, tsumogiri: bool = False) -> str:
    """Encode a discard action for Tenhou."""
    # D for player 0, add 60 for tsumogiri
    tile = tile_id + (60 if tsumogiri else 0)
    return f"<D p=\"{tile}\"/>"


def encode_call_response(call_type: str, meld_tiles: List[int] = None) -> str:
    """Encode a call response (chi, pon, kan, pass)."""
    if call_type == "pass":
        return "<N />"
    elif call_type == "chi":
        # Chi encoding
        return f"<N type=\"1\" hai=\"{','.join(map(str, meld_tiles))}\"/>"
    elif call_type == "pon":
        return f"<N type=\"2\" hai=\"{','.join(map(str, meld_tiles))}\"/>"
    elif call_type == "kan":
        return f"<N type=\"3\" hai=\"{','.join(map(str, meld_tiles))}\"/>"
    elif call_type == "ankan":
        return f"<N type=\"4\" hai=\"{','.join(map(str, meld_tiles))}\"/>"
    return "<N />"


def generate_auth_token(auth_string: str) -> str:
    """
    Generate authentication token for Tenhou.
    Based on Tenhou's authentication algorithm.
    """
    # Simplified auth - real Tenhou uses more complex algorithm
    table = [
        22136, 52719, 55146, 42104, 59591, 46934, 9248, 28891,
        49597, 52974, 62844, 4015, 18311, 50730, 43056, 17939,
        64838, 38145, 27008, 39128, 35652, 63407, 65535, 23473,
        35164, 55230, 27536, 4386, 64920, 29075, 42617, 17294,
        18868, 2081
    ]
    
    if not auth_string:
        return ""
    
    parts = auth_string.split('-')
    if len(parts) != 2:
        return ""
    
    try:
        a = int(parts[0])
        b = int(parts[1], 16)
        
        result = a ^ b ^ table[a % len(table)]
        return f"{result:08x}"
    except:
        return ""


@dataclass
class TenhouMessage:
    """Parsed Tenhou message"""
    tag: str
    attributes: Dict[str, str] = field(default_factory=dict)
    raw: str = ""
    
    @classmethod
    def parse(cls, xml_str: str) -> 'TenhouMessage':
        """Parse XML message from Tenhou."""
        xml_str = xml_str.strip()
        
        # Extract tag name
        match = re.match(r'<(\w+)', xml_str)
        if not match:
            return cls(tag="UNKNOWN", raw=xml_str)
        
        tag = match.group(1)
        
        # Extract attributes
        attrs = {}
        for m in re.finditer(r'(\w+)="([^"]*)"', xml_str):
            attrs[m.group(1)] = m.group(2)
        
        return cls(tag=tag, attributes=attrs, raw=xml_str)


class TenhouWebSocketClient(TenhouClientInterface):
    """
    WebSocket client for Tenhou.
    
    Usage:
        client = TenhouWebSocketClient()
        await client.connect(username="NoName")
        await client.join_game(TenhouGameType.HANCHAN)
        
        while not client.game_over:
            if await client.wait_for_turn():
                state = await client.get_state()
                obs = state.to_observation()
                action = agent.predict(obs)
                await client.send_action(action)
    """
    
    def __init__(self, server: str = "main"):
        super().__init__()
        
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required: pip install websockets")
        
        self.server_url = TENHOU_SERVERS.get(server, TENHOU_SERVERS["main"])
        self.ws = None
        self.username = None
        self.seat = 0  # 0=East, 1=South, 2=West, 3=North
        
        # Game state
        self.hand: List[int] = []
        self.melds: List[List[int]] = [[] for _ in range(4)]
        self.discards: List[List[int]] = [[] for _ in range(4)]
        self.dora_indicators: List[int] = []
        self.scores: List[int] = [25000] * 4
        
        self.round_wind = 0
        self.round_number = 0
        self.honba = 0
        self.riichi_sticks = 0
        
        self.current_player = 0
        self.last_draw: Optional[int] = None
        self.last_discard: Optional[int] = None
        self.last_discard_player: Optional[int] = None
        
        self.riichi_status = [False] * 4
        self.furiten = False
        
        # Pending actions
        self.can_actions: Dict[str, bool] = {}
        self.pending_action = None
        
        # Message queue
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._receive_task = None
        
        # Callbacks
        self._callbacks: Dict[str, Callable] = {}
    
    async def connect(
        self, 
        username: str = "NoName",
        password: Optional[str] = None,
        lobby: TenhouLobby = TenhouLobby.IPPAN
    ) -> bool:
        """Connect to Tenhou server."""
        try:
            logger.info(f"Connecting to {self.server_url}...")
            
            self.ws = await websockets.connect(
                self.server_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5,
            )
            
            self.username = username
            self.connected = True
            
            # Start message receiver
            self._receive_task = asyncio.create_task(self._receive_messages())
            
            # Send HELO
            helo_msg = f'<HELO name="{quote(username)}" tid="f0" sx="M" />'
            await self._send(helo_msg)
            
            # Wait for auth challenge
            msg = await self._wait_for_message("HELO", timeout=10)
            if msg:
                auth_string = msg.attributes.get("auth", "")
                if auth_string:
                    token = generate_auth_token(auth_string)
                    auth_response = f'<AUTH val="{token}"/>'
                    await self._send(auth_response)
            
            logger.info(f"Connected as {username}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Tenhou."""
        if self._receive_task:
            self._receive_task.cancel()
        
        if self.ws:
            await self.ws.close()
        
        self.connected = False
        logger.info("Disconnected from Tenhou")
    
    async def join_game(
        self, 
        game_type: TenhouGameType = TenhouGameType.HANCHAN
    ) -> bool:
        """Join a game queue."""
        if not self.connected:
            return False
        
        # Game type encoding
        # Bit flags: hanchan, kuitan, aka, etc.
        type_flags = 0b0000001  # Default: Ippan hanchan
        
        if game_type == TenhouGameType.TONPUU:
            type_flags = 0b0000001
        elif game_type == TenhouGameType.HANCHAN:
            type_flags = 0b0001001
        
        join_msg = f'<JOIN t="{type_flags}" />'
        await self._send(join_msg)
        
        logger.info(f"Joining {game_type.name} game...")
        
        # Wait for game start (GO message)
        msg = await self._wait_for_message("GO", timeout=300)
        if msg:
            self.in_game = True
            logger.info("Game started!")
            return True
        
        return False
    
    async def get_state(self) -> TenhouGameState:
        """Get current game state."""
        state = TenhouGameState(
            player_id=0,
            seat=self.seat,
            hand=self.hand.copy(),
            melds=[m.copy() for m in self.melds],
            discards=[d.copy() for d in self.discards],
            dora_indicators=self.dora_indicators.copy(),
            round_wind=self.round_wind,
            round_number=self.round_number,
            honba=self.honba,
            riichi_sticks=self.riichi_sticks,
            scores=self.scores.copy(),
            current_player=self.current_player,
            last_draw=self.last_draw,
            last_discard=self.last_discard,
            last_discard_player=self.last_discard_player,
            riichi_status=self.riichi_status.copy(),
            furiten=self.furiten,
        )
        
        # Set available actions
        state.can_discard = self.can_actions.get("discard", False)
        state.can_riichi = self.can_actions.get("riichi", False)
        state.can_tsumo = self.can_actions.get("tsumo", False)
        state.can_ron = self.can_actions.get("ron", False)
        state.can_pon = self.can_actions.get("pon", False)
        state.can_chii = self.can_actions.get("chi", False)
        state.can_kan = self.can_actions.get("kan", False)
        
        self.current_state = state
        return state
    
    async def send_action(self, action: int) -> bool:
        """Send an action to Tenhou."""
        if not self.connected or not self.in_game:
            return False
        
        state = await self.get_state()
        tenhou_action = self.action_to_tenhou(action, state)
        
        action_type = tenhou_action.get("type", "pass")
        
        if action_type == "discard":
            tile = tenhou_action.get("tile", 0)
            tsumogiri = (tile == self.last_draw)
            msg = encode_discard(tile, tsumogiri)
        
        elif action_type == "riichi":
            tile = tenhou_action.get("tile", 0)
            msg = f'<REACH hai="{tile}" />'
        
        elif action_type == "tsumo":
            msg = '<N type="7" />'  # Tsumo
        
        elif action_type == "ron":
            msg = '<N type="6" />'  # Ron
        
        elif action_type == "chi":
            tiles = tenhou_action.get("tiles", [])
            msg = encode_call_response("chi", tiles)
        
        elif action_type == "pon":
            tiles = tenhou_action.get("tiles", [])
            msg = encode_call_response("pon", tiles)
        
        elif action_type == "kan":
            tiles = tenhou_action.get("tiles", [])
            msg = encode_call_response("kan", tiles)
        
        elif action_type == "ankan":
            tiles = tenhou_action.get("tiles", [])
            msg = encode_call_response("ankan", tiles)
        
        else:  # pass
            msg = '<N />'
        
        await self._send(msg)
        
        # Clear pending actions
        self.can_actions.clear()
        
        return True
    
    async def wait_for_turn(self, timeout: float = 60.0) -> bool:
        """Wait for it to be player's turn."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.game_over:
                return False
            
            # Process incoming messages
            msg = await self._get_next_message(timeout=1.0)
            if msg:
                await self._handle_message(msg)
            
            # Check if we have any available actions
            if any(self.can_actions.values()):
                return True
        
        return False
    
    async def _send(self, message: str) -> None:
        """Send message to server."""
        if self.ws:
            logger.debug(f"SEND: {message}")
            await self.ws.send(message)
    
    async def _receive_messages(self) -> None:
        """Background task to receive messages."""
        try:
            async for message in self.ws:
                logger.debug(f"RECV: {message}")
                parsed = TenhouMessage.parse(message)
                await self._message_queue.put(parsed)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Receive error: {e}")
    
    async def _get_next_message(self, timeout: float = 5.0) -> Optional[TenhouMessage]:
        """Get next message from queue."""
        try:
            msg = await asyncio.wait_for(
                self._message_queue.get(),
                timeout=timeout
            )
            return msg
        except asyncio.TimeoutError:
            return None
    
    async def _wait_for_message(
        self, 
        tag: str, 
        timeout: float = 30.0
    ) -> Optional[TenhouMessage]:
        """Wait for specific message type."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = await self._get_next_message(timeout=1.0)
            if msg:
                if msg.tag == tag:
                    return msg
                else:
                    await self._handle_message(msg)
        
        return None
    
    async def _handle_message(self, msg: TenhouMessage) -> None:
        """Handle incoming message."""
        handler = getattr(self, f'_handle_{msg.tag.lower()}', None)
        if handler:
            await handler(msg)
        else:
            logger.debug(f"Unhandled message: {msg.tag}")
    
    async def _handle_init(self, msg: TenhouMessage) -> None:
        """Handle INIT (round start) message."""
        # Parse seed (round info)
        seed_str = msg.attributes.get("seed", "0,0,0,0,0,0")
        seed_parts = list(map(int, seed_str.split(',')))
        
        self.round_number = seed_parts[0] if len(seed_parts) > 0 else 0
        self.round_wind = self.round_number // 4
        self.honba = seed_parts[1] if len(seed_parts) > 1 else 0
        self.riichi_sticks = seed_parts[2] if len(seed_parts) > 2 else 0
        
        # Dora indicator
        if len(seed_parts) > 5:
            self.dora_indicators = [seed_parts[5]]
        
        # Parse hand
        hand_keys = ['hai0', 'hai1', 'hai2', 'hai3']
        for i, key in enumerate(hand_keys):
            if key in msg.attributes:
                tiles = list(map(int, msg.attributes[key].split(',')))
                if i == self.seat:
                    self.hand = tiles
        
        # Parse scores
        scores_str = msg.attributes.get("ten", "250,250,250,250")
        self.scores = [int(x) * 100 for x in scores_str.split(',')]
        
        # Reset state
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.riichi_status = [False] * 4
        self.last_draw = None
        self.last_discard = None
        self.last_discard_player = None
        self.current_player = self.round_number % 4
        
        logger.info(f"Round {self.round_number + 1} started. Hand: {len(self.hand)} tiles")
        
        if self._callbacks.get("round_start"):
            self._callbacks["round_start"](msg)
    
    async def _handle_t(self, msg: TenhouMessage) -> None:
        """Handle T (player 0 draw) message."""
        await self._handle_draw(0, msg)
    
    async def _handle_u(self, msg: TenhouMessage) -> None:
        """Handle U (player 1 draw) message."""
        await self._handle_draw(1, msg)
    
    async def _handle_v(self, msg: TenhouMessage) -> None:
        """Handle V (player 2 draw) message."""
        await self._handle_draw(2, msg)
    
    async def _handle_w(self, msg: TenhouMessage) -> None:
        """Handle W (player 3 draw) message."""
        await self._handle_draw(3, msg)
    
    async def _handle_draw(self, player: int, msg: TenhouMessage) -> None:
        """Handle draw for any player."""
        self.current_player = player
        
        # Extract tile from tag (e.g., <T24/> means player 0 drew tile 24)
        tile_match = re.search(r'[TUVW](\d+)', msg.raw)
        if tile_match:
            tile = int(tile_match.group(1))
            
            if player == self.seat:
                self.hand.append(tile)
                self.last_draw = tile
                self.can_actions["discard"] = True
                
                # Check for tsumo possibility (would be indicated in attributes)
                if "t" in msg.attributes:
                    self.can_actions["tsumo"] = True
                
                logger.debug(f"Drew tile: {tile}")
    
    async def _handle_d(self, msg: TenhouMessage) -> None:
        """Handle D (player 0 discard) message."""
        await self._handle_discard(0, msg)
    
    async def _handle_e(self, msg: TenhouMessage) -> None:
        """Handle E (player 1 discard) message."""
        await self._handle_discard(1, msg)
    
    async def _handle_f(self, msg: TenhouMessage) -> None:
        """Handle F (player 2 discard) message."""
        await self._handle_discard(2, msg)
    
    async def _handle_g(self, msg: TenhouMessage) -> None:
        """Handle G (player 3 discard) message."""
        await self._handle_discard(3, msg)
    
    async def _handle_discard(self, player: int, msg: TenhouMessage) -> None:
        """Handle discard for any player."""
        # Extract tile from tag
        tile_match = re.search(r'[DEFG](\d+)', msg.raw)
        if tile_match:
            tile = int(tile_match.group(1))
            actual_tile = tile % 60  # Remove tsumogiri flag
            
            self.discards[player].append(actual_tile)
            self.last_discard = actual_tile
            self.last_discard_player = player
            
            if player == self.seat and actual_tile in self.hand:
                self.hand.remove(actual_tile)
            
            # Check for call opportunities (indicated in attributes)
            if player != self.seat:
                if "t" in msg.attributes:  # Can call
                    # Parse available calls
                    t_val = int(msg.attributes.get("t", "0"))
                    if t_val & 1:
                        self.can_actions["chi"] = True
                    if t_val & 2:
                        self.can_actions["pon"] = True
                    if t_val & 4:
                        self.can_actions["kan"] = True
                    if t_val & 8:
                        self.can_actions["ron"] = True
            
            logger.debug(f"Player {player} discarded: {actual_tile}")
    
    async def _handle_n(self, msg: TenhouMessage) -> None:
        """Handle N (call) message."""
        who = int(msg.attributes.get("who", "0"))
        meld_int = int(msg.attributes.get("m", "0"))
        
        meld_type, tiles, from_offset = decode_meld(meld_int)
        
        # Add to melds
        self.melds[who].append(tiles)
        
        # Remove tiles from hand if it's our call
        if who == self.seat:
            for tile in tiles:
                if tile in self.hand:
                    self.hand.remove(tile)
        
        logger.debug(f"Player {who} called {meld_type.name}: {tiles}")
    
    async def _handle_reach(self, msg: TenhouMessage) -> None:
        """Handle REACH (riichi) message."""
        who = int(msg.attributes.get("who", "0"))
        self.riichi_status[who] = True
        logger.info(f"Player {who} declared riichi")
    
    async def _handle_dora(self, msg: TenhouMessage) -> None:
        """Handle DORA (new dora indicator) message."""
        hai = int(msg.attributes.get("hai", "0"))
        self.dora_indicators.append(hai)
        logger.debug(f"New dora indicator: {hai}")
    
    async def _handle_agari(self, msg: TenhouMessage) -> None:
        """Handle AGARI (win) message."""
        who = int(msg.attributes.get("who", "0"))
        from_who = int(msg.attributes.get("fromWho", who))
        
        is_tsumo = (who == from_who)
        win_type = "Tsumo" if is_tsumo else "Ron"
        
        logger.info(f"Player {who} wins by {win_type}!")
        
        # Parse score changes
        sc_str = msg.attributes.get("sc", "")
        if sc_str:
            sc_parts = list(map(int, sc_str.split(',')))
            # Score format: player0_score, player0_change, player1_score, ...
            for i in range(min(4, len(sc_parts) // 2)):
                self.scores[i] = sc_parts[i * 2] * 100
        
        if self._callbacks.get("round_end"):
            self._callbacks["round_end"](msg)
    
    async def _handle_ryuukyoku(self, msg: TenhouMessage) -> None:
        """Handle RYUUKYOKU (draw game) message."""
        logger.info("Round ended in draw")
        
        if self._callbacks.get("round_end"):
            self._callbacks["round_end"](msg)
    
    async def _handle_owari(self, msg: TenhouMessage) -> None:
        """Handle OWARI (game end) message."""
        self.game_over = True
        self.in_game = False
        
        # Parse final scores
        owari_str = msg.attributes.get("owari", "")
        if owari_str:
            parts = owari_str.split(',')
            final_scores = []
            for i in range(0, min(8, len(parts)), 2):
                try:
                    final_scores.append(int(parts[i]) * 100)
                except:
                    pass
            if len(final_scores) == 4:
                self.scores = final_scores
        
        logger.info(f"Game ended! Final scores: {self.scores}")
        
        if self._callbacks.get("game_end"):
            self._callbacks["game_end"](msg)
    
    async def _handle_prof(self, msg: TenhouMessage) -> None:
        """Handle PROF (profile/ranking update) message."""
        logger.debug("Profile update received")
    
    async def _handle_rejoin(self, msg: TenhouMessage) -> None:
        """Handle REJOIN message."""
        logger.info("Rejoining game...")
    
    async def _handle_un(self, msg: TenhouMessage) -> None:
        """Handle UN (user names) message."""
        # Player names
        for i in range(4):
            name_key = f"n{i}"
            if name_key in msg.attributes:
                logger.debug(f"Player {i}: {msg.attributes[name_key]}")


async def run_bot_on_tenhou(
    model_path: str,
    username: str = "NoName",
    num_games: int = 1,
    game_type: TenhouGameType = TenhouGameType.HANCHAN,
):
    """
    Run trained bot on real Tenhou.
    
    Args:
        model_path: Path to trained model
        username: Tenhou username
        num_games: Number of games to play
        game_type: Type of game
    """
    from stable_baselines3 import PPO
    
    logger.info(f"Loading model: {model_path}")
    agent = PPO.load(model_path)
    
    client = TenhouWebSocketClient()
    
    results = []
    
    try:
        if not await client.connect(username=username):
            logger.error("Failed to connect")
            return results
        
        for game_num in range(num_games):
            logger.info(f"\n=== Game {game_num + 1}/{num_games} ===\n")
            
            if not await client.join_game(game_type):
                logger.error("Failed to join game")
                continue
            
            while not client.game_over:
                if await client.wait_for_turn(timeout=120):
                    state = await client.get_state()
                    obs = state.to_observation()
                    
                    # Get agent action
                    action, _ = agent.predict(obs, deterministic=True)
                    action = int(action)
                    
                    # Validate action
                    valid_mask = obs["valid_actions"]
                    if valid_mask[action] != 1:
                        # Find any valid action
                        valid_indices = [i for i, v in enumerate(valid_mask) if v == 1]
                        if valid_indices:
                            action = valid_indices[0]
                        else:
                            action = 240  # Pass
                    
                    await client.send_action(action)
            
            results.append({
                "game": game_num + 1,
                "final_scores": client.scores.copy(),
                "seat": client.seat,
            })
            
            # Small delay between games
            await asyncio.sleep(5)
        
    finally:
        await client.disconnect()
    
    return results


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--username", type=str, default="NoName")
    parser.add_argument("--games", type=int, default=1)
    
    args = parser.parse_args()
    
    asyncio.run(run_bot_on_tenhou(
        model_path=args.model,
        username=args.username,
        num_games=args.games,
    ))

