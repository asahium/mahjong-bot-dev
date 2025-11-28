"""
Tenhou Client Interface

Provides an interface for connecting trained RL agents to Tenhou.
This is an abstraction layer that can be implemented using various methods:
1. WebSocket connection to Tenhou servers
2. Browser automation
3. Third-party libraries (e.g., tenhou-python-bot)

Note: Using bots on Tenhou may violate their terms of service.
This interface is provided for educational and research purposes.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Optional, Dict, Any, Callable
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcr_mahjong.tiles import Tile, TileSet, TileSuit
from riichi_mahjong.game import RiichiActionType


logger = logging.getLogger(__name__)


class TenhouLobby(IntEnum):
    """Tenhou lobby types"""
    IPPAN = 0       # General (0-1800 rating)
    JOUKYUU = 1     # Upper (1800+ rating)
    TOKUJOU = 2     # Special Upper (2000+ rating)
    HOUOU = 3       # Phoenix (2200+ rating)


class TenhouGameType(IntEnum):
    """Tenhou game types"""
    TONPUU = 0      # East only
    HANCHAN = 1     # East-South
    SANMA = 2       # 3-player


@dataclass
class TenhouGameState:
    """
    Represents the current game state received from Tenhou.
    
    This mirrors the observation space used by the RL agent,
    allowing easy conversion between formats.
    """
    # Player info
    player_id: int = 0
    seat: int = 0  # 0=East, 1=South, 2=West, 3=North
    
    # Hand and tiles
    hand: List[int] = field(default_factory=list)  # Tile IDs
    melds: List[List[int]] = field(default_factory=list)
    discards: List[List[int]] = field(default_factory=list)  # Per player
    
    # Dora
    dora_indicators: List[int] = field(default_factory=list)
    
    # Game state
    round_wind: int = 0  # 0=East, 1=South
    round_number: int = 0
    honba: int = 0
    riichi_sticks: int = 0
    scores: List[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    
    # Turn info
    current_player: int = 0
    last_draw: Optional[int] = None
    last_discard: Optional[int] = None
    last_discard_player: Optional[int] = None
    
    # Status
    riichi_status: List[bool] = field(default_factory=lambda: [False] * 4)
    furiten: bool = False
    
    # Available actions
    can_discard: bool = False
    can_riichi: bool = False
    can_tsumo: bool = False
    can_ron: bool = False
    can_pon: bool = False
    can_chii: bool = False
    can_kan: bool = False
    
    def to_observation(self) -> Dict[str, np.ndarray]:
        """
        Convert Tenhou state to RL agent observation format.
        
        Returns:
            Dictionary observation compatible with RiichiMahjongEnv
        """
        # Hand (34 tile types)
        hand_counts = np.zeros(34, dtype=np.int8)
        for tile_id in self.hand:
            tile_idx = tile_id // 4  # Convert tile ID to type index
            hand_counts[tile_idx] += 1
        
        # Melds (4 players x 4 melds x 34 tiles)
        melds_array = np.zeros((4, 4, 34), dtype=np.int8)
        for p_idx, player_melds in enumerate(self.melds[:4]):
            for m_idx, meld in enumerate(player_melds[:4] if player_melds else []):
                for tile_id in meld:
                    tile_idx = tile_id // 4
                    melds_array[p_idx, m_idx, tile_idx] += 1
        
        # Discards (4 x 34)
        discards_array = np.zeros((4, 34), dtype=np.int8)
        for p_idx, player_discards in enumerate(self.discards[:4]):
            for tile_id in player_discards:
                tile_idx = tile_id // 4
                discards_array[p_idx, tile_idx] += 1
        
        # Dora indicators (5 x 34, one-hot)
        dora_array = np.zeros((5, 34), dtype=np.int8)
        for i, tile_id in enumerate(self.dora_indicators[:5]):
            tile_idx = tile_id // 4
            dora_array[i, tile_idx] = 1
        
        # Riichi status
        riichi_array = np.array(
            [1 if r else 0 for r in self.riichi_status],
            dtype=np.int8
        )
        
        # Furiten
        furiten_array = np.array([1 if self.furiten else 0], dtype=np.int8)
        
        # Scores
        scores_array = np.array(self.scores, dtype=np.float32)
        
        # Game info
        game_info = np.array([
            self.current_player,
            1 if self.can_discard else 3,  # Simplified phase
            self.round_wind,
            self.seat,
            self.round_number % 4,  # Dealer seat
            self.honba,
            self.riichi_sticks,
            70,  # Approximate wall remaining
            self.round_number * 20,  # Approximate turn
            1 if self.current_player == self.seat else 0,
            1 if self.can_tsumo else 0,
            1 if self.can_ron else 0,
        ], dtype=np.float32)
        
        # Valid actions mask (simplified)
        valid_actions = np.zeros(250, dtype=np.int8)
        
        if self.can_discard:
            for tile_id in self.hand:
                tile_idx = tile_id // 4
                valid_actions[tile_idx] = 1
            if self.can_riichi:
                for tile_id in self.hand:
                    tile_idx = tile_id // 4
                    valid_actions[34 + tile_idx] = 1
        
        if self.can_tsumo:
            valid_actions[238] = 1
        if self.can_ron:
            valid_actions[239] = 1
        if self.can_pon:
            valid_actions[240] = 1  # Pass
            if self.last_discard is not None:
                tile_idx = self.last_discard // 4
                valid_actions[102 + tile_idx] = 1
        if self.can_chii:
            valid_actions[240] = 1  # Pass
            # Chii options would need more complex handling
        if self.can_kan:
            valid_actions[240] = 1  # Pass
        
        if not any([self.can_discard, self.can_tsumo, self.can_ron, 
                    self.can_pon, self.can_chii, self.can_kan]):
            valid_actions[241] = 1  # Draw
        
        return {
            "hand": hand_counts,
            "melds": melds_array,
            "discards": discards_array,
            "dora_indicators": dora_array,
            "riichi_status": riichi_array,
            "furiten": furiten_array,
            "scores": scores_array,
            "valid_actions": valid_actions,
            "game_info": game_info,
        }


class TenhouClientInterface(ABC):
    """
    Abstract interface for connecting to Tenhou.
    
    This class defines the methods that must be implemented
    to connect a trained agent to Tenhou servers.
    
    Usage:
        1. Implement a concrete class (e.g., using websockets or selenium)
        2. Create agent and load trained model
        3. Connect to Tenhou
        4. Run game loop
        
    Example:
        client = MyTenhouClient()
        agent = load_trained_agent("models/riichi_ppo/best")
        
        await client.connect(username="user", lobby=TenhouLobby.IPPAN)
        await client.join_game(TenhouGameType.HANCHAN)
        
        while not client.game_over:
            state = await client.get_state()
            obs = state.to_observation()
            action, _ = agent.predict(obs)
            await client.send_action(action)
    """
    
    def __init__(self):
        self.connected = False
        self.in_game = False
        self.game_over = False
        self.player_id = None
        self.current_state: Optional[TenhouGameState] = None
    
    @abstractmethod
    async def connect(
        self, 
        username: str, 
        password: Optional[str] = None,
        lobby: TenhouLobby = TenhouLobby.IPPAN
    ) -> bool:
        """
        Connect to Tenhou servers.
        
        Args:
            username: Tenhou username
            password: Password (if required)
            lobby: Lobby to join
            
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from Tenhou."""
        pass
    
    @abstractmethod
    async def join_game(
        self, 
        game_type: TenhouGameType = TenhouGameType.HANCHAN
    ) -> bool:
        """
        Join a game queue.
        
        Args:
            game_type: Type of game to join
            
        Returns:
            True if successfully joined queue
        """
        pass
    
    @abstractmethod
    async def get_state(self) -> TenhouGameState:
        """
        Get current game state.
        
        Returns:
            Current game state
        """
        pass
    
    @abstractmethod
    async def send_action(self, action: int) -> bool:
        """
        Send an action to Tenhou.
        
        Args:
            action: Action index from agent
            
        Returns:
            True if action was accepted
        """
        pass
    
    @abstractmethod
    async def wait_for_turn(self, timeout: float = 60.0) -> bool:
        """
        Wait for it to be player's turn.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if it's player's turn, False if timeout
        """
        pass
    
    def action_to_tenhou(self, action: int, state: TenhouGameState) -> Dict[str, Any]:
        """
        Convert agent action to Tenhou protocol format.
        
        Args:
            action: Action index from agent
            state: Current game state
            
        Returns:
            Dictionary with Tenhou action parameters
        """
        # Discard
        if 0 <= action < 34:
            tile_type = action
            # Find a tile of this type in hand
            for tile_id in state.hand:
                if tile_id // 4 == tile_type:
                    return {"type": "discard", "tile": tile_id}
        
        # Riichi + Discard
        elif 34 <= action < 68:
            tile_type = action - 34
            for tile_id in state.hand:
                if tile_id // 4 == tile_type:
                    return {"type": "riichi", "tile": tile_id}
        
        # Chii
        elif 68 <= action < 102:
            return {"type": "chii", "tiles": []}  # Would need proper tile selection
        
        # Pon
        elif 102 <= action < 136:
            tile_type = action - 102
            tiles = [t for t in state.hand if t // 4 == tile_type][:2]
            return {"type": "pon", "tiles": tiles}
        
        # Kan
        elif 136 <= action < 170:
            tile_type = action - 136
            tiles = [t for t in state.hand if t // 4 == tile_type][:3]
            return {"type": "kan", "tiles": tiles}
        
        # Ankan
        elif 170 <= action < 204:
            tile_type = action - 170
            tiles = [t for t in state.hand if t // 4 == tile_type][:4]
            return {"type": "ankan", "tiles": tiles}
        
        # Shouminkan
        elif 204 <= action < 238:
            tile_type = action - 204
            for tile_id in state.hand:
                if tile_id // 4 == tile_type:
                    return {"type": "kakan", "tile": tile_id}
        
        # Tsumo
        elif action == 238:
            return {"type": "tsumo"}
        
        # Ron
        elif action == 239:
            return {"type": "ron"}
        
        # Pass
        elif action == 240:
            return {"type": "pass"}
        
        # Draw
        elif action == 241:
            return {"type": "draw"}
        
        return {"type": "pass"}
    
    def set_callback(
        self, 
        event: str, 
        callback: Callable[[Any], None]
    ) -> None:
        """
        Set a callback for game events.
        
        Args:
            event: Event name (e.g., "game_start", "round_end", "game_end")
            callback: Callback function
        """
        if not hasattr(self, '_callbacks'):
            self._callbacks = {}
        self._callbacks[event] = callback


class MockTenhouClient(TenhouClientInterface):
    """
    Mock Tenhou client for testing.
    
    Simulates Tenhou connection using local game engine.
    Useful for testing agent integration without actual Tenhou connection.
    """
    
    def __init__(self):
        super().__init__()
        self._game = None
        self._agent_seat = 0
    
    async def connect(
        self, 
        username: str, 
        password: Optional[str] = None,
        lobby: TenhouLobby = TenhouLobby.IPPAN
    ) -> bool:
        logger.info(f"Mock: Connected as {username} to {lobby.name}")
        self.connected = True
        return True
    
    async def disconnect(self) -> None:
        logger.info("Mock: Disconnected")
        self.connected = False
    
    async def join_game(
        self, 
        game_type: TenhouGameType = TenhouGameType.HANCHAN
    ) -> bool:
        from riichi_mahjong.game import RiichiGame
        from riichi_mahjong.rules import TENHOU_RULES
        
        logger.info(f"Mock: Joined {game_type.name} game")
        self._game = RiichiGame(rules=TENHOU_RULES)
        self._game.reset()
        self._game.start_round()
        self.in_game = True
        self.game_over = False
        return True
    
    async def get_state(self) -> TenhouGameState:
        if self._game is None:
            return TenhouGameState()
        
        player = self._game.players[self._agent_seat]
        
        state = TenhouGameState(
            player_id=0,
            seat=self._agent_seat,
            hand=[t.tile_index * 4 for t in player.hand.tiles],
            round_wind=self._game.round_wind,
            honba=self._game.honba,
            riichi_sticks=self._game.riichi_sticks,
            scores=[p.score for p in self._game.players],
            current_player=self._game.current_player,
            riichi_status=[p.is_riichi for p in self._game.players],
            furiten=player.furiten.is_furiten,
        )
        
        # Set available actions
        valid_actions = self._game.get_valid_actions(self._agent_seat)
        for action in valid_actions:
            if action.action_type == RiichiActionType.DISCARD:
                state.can_discard = True
            elif action.action_type == RiichiActionType.RIICHI:
                state.can_riichi = True
            elif action.action_type == RiichiActionType.TSUMO:
                state.can_tsumo = True
            elif action.action_type == RiichiActionType.RON:
                state.can_ron = True
            elif action.action_type == RiichiActionType.PON:
                state.can_pon = True
            elif action.action_type == RiichiActionType.CHII:
                state.can_chii = True
            elif action.action_type == RiichiActionType.KAN:
                state.can_kan = True
        
        if self._game.last_discard:
            state.last_discard = self._game.last_discard.tile_index * 4
            state.last_discard_player = self._game.last_discard_player
        
        if self._game.last_draw:
            state.last_draw = self._game.last_draw.tile_index * 4
        
        # Dora indicators
        state.dora_indicators = [d.tile_index * 4 for d in self._game.dora_indicators]
        
        self.current_state = state
        return state
    
    async def send_action(self, action: int) -> bool:
        if self._game is None:
            return False
        
        # Convert action index to game action
        from riichi_mahjong.game import RiichiAction, RiichiActionType
        from mcr_mahjong.tiles import Tile
        
        player_idx = self._agent_seat
        
        if 0 <= action < 34:
            tile = Tile.from_index(action)
            game_action = RiichiAction(RiichiActionType.DISCARD, player_idx, tile)
        elif 34 <= action < 68:
            tile = Tile.from_index(action - 34)
            game_action = RiichiAction(RiichiActionType.RIICHI, player_idx, tile)
        elif action == 238:
            game_action = RiichiAction(RiichiActionType.TSUMO, player_idx, self._game.last_draw)
        elif action == 239:
            game_action = RiichiAction(RiichiActionType.RON, player_idx, self._game.last_discard)
        elif action == 240:
            game_action = RiichiAction(RiichiActionType.PASS, player_idx)
        elif action == 241:
            game_action = RiichiAction(RiichiActionType.DRAW, player_idx)
        else:
            game_action = RiichiAction(RiichiActionType.PASS, player_idx)
        
        try:
            over, result = self._game.step(game_action)
            if over:
                self.game_over = True
                logger.info(f"Mock: Game over - {result}")
            return True
        except Exception as e:
            logger.error(f"Mock: Action failed - {e}")
            return False
    
    async def wait_for_turn(self, timeout: float = 60.0) -> bool:
        if self._game is None:
            return False
        
        # In mock, run opponent turns
        import random
        
        max_iterations = 100
        for _ in range(max_iterations):
            if self._game.current_player == self._agent_seat:
                return True
            
            # Run opponent action
            opp_idx = self._game.current_player
            valid_actions = self._game.get_valid_actions(opp_idx)
            
            if valid_actions:
                action = random.choice(valid_actions)
                over, result = self._game.step(action)
                if over:
                    self.game_over = True
                    return False
        
        return False


async def run_agent_on_tenhou(
    agent,
    client: TenhouClientInterface,
    username: str,
    num_games: int = 1,
    game_type: TenhouGameType = TenhouGameType.HANCHAN,
) -> List[Dict[str, Any]]:
    """
    Run a trained agent on Tenhou.
    
    Args:
        agent: Trained agent with predict() method
        client: Tenhou client implementation
        username: Tenhou username
        num_games: Number of games to play
        game_type: Type of game
        
    Returns:
        List of game results
    """
    results = []
    
    await client.connect(username=username)
    
    for game_num in range(num_games):
        logger.info(f"Starting game {game_num + 1}/{num_games}")
        
        await client.join_game(game_type)
        
        while not client.game_over:
            # Wait for turn
            if not await client.wait_for_turn():
                break
            
            # Get state and convert to observation
            state = await client.get_state()
            obs = state.to_observation()
            
            # Get agent action
            action, _ = agent.predict(obs)
            
            # Send action
            await client.send_action(action)
        
        results.append({
            "game": game_num + 1,
            "state": client.current_state,
        })
    
    await client.disconnect()
    
    return results

