"""
Tests for MCR Mahjong Game Engine
"""

import pytest
import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcr_mahjong.tiles import (
    Tile, TileSet, TileSuit, WindType, DragonType,
    char, bam, dot, wind, dragon, EAST, SOUTH, WEST, NORTH,
    RED_DRAGON, GREEN_DRAGON, WHITE_DRAGON
)
from mcr_mahjong.player import Player, Meld, MeldType
from mcr_mahjong.wall import Wall
from mcr_mahjong.game import Game, GamePhase, Action, ActionType
from mcr_mahjong.scoring import MCRScorer, ScoringContext


class TestTiles:
    """Test tile system"""
    
    def test_tile_creation(self):
        """Test creating tiles"""
        # Numbered suit tiles
        t1 = char(1)
        assert t1.suit == TileSuit.CHARACTERS
        assert t1.value == 1
        
        t2 = bam(5)
        assert t2.suit == TileSuit.BAMBOOS
        assert t2.value == 5
        
        t3 = dot(9)
        assert t3.suit == TileSuit.DOTS
        assert t3.value == 9
    
    def test_honor_tiles(self):
        """Test honor tile properties"""
        east = wind(WindType.EAST)
        assert east.is_honor
        assert not east.is_terminal
        
        red = dragon(DragonType.RED)
        assert red.is_honor
    
    def test_terminal_tiles(self):
        """Test terminal tile properties"""
        one_char = char(1)
        nine_char = char(9)
        five_char = char(5)
        
        assert one_char.is_terminal
        assert nine_char.is_terminal
        assert not five_char.is_terminal
        
        assert one_char.is_terminal_or_honor
        assert not five_char.is_terminal_or_honor
    
    def test_green_tiles(self):
        """Test green tile identification"""
        green_tiles = [bam(2), bam(3), bam(4), bam(6), bam(8), GREEN_DRAGON]
        non_green = [bam(1), bam(5), bam(7), RED_DRAGON, char(3)]
        
        for t in green_tiles:
            assert t.is_green, f"{t} should be green"
        
        for t in non_green:
            assert not t.is_green, f"{t} should not be green"
    
    def test_tile_index(self):
        """Test tile index calculation"""
        # Characters: 0-8
        assert char(1).tile_index == 0
        assert char(9).tile_index == 8
        
        # Bamboos: 9-17
        assert bam(1).tile_index == 9
        assert bam(9).tile_index == 17
        
        # Dots: 18-26
        assert dot(1).tile_index == 18
        assert dot(9).tile_index == 26
        
        # Winds: 27-30
        assert EAST.tile_index == 27
        assert NORTH.tile_index == 30
        
        # Dragons: 31-33
        assert RED_DRAGON.tile_index == 31
        assert WHITE_DRAGON.tile_index == 33
    
    def test_tile_from_index(self):
        """Test creating tiles from index"""
        t = Tile.from_index(0)
        assert t == char(1)
        
        t = Tile.from_index(27)
        assert t == EAST
        
        t = Tile.from_index(33)
        assert t == WHITE_DRAGON
    
    def test_tile_from_string(self):
        """Test parsing tiles from strings"""
        assert Tile.from_string("1万") == char(1)
        assert Tile.from_string("5条") == bam(5)
        assert Tile.from_string("9筒") == dot(9)
        assert Tile.from_string("东") == EAST
        assert Tile.from_string("中") == RED_DRAGON


class TestTileSet:
    """Test TileSet operations"""
    
    def test_create_full_set(self):
        """Test creating full tile set"""
        tiles = TileSet.create_full_set()
        assert len(tiles) == 136
    
    def test_tile_counts(self):
        """Test counting tiles"""
        tiles = TileSet([char(1), char(1), char(2), bam(3)])
        
        assert tiles.count(char(1)) == 2
        assert tiles.count(char(2)) == 1
        assert tiles.count(char(3)) == 0
    
    def test_to_count_array(self):
        """Test conversion to count array"""
        tiles = TileSet([char(1), char(1), char(2), EAST])
        counts = tiles.to_count_array()
        
        assert counts[0] == 2  # 1万
        assert counts[1] == 1  # 2万
        assert counts[27] == 1  # 东
        assert counts[10] == 0  # 2条
    
    def test_add_remove(self):
        """Test adding and removing tiles"""
        tiles = TileSet()
        tiles.add(char(1))
        tiles.add(char(2))
        
        assert len(tiles) == 2
        
        tiles.remove(char(1))
        assert len(tiles) == 1
        assert not tiles.contains(char(1))


class TestPlayer:
    """Test player operations"""
    
    def test_player_hand(self):
        """Test player hand management"""
        player = Player(0)
        player.add_tile(char(1))
        player.add_tile(char(2))
        player.add_tile(char(3))
        
        assert len(player.hand) == 3
        
        player.discard_tile(char(2))
        assert len(player.hand) == 2
        assert len(player.discards) == 1
    
    def test_can_pong(self):
        """Test pong detection"""
        player = Player(0)
        player.add_tile(char(5))
        player.add_tile(char(5))
        
        assert player.can_pong(char(5))
        assert not player.can_pong(char(6))
    
    def test_can_chow(self):
        """Test chow detection"""
        player = Player(0)
        player.add_tile(char(1))
        player.add_tile(char(2))
        
        possible = player.can_chow(char(3))
        assert len(possible) > 0
        
        # Can't chow honors
        player.add_tile(EAST)
        assert len(player.can_chow(SOUTH)) == 0


class TestMeld:
    """Test meld operations"""
    
    def test_chow_creation(self):
        """Test creating chow melds"""
        meld = Meld(
            meld_type=MeldType.CHOW,
            tiles=[char(1), char(2), char(3)]
        )
        assert meld.meld_type == MeldType.CHOW
    
    def test_pong_creation(self):
        """Test creating pong melds"""
        meld = Meld(
            meld_type=MeldType.PONG,
            tiles=[char(5), char(5), char(5)]
        )
        assert meld.meld_type == MeldType.PONG
    
    def test_invalid_pong(self):
        """Test invalid pong raises error"""
        with pytest.raises(ValueError):
            Meld(
                meld_type=MeldType.PONG,
                tiles=[char(5), char(5), char(6)]  # Not identical
            )


class TestWall:
    """Test wall operations"""
    
    def test_wall_creation(self):
        """Test wall creation"""
        wall = Wall()
        assert wall.remaining == 136
    
    def test_draw(self):
        """Test drawing tiles"""
        wall = Wall()
        tile = wall.draw()
        
        assert tile is not None
        assert wall.remaining == 135
    
    def test_deal_hands(self):
        """Test dealing hands"""
        wall = Wall()
        hands = wall.deal_hands(4)
        
        assert len(hands) == 4
        for hand in hands:
            assert len(hand) == 13
        
        # 4 players * 13 tiles = 52 tiles dealt
        assert wall.remaining == 136 - 52


class TestGame:
    """Test game operations"""
    
    def test_game_creation(self):
        """Test game initialization"""
        game = Game(seed=42)
        assert game.phase == GamePhase.NOT_STARTED
    
    def test_game_start(self):
        """Test starting game"""
        game = Game(seed=42)
        game.start_game()
        
        assert game.phase == GamePhase.DRAWING
        for player in game.players:
            assert len(player.hand) == 13
    
    def test_draw_action(self):
        """Test draw action"""
        game = Game(seed=42)
        game.start_game()
        
        player = game.players[0]
        initial_hand = len(player.hand)
        
        action = Action(ActionType.DRAW, 0)
        game.step(action)
        
        assert len(player.hand) == initial_hand + 1
        assert game.phase == GamePhase.DISCARDING
    
    def test_discard_action(self):
        """Test discard action"""
        game = Game(seed=42)
        game.start_game()
        
        # Draw first
        game.step(Action(ActionType.DRAW, 0))
        
        # Get a tile to discard
        player = game.players[0]
        tile_to_discard = player.hand.tiles[0]
        
        # Discard
        action = Action(ActionType.DISCARD, 0, tile_to_discard)
        game.step(action)
        
        assert game.phase == GamePhase.CLAIMING
        assert game.last_discard == tile_to_discard
    
    def test_valid_actions(self):
        """Test getting valid actions"""
        game = Game(seed=42)
        game.start_game()
        
        valid = game.get_valid_actions(0)
        assert len(valid) > 0
        assert valid[0].action_type == ActionType.DRAW
    
    def test_winning_hand_detection(self):
        """Test detection of winning hands"""
        game = Game(seed=42)
        
        # Create a simple winning hand: 111 222 333 444 55 (all characters)
        hand = TileSet()
        for val in [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]:
            hand.add(char(val))
        
        assert game._is_valid_winning_hand(hand, [])


class TestScoring:
    """Test scoring system"""
    
    def test_all_pungs(self):
        """Test All Pungs pattern"""
        scorer = MCRScorer()
        
        hand = TileSet([char(5), char(5)])  # Pair
        melds = [
            Meld(MeldType.PONG, [char(1), char(1), char(1)]),
            Meld(MeldType.PONG, [char(2), char(2), char(2)]),
            Meld(MeldType.PONG, [bam(3), bam(3), bam(3)]),
            Meld(MeldType.PONG, [dot(4), dot(4), dot(4)]),
        ]
        
        ctx = ScoringContext(
            hand=hand,
            melds=melds,
            winning_tile=char(5),
            is_zimo=True,
            round_wind=0,
            seat_wind=0,
            is_last_tile=False,
            is_kong_draw=False,
        )
        
        patterns = scorer.get_matching_patterns(ctx)
        pattern_names = [p.name for p in patterns]
        
        assert "All Pungs" in pattern_names
    
    def test_half_flush(self):
        """Test Half Flush pattern"""
        scorer = MCRScorer()
        
        hand = TileSet([char(5), char(5)])  # Pair
        melds = [
            Meld(MeldType.CHOW, [char(1), char(2), char(3)]),
            Meld(MeldType.CHOW, [char(4), char(5), char(6)]),
            Meld(MeldType.PONG, [char(9), char(9), char(9)]),
            Meld(MeldType.PONG, [EAST, EAST, EAST]),
        ]
        
        ctx = ScoringContext(
            hand=hand,
            melds=melds,
            winning_tile=char(5),
            is_zimo=True,
            round_wind=0,
            seat_wind=0,
            is_last_tile=False,
            is_kong_draw=False,
        )
        
        patterns = scorer.get_matching_patterns(ctx)
        pattern_names = [p.name for p in patterns]
        
        assert "Half Flush" in pattern_names
    
    def test_minimum_score(self):
        """Test minimum score requirement"""
        scorer = MCRScorer()
        
        # Simple hand that might not reach 8 points
        hand = TileSet([char(5), char(5)])
        melds = [
            Meld(MeldType.CHOW, [char(1), char(2), char(3)]),
            Meld(MeldType.CHOW, [bam(4), bam(5), bam(6)]),
            Meld(MeldType.CHOW, [dot(7), dot(8), dot(9)]),
            Meld(MeldType.CHOW, [char(4), char(5), char(6)]),
        ]
        
        score = scorer.calculate_score(
            hand=hand,
            melds=melds,
            winning_tile=char(5),
            is_zimo=True,
            round_wind=0,
            seat_wind=0,
        )
        
        # Score should be calculated (might be low)
        assert isinstance(score, int)


class TestEnvironment:
    """Test Gymnasium environment"""
    
    def test_env_creation(self):
        """Test environment creation"""
        from envs.mcr_env import MCRMahjongEnv
        
        env = MCRMahjongEnv()
        assert env is not None
        env.close()
    
    def test_env_reset(self):
        """Test environment reset"""
        from envs.mcr_env import MCRMahjongEnv
        
        env = MCRMahjongEnv()
        obs, info = env.reset()
        
        assert "hand" in obs
        assert "valid_actions" in obs
        assert obs["hand"].shape == (34,)
        
        env.close()
    
    def test_env_step(self):
        """Test environment step"""
        from envs.mcr_env import MCRMahjongEnv
        
        env = MCRMahjongEnv()
        obs, info = env.reset()
        
        # Get a valid action
        valid_mask = obs["valid_actions"]
        valid_indices = np.where(valid_mask == 1)[0]
        
        if len(valid_indices) > 0:
            action = valid_indices[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
        
        env.close()
    
    def test_env_observation_space(self):
        """Test observation space structure"""
        from envs.mcr_env import MCRMahjongEnv
        
        env = MCRMahjongEnv()
        
        assert "hand" in env.observation_space.spaces
        assert "melds" in env.observation_space.spaces
        assert "discards" in env.observation_space.spaces
        assert "valid_actions" in env.observation_space.spaces
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

