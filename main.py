#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PP1000 — AI poker training software. AI that makes you pro.
Single-file app: hand evaluation, range training, session tracking, and optional
PokerPro contract integration for on-chain session anchoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# -----------------------------------------------------------------------------
# Constants and config
# -----------------------------------------------------------------------------

class PP1000Constants:
    APP_NAME = "PP1000"
    VERSION = "1.0.0"
    CONFIG_DIR = ".pp1000"
    CONFIG_FILE = "config.json"
    SESSIONS_FILE = "sessions.json"
    HISTORY_FILE = "hand_history.json"
    DEFAULT_RPC = "https://rpc.ankr.com/eth"
    DEFAULT_CHAIN_ID = 1
    POKERPRO_DEPLOYED_AT = "0x3C7f2a9E4b1D6c8F0a2B4e6A8c0D2f4B6e8"
    TRAINER_ROLE_ADDR = "0x5D8bE1f3A4c6E9a0B2d4F7a9C1e3B5d7F9a"
    AI_ORACLE_ADDR = "0x6E9cF2b5A7d0E1f3B6a8C0e2A4c6E8f0B2d"
    VAULT_ADDR = "0x7F0dE3a6B8c1D4f2A5e7B9c0D2f4A6e8B0d"
    MAX_HANDS_PER_SESSION = 1024
    MAX_SESSIONS_STORED = 72000
    STAKES_TIERS = 11
    TRAINING_LEVELS = 20
    QUALITY_BANDS = 11
    RANK_NAMES = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
    SUIT_NAMES = ("c", "d", "h", "s")
    HAND_RANK_NAMES = (
        "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
        "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"
    )


# -----------------------------------------------------------------------------
# Card and deck
# -----------------------------------------------------------------------------

class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        r = PP1000Constants.RANK_NAMES[self.rank]
        s = PP1000Constants.SUIT_NAMES[self.suit]
        return r + s

    @classmethod
    def from_string(cls, s: str) -> Card:
        s = s.strip().upper()
        if len(s) < 2:
            raise ValueError(f"Invalid card: {s}")
        rchar = s[0]
        schar = s[1]
        rank_map = {c: Rank(i) for i, c in enumerate(PP1000Constants.RANK_NAMES)}
        suit_map = {c: Suit(i) for i, c in enumerate(PP1000Constants.SUIT_NAMES)}
        if rchar == "1" and len(s) >= 3 and s[1:3] == "0":
            rchar, schar = "T", s[2]
        rank = rank_map.get(rchar)
        suit = suit_map.get(schar)
        if rank is None or suit is None:
            raise ValueError(f"Invalid card: {s}")
        return cls(rank=rank, suit=suit)

    def to_index(self) -> int:
        return int(self.rank) * 4 + int(self.suit)


def make_deck() -> List[Card]:
    return [Card(Rank(r), Suit(s)) for r in range(13) for s in range(4)]


def shuffle_deck(deck: List[Card], rng: Optional[random.Random] = None) -> List[Card]:
    out = list(deck)
    (rng or random).shuffle(out)
    return out


# -----------------------------------------------------------------------------
# Hand evaluation (simplified but deterministic)
# -----------------------------------------------------------------------------

def _rank_counts(cards: Sequence[Card]) -> List[Tuple[int, int]]:
    counts: Dict[int, int] = {}
    for c in cards:
        r = int(c.rank)
        counts[r] = counts.get(r, 0) + 1
    pairs = [(count, rank) for rank, count in counts.items()]
    pairs.sort(key=lambda x: (-x[0], -x[1]))
    return pairs


def _is_straight(ranks: List[int]) -> Optional[int]:
    s = sorted(set(ranks), reverse=True)
    if len(s) < 5:
        return None
    for i in range(len(s) - 4):
        run = s[i : i + 5]
        if run[0] - run[-1] == 4:
            return run[0]
    if 12 in s and 3 in s and 2 in s and 1 in s and 0 in s:
        return 3
    return None


def _is_flush(cards: Sequence[Card]) -> Optional[List[int]]:
    by_suit: Dict[int, List[int]] = {}
    for c in cards:
        by_suit.setdefault(int(c.suit), []).append(int(c.rank))
    for suit, ranks in by_suit.items():
        if len(ranks) >= 5:
            return sorted(ranks, reverse=True)[:5]
    return None


def evaluate_hand(cards: Sequence[Card]) -> Tuple[int, List[int]]:
    if len(cards) < 5:
        raise ValueError("Need at least 5 cards")
    ranks = [int(c.rank) for c in cards]
    suits = [int(c.suit) for c in cards]
    rank_cnt = _rank_counts(cards)
    flush_ranks = _is_flush(cards)
    straight_high = _is_straight(ranks)
    is_flush = flush_ranks is not None
    is_straight = straight_high is not None
    if is_flush and is_straight:
        sr = sorted(set(ranks), reverse=True)
        for i in range(len(sr) - 4):
            run = sr[i : i + 5]
            if run[0] - run[-1] == 4 or (run == [12, 11, 10, 9, 8] or (12 in run and 3 in run and 2 in run and 1 in run and 0 in run)):
                high = run[0] if run[0] - run[-1] == 4 else 3
                return (9, [high])
        if 12 in ranks and 11 in ranks and 10 in ranks and 9 in ranks and 8 in ranks:
            return (10, [12])
    if rank_cnt[0][0] == 4:
        kickers = sorted([r for r in ranks if r != rank_cnt[0][1]], reverse=True)[:1]
        return (8, [rank_cnt[0][1]] + kickers)
    if rank_cnt[0][0] >= 3 and len(rank_cnt) >= 2 and rank_cnt[1][0] >= 2:
        return (7, [rank_cnt[0][1], rank_cnt[1][1]])
    if is_flush:
        return (6, flush_ranks[:5])
    if is_straight:
        return (5, [straight_high])
    if rank_cnt[0][0] == 3:
        kickers = sorted([r for r in ranks if r != rank_cnt[0][1]], reverse=True)[:2]
        return (4, [rank_cnt[0][1]] + kickers)
    if rank_cnt[0][0] == 2 and len(rank_cnt) >= 2 and rank_cnt[1][0] == 2:
        k = [rank_cnt[0][1], rank_cnt[1][1]]
        kickers = sorted([r for r in ranks if r not in k], reverse=True)[:1]
        return (3, sorted(k, reverse=True) + kickers)
    if rank_cnt[0][0] == 2:
        kickers = sorted([r for r in ranks if r != rank_cnt[0][1]], reverse=True)[:3]
        return (2, [rank_cnt[0][1]] + kickers)
    return (1, sorted(ranks, reverse=True)[:5])


def compare_hands(h1: Sequence[Card], h2: Sequence[Card]) -> int:
    e1 = evaluate_hand(h1)
    e2 = evaluate_hand(h2)
    if e1[0] != e2[0]:
        return 1 if e1[0] > e2[0] else -1
    for a, b in zip(e1[1], e2[1]):
        if a != b:
            return 1 if a > b else -1
    return 0


def hand_rank_name(hand_type: int) -> str:
    if 1 <= hand_type <= len(PP1000Constants.HAND_RANK_NAMES):
        return PP1000Constants.HAND_RANK_NAMES[hand_type - 1]
    return "Unknown"


# -----------------------------------------------------------------------------
# AI training engine: ranges and suggestions
# -----------------------------------------------------------------------------

@dataclass
class AISuggestion:
    action: str
    confidence: float
    ev_estimate: Optional[float]
    reasoning: str
    alternatives: List[str]


class AITrainingEngine:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._range_tight = 0.22
        self._range_loose = 0.38

    def suggest_preflop(self, hole: Sequence[Card], position: str, stakes_tier: int) -> AISuggestion:
        r1, r2 = int(hole[0].rank), int(hole[1].rank)
        suited = hole[0].suit == hole[1].suit
        high, low = max(r1, r2), min(r1, r2)
        pp = r1 == r2
        if pp and high >= 9:
            return AISuggestion("raise", 0.92, 12.5, "Premium pair; raise for value.", ["all-in", "call"])
        if high >= 11 and low >= 9 and (suited or high == low):
            return AISuggestion("raise", 0.85, 8.2, "Strong hand; build pot.", ["call", "fold"])
        if high >= 10 and suited:
            return AISuggestion("call", 0.72, 3.1, "Playable suited; see flop.", ["raise", "fold"])
        if position in ("btn", "co", "hj") and high >= 8:
            return AISuggestion("call", 0.58, 1.2, "Positional play.", ["fold", "raise"])
        if stakes_tier >= 7 and high < 8 and not suited:
            return AISuggestion("fold", 0.88, -0.5, "Weak hand at high stakes.", ["call"])
        return AISuggestion("fold", 0.65, -0.3, "Marginal; fold in early position.", ["call"])

    def suggest_postflop(self, hole: Sequence[Card], board: Sequence[Card], pot_frac: float) -> AISuggestion:
        all_cards = list(hole) + list(board)
        if len(all_cards) < 5:
            return AISuggestion("check", 0.5, 0.0, "Insufficient board.", [])
        ht, kick = evaluate_hand(all_cards)
        if ht >= 7:
            return AISuggestion("bet", 0.9, 15.0, "Strong made hand; value bet.", ["check", "raise"])
        if ht >= 5 and pot_frac < 0.4:
            return AISuggestion("bet", 0.75, 6.0, "Good hand; build pot.", ["check"])
        if ht >= 3:
            return AISuggestion("check", 0.6, 1.0, "Medium strength; pot control.", ["bet", "fold"])
        return AISuggestion("check", 0.55, -0.2, "Weak; check or fold.", ["bet", "fold"])

    def quality_band(self, suggestion: AISuggestion, actual_action: str) -> int:
        match = 1 if suggestion.action.lower() == actual_action.lower() else 0
        c = suggestion.confidence
        band = min(10, int(c * 10)) if match else max(0, int((1 - c) * 5))
        return band


# -----------------------------------------------------------------------------
# Session and hand history (local storage)
# -----------------------------------------------------------------------------

@dataclass
class HandRecord:
    hand_id: str
    hole: List[str]
    board: List[str]
    action_taken: str
    ai_suggestion: str
    quality_band: int
    stakes_tier: int
    timestamp: float


@dataclass
class TrainingSession:
    session_id: str
    stakes_tier: int
    opened_at: float
    closed_at: Optional[float]
    hands: List[HandRecord]
    level_unlocked: int


def _ensure_config_dir() -> Path:
    path = Path.home() / PP1000Constants.CONFIG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sessions_path() -> Path:
    return _ensure_config_dir() / PP1000Constants.SESSIONS_FILE


def _history_path() -> Path:
    return _ensure_config_dir() / PP1000Constants.HISTORY_FILE


def load_sessions() -> List[Dict[str, Any]]:
    p = _sessions_path()
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_sessions(sessions: List[Dict[str, Any]]) -> None:
    _sessions_path().parent.mkdir(parents=True, exist_ok=True)
    with open(_sessions_path(), "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)


def load_history() -> List[Dict[str, Any]]:
    p = _history_path()
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: List[Dict[str, Any]]) -> None:
    _history_path().parent.mkdir(parents=True, exist_ok=True)
    with open(_history_path(), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# -----------------------------------------------------------------------------
# PokerPro contract hashes (for optional on-chain anchoring)
# -----------------------------------------------------------------------------

def _keccak256(data: bytes) -> bytes:
    try:
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()
    except Exception:
        try:
            import sha3
            return sha3.keccak_256(data).digest()
        except Exception:
            return hashlib.sha3_256(data).digest()


def hand_hash_for_contract(hole: List[str], board: List[str], session_salt: str) -> str:
    payload = f"{session_salt}:{','.join(hole)}:{','.join(board)}"
    h = _keccak256(payload.encode("utf-8"))
    return "0x" + h.hex()


def feedback_hash_for_contract(hand_id: str, quality_band: int, suggestion: str) -> str:
    payload = f"{hand_id}:{quality_band}:{suggestion}"
    h = _keccak256(payload.encode("utf-8"))
    return "0x" + h.hex()


# -----------------------------------------------------------------------------
# CLI menus and flows
# -----------------------------------------------------------------------------

def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  PP1000 — AI Poker Training. AI that makes you pro.")
    print("  Version:", PP1000Constants.VERSION)
    print("=" * 60 + "\n")


def menu_main() -> str:
    print("1. New training session")
    print("2. Hand evaluator")
    print("3. AI suggestion (preflop)")
    print("4. AI suggestion (postflop)")
    print("5. View session history")
    print("6. Stats and progress")
    print("7. Drills")
    print("8. Settings")
    print("9. Export sessions (CSV)")
    print("a. Help")
    print("b. Preflop quiz bank")
    print("0. Exit")
    return input("Choice: ").strip()


def run_hand_evaluator() -> None:
    print("Enter 5–7 cards (e.g. As Kh Qd Jc Th). One line:")
    line = input().strip()
    cards = []
    for part in line.split():
        try:
            cards.append(Card.from_string(part))
        except ValueError as e:
            print("Error:", e)
            return
    if len(cards) < 5:
        print("Need at least 5 cards.")
        return
    ht, kickers = evaluate_hand(cards)
    print("Hand:", hand_rank_name(ht), "| Kickers:", kickers)


def run_ai_preflop(engine: AITrainingEngine) -> None:
    print("Enter 2 hole cards (e.g. As Kh):")
    line = input().strip().split()
    if len(line) < 2:
        print("Need 2 cards.")
        return
    try:
        hole = [Card.from_string(line[0]), Card.from_string(line[1])]
    except ValueError as e:
        print("Error:", e)
        return
    pos = input("Position (utg, hj, co, btn, sb, bb): ").strip() or "btn"
    tier = 5
    try:
        t = input("Stakes tier 0–10 [5]: ").strip()
        if t:
            tier = max(0, min(10, int(t)))
    except ValueError:
        pass
    s = engine.suggest_preflop(hole, pos, tier)
    print("AI:", s.action, "| Confidence:", s.confidence, "|", s.reasoning)


def run_ai_postflop(engine: AITrainingEngine) -> None:
    print("Enter hole cards (e.g. As Kh):")
    hole_parts = input().strip().split()[:2]
    if len(hole_parts) < 2:
        print("Need 2 hole cards.")
        return
    print("Enter board (e.g. Qd Jc Th 2s):")
    board_parts = input().strip().split()[:5]
    try:
        hole = [Card.from_string(hole_parts[0]), Card.from_string(hole_parts[1])]
        board = [Card.from_string(p) for p in board_parts]
    except ValueError as e:
        print("Error:", e)
        return
    pot_frac = 0.3
    try:
        p = input("Pot fraction 0–1 [0.3]: ").strip()
        if p:
            pot_frac = max(0.0, min(1.0, float(p)))
    except ValueError:
        pass
    s = engine.suggest_postflop(hole, board, pot_frac)
    print("AI:", s.action, "| Confidence:", s.confidence, "|", s.reasoning)


def run_new_session(engine: AITrainingEngine) -> None:
    session_id = secrets.token_hex(16)
    tier = 5
    try:
        t = input("Stakes tier 0–10 [5]: ").strip()
        if t:
            tier = max(0, min(10, int(t)))
    except ValueError:
        pass
    sessions = load_sessions()
    sessions.append({
        "session_id": session_id,
        "stakes_tier": tier,
        "opened_at": time.time(),
        "closed_at": None,
        "hands": [],
        "level_unlocked": 0,
    })
    save_sessions(sessions)
    print("Session started:", session_id[:16] + "...")
    hands_done = 0
    while True:
        print("\n1. Add hand  2. Close session")
        ch = input("Choice: ").strip()
        if ch == "2":
            break
        if ch != "1":
            continue
        print("Hole cards (e.g. As Kh):")
        hole_str = input().strip().split()[:2]
        if len(hole_str) < 2:
            continue
        print("Board (optional, space-separated):")
        board_str = input().strip().split()[:5]
        print("Your action (fold/call/check/bet/raise):")
        action = input().strip() or "check"
        try:
            hole = [Card.from_string(hole_str[0]), Card.from_string(hole_str[1])]
            board = [Card.from_string(b) for b in board_str] if board_str else []
            if board:
                sug = engine.suggest_postflop(hole, board, 0.3)
            else:
                sug = engine.suggest_preflop(hole, "btn", tier)
            band = engine.quality_band(sug, action)
            hand_id = secrets.token_hex(8)
            rec = {
                "hand_id": hand_id,
                "hole": hole_str,
                "board": board_str,
                "action_taken": action,
                "ai_suggestion": sug.action,
                "quality_band": band,
                "stakes_tier": tier,
                "timestamp": time.time(),
            }
            for s in sessions:
                if s.get("session_id") == session_id:
                    s.setdefault("hands", []).append(rec)
                    break
            save_sessions(sessions)
            hands_done += 1
            print("Hand recorded. AI suggested:", sug.action, "| Quality band:", band)
        except ValueError as e:
            print("Error:", e)
    for s in sessions:
        if s.get("session_id") == session_id:
            s["closed_at"] = time.time()
            s["level_unlocked"] = min(PP1000Constants.TRAINING_LEVELS - 1, hands_done // 10)
            break
    save_sessions(sessions)
    print("Session closed. Hands:", hands_done)


def run_view_history() -> None:
    sessions = load_sessions()
    if not sessions:
        print("No sessions yet.")
        return
    for i, s in enumerate(sessions[-20:]):
        sid = s.get("session_id", "?")[:12]
        tier = s.get("stakes_tier", 0)
        hands = s.get("hands", [])
        print(f"  {i+1}. {sid}... tier={tier} hands={len(hands)}")
    idx = input("Session number to list hands (Enter to skip): ").strip()
    if not idx:
        return
    try:
        i = int(idx)
        s = sessions[-min(i, len(sessions))]
        for h in s.get("hands", [])[-15:]:
            print("  ", h.get("hole"), "->", h.get("action_taken"), "| AI:", h.get("ai_suggestion"))
    except (ValueError, IndexError):
        pass


def run_stats() -> None:
    sessions = load_sessions()
    total_hands = sum(len(s.get("hands", [])) for s in sessions)
    total_sessions = len(sessions)
    if total_hands == 0:
        print("No hands recorded yet.")
        return
    match = 0
    for s in sessions:
        for h in s.get("hands", []):
            if h.get("action_taken", "").lower() == h.get("ai_suggestion", "").lower():
                match += 1
    level = compute_level(total_hands, match)
    print("Total sessions:", total_sessions)
    print("Total hands:", total_hands)
    print("AI agreement rate: {:.1f}%".format(100.0 * match / total_hands if total_hands else 0))
    print("Training level:", level, "—", level_title(level))
    run_progress_detail(sessions)
    run_stats_by_tier(sessions)
    run_stats_by_action(sessions)


def run_progress_detail(sessions: List[Dict[str, Any]]) -> None:
    total = sum(len(s.get("hands", [])) for s in sessions)
    match = sum(
        1 for s in sessions for h in s.get("hands", [])
        if h.get("action_taken", "").lower() == h.get("ai_suggestion", "").lower()
    )
    level = compute_level(total, match)
    next_level_hands = (level + 1) * 50
    print("  Progress to next level: {} / {} hands".format(total, next_level_hands))
    if level < PP1000Constants.TRAINING_LEVELS - 1:
        print("  Next: ", level_title(level + 1))


def run_stats_by_tier(sessions: List[Dict[str, Any]]) -> None:
    by_tier: Dict[int, List[int]] = {}
    for s in sessions:
        t = s.get("stakes_tier", 0)
        by_tier.setdefault(t, []).append(len(s.get("hands", [])))
    print("  Hands by stakes tier:")
    for t in sorted(by_tier.keys()):
        total = sum(by_tier[t])
        print("    Tier {}: {} hands ({} sessions)".format(t, total, len(by_tier[t])))


def run_stats_by_action(sessions: List[Dict[str, Any]]) -> None:
    actions: Dict[str, int] = {}
    for s in sessions:
        for h in s.get("hands", []):
            a = h.get("action_taken", "unknown").lower()
            actions[a] = actions.get(a, 0) + 1
    print("  Actions taken:")
    for a, c in sorted(actions.items(), key=lambda x: -x[1]):
        print("    {}: {}".format(a, c))


def run_drills(engine: AITrainingEngine) -> None:
    print("1. Preflop 5-hand  2. Position drill  3. Postflop drill  4. Range quiz  5. Equity quiz")
    ch = input("Choice: ").strip()
    if ch == "1":
        _drill_preflop_five(engine)
    elif ch == "2":
        _drill_position(engine)
    elif ch == "3":
        _drill_postflop(engine)
    elif ch == "4":
        _drill_range_quiz(engine)
    elif ch == "5":
        _drill_equity_quiz(engine)
    else:
        print("Unknown; running preflop 5-hand.")
        _drill_preflop_five(engine)


def _drill_preflop_five(engine: AITrainingEngine) -> None:
    print("Drill: Preflop decision. 5 random hands.")
    deck = shuffle_deck(make_deck())
    for i in range(5):
        hole = [deck.pop(), deck.pop()]
        pos = random.choice(["utg", "hj", "co", "btn", "sb", "bb"])
        sug = engine.suggest_preflop(hole, pos, 5)
        print("  Hole:", hole[0], hole[1], "| Position:", pos, "| AI:", sug.action, sug.reasoning[:40])


def _drill_position(engine: AITrainingEngine) -> None:
    positions = ["utg", "hj", "co", "btn", "sb", "bb"]
    deck = shuffle_deck(make_deck())
    score = 0
    for _ in range(6):
        hole = [deck.pop(), deck.pop()]
        pos = positions[len(deck) % 6]
        sug = engine.suggest_preflop(hole, pos, 5)
        ans = input(f"  {hole[0]} {hole[1]} @ {pos}. Your action (fold/call/raise): ").strip().lower()
        if ans and ans[0] == sug.action[0]:
            score += 1
            print("    Correct.")
        else:
            print("    AI says:", sug.action)
    print("Score:", score, "/ 6")


def _drill_postflop(engine: AITrainingEngine) -> None:
    deck = shuffle_deck(make_deck())
    hole = [deck.pop(), deck.pop()]
    board = [deck.pop(), deck.pop(), deck.pop()]
    sug = engine.suggest_postflop(hole, board, 0.35)
    print("Hole:", hole[0], hole[1], "| Board:", " ".join(str(c) for c in board))
    ans = input("Your action (check/bet/fold): ").strip().lower()
