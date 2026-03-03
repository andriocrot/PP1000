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
    if ans and ans[0] == sug.action[0]:
        print("Correct. AI:", sug.action, sug.reasoning[:50])
    else:
        print("AI:", sug.action, sug.reasoning)


def _drill_range_quiz(engine: AITrainingEngine) -> None:
    hands_raise = ["AA", "KK", "QQ", "AKs", "AQs", "JJ", "TT", "AK"]
    hands_fold = ["72o", "83o", "92o", "T2o", "J3o"]
    print("Is this a raise or fold from BTN? (Enter r/f)")
    deck = make_deck()
    random.shuffle(deck)
    for _ in range(3):
        hole = [deck.pop(), deck.pop()]
        sug = engine.suggest_preflop(hole, "btn", 5)
        a = input(f"  {hole[0]} {hole[1]}: ").strip().lower()
        if a == "r" and sug.action == "raise":
            print("    Correct.")
        elif a == "f" and sug.action == "fold":
            print("    Correct.")
        else:
            print("    AI:", sug.action)


def _drill_equity_quiz(engine: AITrainingEngine) -> None:
    print("Which hand has higher equity vs random? 1 or 2?")
    deck = shuffle_deck(make_deck())
    h1 = [deck.pop(), deck.pop()]
    h2 = [deck.pop(), deck.pop()]
    e1 = _monte_carlo_equity(h1, [], 200)
    e2 = _monte_carlo_equity(h2, [], 200)
    print("  Hand 1:", h1[0], h1[1], "  Hand 2:", h2[0], h2[1])
    ans = input("  Answer (1/2): ").strip()
    win1 = e1 > e2
    if (ans == "1" and win1) or (ans == "2" and not win1):
        print("  Correct.")
    else:
        print("  Hand 1 equity:", round(e1, 2), "Hand 2:", round(e2, 2))


def _monte_carlo_equity(hole: List[Card], board: List[Card], trials: int) -> float:
    wins = 0
    used = set(c.to_index() for c in hole) | set(c.to_index() for c in board)
    deck = make_deck()
    for _ in range(trials):
        rest = [c for c in deck if c.to_index() not in used]
        random.shuffle(rest)
        need = 5 - len(board)
        opp_hole = [rest.pop(), rest.pop()]
        board_complete = list(board) + rest[:need]
        our_best = _best_five(list(hole) + board_complete)
        opp_best = _best_five(opp_hole + board_complete)
        if compare_hands(our_best, opp_best) > 0:
            wins += 1
        elif compare_hands(our_best, opp_best) == 0:
            wins += 0.5
    return 100.0 * wins / trials


def _best_five(cards: List[Card]) -> List[Card]:
    best = None
    for combo in combinations(cards, 5):
        c = list(combo)
        if best is None or compare_hands(c, best) > 0:
            best = c
    return best or cards[:5]


# -----------------------------------------------------------------------------
# Range matrix and equity (preflop)
# -----------------------------------------------------------------------------

def range_matrix_raise_btn() -> Dict[str, float]:
    """Approximate raise % from BTN; keys like 'AA', 'AKs', '72o'."""
    out = {}
    for r1 in range(13):
        for r2 in range(13):
            if r1 < r2:
                r1, r2 = r2, r1
            ra = PP1000Constants.RANK_NAMES[r1]
            rb = PP1000Constants.RANK_NAMES[r2]
            suited = "s" if r1 != r2 else ""
            key = ra + rb + suited
            if r1 == r2:
                pct = 100.0 if r1 >= 8 else (60.0 + r1 * 4)
            elif r1 == 12 and r2 >= 10:
                pct = 95.0
            elif r1 >= 10 and r2 >= 9:
                pct = 70.0 + (r1 + r2) * 2
            elif r1 >= 8 and r2 >= 6:
                pct = 40.0 + r1 * 3
            else:
                pct = max(0, 20.0 + r1 * 2 - r2)
            out[key] = min(100.0, max(0.0, pct))
    return out


def range_matrix_fold_utg() -> Dict[str, float]:
    """Fold % from UTG; higher = fold more."""
    out = {}
    for r1 in range(13):
        for r2 in range(13):
            if r1 < r2:
                r1, r2 = r2, r1
            ra = PP1000Constants.RANK_NAMES[r1]
            rb = PP1000Constants.RANK_NAMES[r2]
            suited = "s" if r1 != r2 else ""
            key = ra + rb + suited
            if r1 == r2 and r1 >= 10:
                pct = 5.0
            elif r1 >= 11 and r2 >= 10:
                pct = 10.0
            elif r1 >= 9:
                pct = 35.0 - r1 * 2
            else:
                pct = 80.0 - r1 * 3 + r2
            out[key] = min(100.0, max(0.0, pct))
    return out


def get_hand_key(c1: Card, c2: Card) -> str:
    r1, r2 = int(c1.rank), int(c2.rank)
    if r1 < r2:
        r1, r2 = r2, r1
    ra = PP1000Constants.RANK_NAMES[r1]
    rb = PP1000Constants.RANK_NAMES[r2]
    suited = "s" if c1.suit == c2.suit and r1 != r2 else "o"
    return ra + rb + suited


# -----------------------------------------------------------------------------
# Pot odds and implied odds (simplified)
# -----------------------------------------------------------------------------

def pot_odds(call_amount: float, pot_after_call: float) -> float:
    if pot_after_call <= 0:
        return 0.0
    return call_amount / pot_after_call


def break_even_equity(call_amount: float, pot_before_call: float) -> float:
    total = pot_before_call + call_amount
    if total <= 0:
        return 0.0
    return call_amount / total


def suggested_action_by_equity(equity: float, be: float, margin: float = 0.05) -> str:
    if equity >= be + margin:
        return "call"
    if equity <= be - margin:
        return "fold"
    return "marginal"


# -----------------------------------------------------------------------------
# Level and progress system
# -----------------------------------------------------------------------------

def compute_level(hands_played: int, ai_agreement_count: int) -> int:
    if hands_played == 0:
        return 0
    rate = ai_agreement_count / hands_played
    base = min(PP1000Constants.TRAINING_LEVELS - 1, hands_played // 50)
    bonus = int(rate * 5)
    return min(PP1000Constants.TRAINING_LEVELS - 1, base + bonus)


def level_title(level: int) -> str:
    titles = [
        "Rookie", "Beginner", "Learner", "Improving", "Developing",
        "Intermediate", "Solid", "Advanced", "Strong", "Expert",
        "Sharp", "Pro", "Elite", "Master", "Champion",
        "Legend", "Hall of Fame", "All-Star", "PokerPro", "AI Pro"
    ]
    return titles[min(level, len(titles) - 1)]


# -----------------------------------------------------------------------------
# Export / import
# -----------------------------------------------------------------------------

def export_sessions_csv(path: Optional[str] = None) -> None:
    path = path or str(_ensure_config_dir() / "sessions_export.csv")
    sessions = load_sessions()
    lines = ["session_id,stakes_tier,opened_at,closed_at,hand_count,level"]
    for s in sessions:
        hands = s.get("hands", [])
        level = compute_level(len(hands), sum(1 for h in hands if h.get("action_taken") == h.get("ai_suggestion")))
        lines.append(",".join([
            s.get("session_id", ""),
            str(s.get("stakes_tier", 0)),
            str(s.get("opened_at", 0)),
            str(s.get("closed_at") or ""),
            str(len(hands)),
            str(level),
        ]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Exported to", path)


def import_sessions_from_json(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    existing = load_sessions()
    if isinstance(data, list):
        existing.extend(data)
    else:
        existing.append(data)
    save_sessions(existing)
    print("Imported from", path)


# -----------------------------------------------------------------------------
# PokerPro contract integration (optional web3)
# -----------------------------------------------------------------------------

POKERPRO_ABI_OPEN_SESSION = "openSession(uint8)"
POKERPRO_ABI_CLOSE_SESSION = "closeSession(bytes32)"
POKERPRO_ABI_RECORD_HAND = "recordHand(bytes32,bytes32)"
POKERPRO_ABI_ANCHOR_FEEDBACK = "anchorAIFeedback(bytes32,bytes32,uint8)"
POKERPRO_ABI_SET_STAKES = "setStakesTier(bytes32,uint8)"


def _get_contract_address() -> str:
    config_path = _ensure_config_dir() / PP1000Constants.CONFIG_FILE
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                c = json.load(f)
                return c.get("contract") or PP1000Constants.POKERPRO_DEPLOYED_AT
        except Exception:
            pass
    return PP1000Constants.POKERPRO_DEPLOYED_AT


def _encode_open_session(stakes_tier: int) -> Optional[str]:
    try:
        from eth_abi import encode
        return "0x" + encode(["uint8"], [stakes_tier]).hex()
    except Exception:
        return None


def _encode_close_session(session_id_bytes: bytes) -> Optional[str]:
    try:
        from eth_abi import encode
        return "0x" + encode(["bytes32"], [session_id_bytes]).hex()
    except Exception:
        return None


def anchor_session_to_chain(session_id: str, stakes_tier: int, trainee_address: str) -> bool:
    """Optional: send openSession to PokerPro. Returns True if simulated or sent."""
    addr = _get_contract_address()
    payload = _encode_open_session(stakes_tier)
    if not payload:
        return False
    try:
        from web3 import Web3
        rpc = os.environ.get("PP1000_RPC", PP1000Constants.DEFAULT_RPC)
        w3 = Web3(Web3.HTTPProvider(rpc))
        if not w3.is_connected():
            return False
        selector = _keccak256(POKERPRO_ABI_OPEN_SESSION.encode())[:4]
        data = selector.hex() + payload[2:] if isinstance(selector, bytes) else ""
        tx = {"to": Web3.to_checksum_address(addr), "data": data, "from": trainee_address}
        w3.eth.send_transaction(tx)
        return True
    except Exception:
        return False


def anchor_hand_to_chain(session_id: str, hand_hash_hex: str) -> bool:
    """Optional: recordHand on PokerPro."""
    try:
        from web3 import Web3
        addr = _get_contract_address()
        rpc = os.environ.get("PP1000_RPC", PP1000Constants.DEFAULT_RPC)
        w3 = Web3(Web3.HTTPProvider(rpc))
        if not w3.is_connected():
            return False
        session_bytes = bytes.fromhex(session_id.replace("0x", "").zfill(64)[-64:])
        hand_bytes = bytes.fromhex(hand_hash_hex.replace("0x", "").zfill(64)[-64:])
        payload = _encode_record_hand(session_bytes, hand_bytes)
        if not payload:
            return False
        selector = _keccak256(POKERPRO_ABI_RECORD_HAND.encode())[:4]
        data = selector.hex() + payload[2:] if isinstance(selector, bytes) else ""
        tx = {"to": Web3.to_checksum_address(addr), "data": data}
        w3.eth.send_transaction(tx)
        return True
    except Exception:
        return False


def _encode_record_hand(session_bytes: bytes, hand_bytes: bytes) -> Optional[str]:
    try:
        from eth_abi import encode
        return "0x" + encode(["bytes32", "bytes32"], [session_bytes, hand_bytes]).hex()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Table simulation: deal and run-out
# -----------------------------------------------------------------------------

def deal_hand(deck: List[Card], n_players: int) -> Tuple[List[Card], List[List[Card]], List[Card]]:
    """Returns (board, list of hole cards per player, remaining deck)."""
    hole_cards = []
    for _ in range(n_players):
        hole_cards.append([deck.pop(), deck.pop()])
    flop = [deck.pop(), deck.pop(), deck.pop()]
    turn = deck.pop()
    river = deck.pop()
    board = flop + [turn, river]
    return board, hole_cards, deck


def run_out_showdown(hole_sets: List[List[Card]], board: List[Card]) -> List[int]:
    """Returns list of hand ranks (0 = best) for each player."""
    evaluations = []
    for hole in hole_sets:
        full = list(hole) + list(board)
        if len(full) < 5:
            evaluations.append((0, [0]))
            continue
        best_combo = None
        for combo in combinations(full, 5):
            c = list(combo)
            if best_combo is None or compare_hands(c, best_combo) > 0:
                best_combo = c
        ev = evaluate_hand(best_combo) if best_combo else (0, [0])
        evaluations.append(ev)
    order = sorted(range(len(evaluations)), key=lambda i: (-evaluations[i][0], [-x for x in evaluations[i][1]]))
    rank = [0] * len(order)
    for r, i in enumerate(order):
        rank[i] = r
    return rank


# -----------------------------------------------------------------------------
# Additional AI: turn/river and bluff detection
# -----------------------------------------------------------------------------

class AITurnRiverEngine(AITrainingEngine):
    def suggest_turn(self, hole: Sequence[Card], board: Sequence[Card], pot_frac: float, opp_bet: bool) -> AISuggestion:
        if len(board) < 4:
            return AISuggestion("check", 0.5, 0.0, "Need turn card.", [])
        return self.suggest_postflop(hole, board, pot_frac + (0.1 if opp_bet else 0))

    def suggest_river(self, hole: Sequence[Card], board: Sequence[Card], pot_size: float, to_call: float) -> AISuggestion:
        if len(board) < 5:
            return AISuggestion("check", 0.5, 0.0, "Need river.", [])
        be = break_even_equity(to_call, pot_size - to_call)
        eq = _monte_carlo_equity(list(hole), list(board), 300)
        eq_pct = eq / 100.0
        if eq_pct >= be + 0.08:
            return AISuggestion("call", 0.8, eq_pct - be, "Equity above break-even; call.", ["raise", "fold"])
        if eq_pct <= be - 0.08:
            return AISuggestion("fold", 0.75, be - eq_pct, "Equity below break-even; fold.", ["call"])
        return AISuggestion("marginal", 0.5, 0.0, "Close spot; consider pot odds.", ["call", "fold"])


def bluff_frequency_suggestion(position: str, board_texture: str) -> float:
    """Suggested bluff frequency (0–1) by position and board."""
    pos_freq = {"btn": 0.45, "co": 0.35, "hj": 0.28, "utg": 0.2, "sb": 0.4, "bb": 0.25}
    tex_freq = {"dry": 0.35, "wet": 0.25, "paired": 0.2, "monotone": 0.15}
    p = pos_freq.get(position.lower(), 0.3)
    t = tex_freq.get(board_texture.lower(), 0.3)
    return (p + t) / 2


# -----------------------------------------------------------------------------
# Help and training tips
# -----------------------------------------------------------------------------

HELP_TOPICS = {
    "preflop": "Preflop: Play tighter from early position (UTG, HJ). Raise premium pairs and strong broadway. Call more from BTN/CO with suited connectors.",
    "postflop": "Postflop: Value bet when you have a strong hand. Check when marginal. Consider pot odds before calling.",
    "position": "Position: Later position = more hands to play. BTN is most profitable. Play fewer hands from SB/BB without premium.",
    "equity": "Equity: Your share of the pot when all-in. Use break-even equity (call / (pot + call)) to decide calls.",
    "ranges": "Ranges: Think in ranges, not single hands. Tighten vs raises, loosen when you have position.",
    "bluff": "Bluffing: Bluff more on dry boards and when you have position. Balance value and bluff for GTO.",
    "drills": "Drills: Use preflop and postflop drills to build muscle memory. Range quiz improves hand reading.",
}


def show_help(topic: Optional[str] = None) -> None:
    if topic:
        print(HELP_TOPICS.get(topic.lower(), "Unknown topic. Try: preflop, postflop, position, equity, ranges, bluff, drills"))
    else:
        print("Available help topics:", ", ".join(HELP_TOPICS.keys()))
        t = input("Topic: ").strip()
        show_help(t or None)


# -----------------------------------------------------------------------------
# Range display (text grid)
# -----------------------------------------------------------------------------

def range_grid_compact(hand_keys: List[str]) -> str:
    """Compact grid of which hands are in range; 13x13 with suited/off suited."""
    grid = [["." for _ in range(13)] for _ in range(13)]
    for key in hand_keys:
        if len(key) >= 2:
            r1 = PP1000Constants.RANK_NAMES.index(key[0]) if key[0] in PP1000Constants.RANK_NAMES else -1
            r2 = PP1000Constants.RANK_NAMES.index(key[1]) if len(key) > 1 and key[1] in PP1000Constants.RANK_NAMES else -1
            if r1 >= 0 and r2 >= 0:
                if r1 < r2:
                    r1, r2 = r2, r1
                grid[r1][r2] = "s" if "s" in key else "o"
    lines = ["   " + "".join(PP1000Constants.RANK_NAMES)]
    for i, row in enumerate(grid):
        lines.append(PP1000Constants.RANK_NAMES[i] + " " + "".join(row))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Quiz bank for drills
# -----------------------------------------------------------------------------

QUIZ_PREFLOP = [
    {"hole": "AA", "position": "utg", "answer": "raise"},
    {"hole": "72o", "position": "btn", "answer": "fold"},
    {"hole": "AKs", "position": "co", "answer": "raise"},
    {"hole": "T9s", "position": "btn", "answer": "call"},
    {"hole": "33", "position": "utg", "answer": "fold"},
    {"hole": "KQs", "position": "hj", "answer": "raise"},
    {"hole": "JTo", "position": "co", "answer": "call"},
    {"hole": "A2o", "position": "bb", "answer": "fold"},
    {"hole": "QQ", "position": "utg", "answer": "raise"},
    {"hole": "98s", "position": "btn", "answer": "call"},
]


def _shorthand_to_cards(sh: str) -> List[Card]:
    sh = sh.strip().upper()
    if len(sh) < 2:
        raise ValueError("Need at least 2 chars")
    rank_map = {c: Rank(i) for i, c in enumerate(PP1000Constants.RANK_NAMES)}
    r1 = rank_map.get(sh[0])
    r2 = rank_map.get(sh[1]) if len(sh) > 1 else r1
    if r1 is None or r2 is None:
        raise ValueError("Invalid rank")
    suited = "s" in sh or (len(sh) == 2 and r1 == r2)
    if suited or r1 == r2:
        return [Card(r1, Suit.CLUBS), Card(r2, Suit.DIAMONDS if r1 != r2 else Suit.HEARTS)]
    return [Card(r1, Suit.CLUBS), Card(r2, Suit.SPADES)]


def run_quiz_bank(engine: AITrainingEngine) -> None:
    print("Preflop quiz: 10 questions. Enter raise/call/fold.")
    score = 0
    for i, q in enumerate(QUIZ_PREFLOP):
        hole_str = q["hole"]
        try:
            hole = _shorthand_to_cards(hole_str)
            c1, c2 = hole[0], hole[1]
        except Exception:
            c1, c2 = Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)
        sug = engine.suggest_preflop([c1, c2], q["position"], 5)
        ans = input("  Q{} {} @ {}: ".format(i + 1, hole_str, q["position"])).strip().lower()
        if ans and ans[0] == sug.action[0]:
            score += 1
            print("    Correct.")
        else:
            print("    AI:", sug.action)
    print("Score:", score, "/", len(QUIZ_PREFLOP))


# -----------------------------------------------------------------------------
# Hand strength tiers (for display)
# -----------------------------------------------------------------------------

def hand_strength_tier(hole: Sequence[Card], board: Sequence[Card]) -> int:
    """0–9; 9 = nuts, 0 = trash."""
    if len(board) < 3:
        r1, r2 = int(hole[0].rank), int(hole[1].rank)
        pair = r1 == r2
        high = max(r1, r2)
        if pair and high >= 11:
            return 9
        if pair and high >= 8:
            return 7
        if high >= 11 and min(r1, r2) >= 9:
            return 6
        if high >= 9:
            return 4
        return 2
    full = list(hole) + list(board)
    ht, _ = evaluate_hand(full)
    return min(9, ht + 1)


# -----------------------------------------------------------------------------
# Session summary report
# -----------------------------------------------------------------------------

def session_summary_report(session_id: str) -> Optional[str]:
    sessions = load_sessions()
    for s in sessions:
        if s.get("session_id") == session_id:
            hands = s.get("hands", [])
            match = sum(1 for h in hands if h.get("action_taken") == h.get("ai_suggestion"))
            level = compute_level(len(hands), match)
            lines = [
                "Session: " + session_id[:20] + "...",
                "Stakes tier: " + str(s.get("stakes_tier", 0)),
                "Hands: " + str(len(hands)),
                "AI agreement: " + str(match) + " / " + str(len(hands)),
                "Level: " + level_title(level),
            ]
            return "\n".join(lines)
    return None


# -----------------------------------------------------------------------------
# Validate card input
# -----------------------------------------------------------------------------

def parse_cards_line(line: str, min_cards: int = 0, max_cards: int = 7) -> List[Card]:
    cards = []
    for part in line.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        try:
            cards.append(Card.from_string(part))
        except ValueError:
            continue
        if len(cards) >= max_cards:
            break
    if min_cards and len(cards) < min_cards:
        raise ValueError("Need at least {} cards".format(min_cards))
    return cards


# -----------------------------------------------------------------------------
# Batch hand evaluation (for analysis)
# -----------------------------------------------------------------------------

def batch_evaluate_hands(hand_list: List[List[Card]]) -> List[Tuple[int, List[int]]]:
    return [evaluate_hand(h) for h in hand_list]


def batch_compare_winner(hands: List[List[Card]]) -> int:
    """Returns index of winning hand."""
    evals = [evaluate_hand(h) for h in hands]
    best_i = 0
    for i in range(1, len(evals)):
        if compare_hands(hands[i], hands[best_i]) > 0:
            best_i = i
    return best_i


# -----------------------------------------------------------------------------
# Extended simulation: full table run-out and EV
# -----------------------------------------------------------------------------

def simulate_n_hand_ev(hole: List[Card], n_opponents: int, n_trials: int) -> float:
    """Win rate (0–1) vs n opponents with random hands and random board."""
    wins = 0
    deck = make_deck()
    used = set(c.to_index() for c in hole)
    for _ in range(n_trials):
        rest = [c for c in deck if c.to_index() not in used]
        random.shuffle(rest)
        opp_holes = [[rest.pop(), rest.pop()] for _ in range(n_opponents)]
        board = [rest.pop() for _ in range(5)]
        ranks = run_out_showdown([hole] + opp_holes, board)
        our_rank = ranks[0]
        n_winners = sum(1 for r in ranks if r == 0)
        if our_rank == 0 and n_winners > 1:
            wins += 0.5
        elif our_rank == 0:
            wins += 1
    return wins / n_trials


def simulate_hand_vs_range(hole: List[Card], range_keys: List[str], board: List[Card], trials: int) -> float:
    """Equity vs a range (hand keys like AA, AKs). Board can be empty for preflop."""
    deck = make_deck()
    used = set(c.to_index() for c in hole) | set(c.to_index() for c in board)
    range_cards = _range_keys_to_combos(range_keys)
    if not range_cards:
        return 0.5
    wins = 0
    for _ in range(trials):
        rest = [c for c in deck if c.to_index() not in used]
        random.shuffle(rest)
        opp = random.choice(range_cards)
        opp_idx = set(c.to_index() for c in opp)
        if opp_idx & used:
            continue
        need_board = 5 - len(board)
        board_full = list(board) + [c for c in rest if c.to_index() not in opp_idx][:need_board]
        our_best = _best_five(list(hole) + board_full)
        opp_best = _best_five(list(opp) + board_full)
        cmp = compare_hands(our_best, opp_best)
        if cmp > 0:
            wins += 1
        elif cmp == 0:
            wins += 0.5
    return wins / trials


def _range_keys_to_combos(keys: List[str]) -> List[List[Card]]:
    combos = []
    deck = make_deck()
    for key in keys:
        try:
            combos.append(_shorthand_to_cards(key))
        except Exception:
            pass
    return combos


# -----------------------------------------------------------------------------
# Config load/save helpers
# -----------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    path = _ensure_config_dir() / PP1000Constants.CONFIG_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config_key(key: str, value: Any) -> None:
    config = load_config()
    config[key] = value
    path = _ensure_config_dir() / PP1000Constants.CONFIG_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# -----------------------------------------------------------------------------
# Hand history export (per-session)
# -----------------------------------------------------------------------------

def export_session_hands_csv(session_id: str, path: Optional[str] = None) -> None:
    sessions = load_sessions()
    for s in sessions:
        if s.get("session_id") == session_id:
            path = path or str(_ensure_config_dir() / ("session_" + session_id[:8] + ".csv"))
            lines = ["hand_id,hole_1,hole_2,board,action,ai_suggestion,quality_band,timestamp"]
            for h in s.get("hands", []):
                hole = h.get("hole", ["", ""])
                board = ",".join(h.get("board", []))
                lines.append(",".join([
                    h.get("hand_id", ""),
                    hole[0] if len(hole) > 0 else "",
                    hole[1] if len(hole) > 1 else "",
                    board,
                    h.get("action_taken", ""),
                    h.get("ai_suggestion", ""),
                    str(h.get("quality_band", 0)),
                    str(h.get("timestamp", 0)),
                ]))
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print("Exported to", path)
            return
    print("Session not found.")


# -----------------------------------------------------------------------------
# AI suggestion with explanation (long form)
# -----------------------------------------------------------------------------

def explain_suggestion(sug: AISuggestion, context: str) -> str:
    lines = [
        "Action: " + sug.action,
        "Confidence: {:.0%}".format(sug.confidence),
        "Reasoning: " + sug.reasoning,
    ]
    if sug.ev_estimate is not None:
        lines.append("EV estimate: {:.2f}".format(sug.ev_estimate))
    if sug.alternatives:
        lines.append("Alternatives: " + ", ".join(sug.alternatives))
    lines.append("Context: " + context)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Position constants for UI
# -----------------------------------------------------------------------------

POSITION_ORDER = ["utg", "utg+1", "hj", "co", "btn", "sb", "bb"]


def position_index(name: str) -> int:
    try:
        return POSITION_ORDER.index(name.lower())
    except ValueError:
        return 3


# -----------------------------------------------------------------------------
# Stakes tier labels
# -----------------------------------------------------------------------------

def stakes_tier_label(tier: int) -> str:
    labels = [
        "Micro", "Micro+", "Low", "Low+", "Mid", "Mid+", "High", "High+", "High Stakes", "Top", "Elite"
    ]
    return labels[min(max(0, tier), len(labels) - 1)]


# -----------------------------------------------------------------------------
# Session list with pagination
# -----------------------------------------------------------------------------

def list_sessions_paginated(page: int = 0, page_size: int = 10) -> List[Dict[str, Any]]:
    sessions = load_sessions()
    start = page * page_size
    return sessions[-(start + page_size):-(start) if start else None] if start < len(sessions) else []


# -----------------------------------------------------------------------------
# Hand replay (re-evaluate and show AI again)
# -----------------------------------------------------------------------------

def replay_hand(hand_record: Dict[str, Any], engine: AITrainingEngine) -> AISuggestion:
    hole_str = hand_record.get("hole", [])
    board_str = hand_record.get("board", [])
    tier = hand_record.get("stakes_tier", 5)
    if len(hole_str) < 2:
        return AISuggestion("fold", 0.0, None, "Invalid hand.", [])
    try:
        hole = [Card.from_string(hole_str[0]), Card.from_string(hole_str[1])]
        board = [Card.from_string(b) for b in board_str] if board_str else []
        if board:
            return engine.suggest_postflop(hole, board, 0.3)
        return engine.suggest_preflop(hole, "btn", tier)
    except ValueError:
        return AISuggestion("fold", 0.0, None, "Invalid cards.", [])


# -----------------------------------------------------------------------------
# Quality band distribution (for stats)
# -----------------------------------------------------------------------------

def quality_band_distribution(sessions: List[Dict[str, Any]]) -> Dict[int, int]:
    dist: Dict[int, int] = {}
    for s in sessions:
        for h in s.get("hands", []):
            b = h.get("quality_band", 0)
            dist[b] = dist.get(b, 0) + 1
    return dist


# -----------------------------------------------------------------------------
# Daily goal tracking (placeholder)
# -----------------------------------------------------------------------------

def daily_goal_hands() -> int:
    return 50


def daily_goal_progress() -> Tuple[int, int]:
    sessions = load_sessions()
    today_start = time.time() - (time.time() % 86400)
    today_hands = 0
    for s in sessions:
        for h in s.get("hands", []):
            if h.get("timestamp", 0) >= today_start:
                today_hands += 1
    return today_hands, daily_goal_hands()


# -----------------------------------------------------------------------------
# Card formatting for display
# -----------------------------------------------------------------------------

def format_card_short(c: Card) -> str:
    return str(c)


def format_hand_short(hole: Sequence[Card]) -> str:
    return " ".join(format_card_short(c) for c in hole)


def format_board_short(board: Sequence[Card]) -> str:
    return " ".join(format_card_short(c) for c in board)


# -----------------------------------------------------------------------------
# Random hand generator (for drills)
# -----------------------------------------------------------------------------

def random_hand(rng: Optional[random.Random] = None) -> List[Card]:
    deck = shuffle_deck(make_deck(), rng)
    return [deck.pop(), deck.pop()]


def random_board(deck: List[Card], rng: Optional[random.Random] = None) -> List[Card]:
    r = rng or random
    out = []
    for _ in range(5):
        out.append(deck.pop())
    return out


# -----------------------------------------------------------------------------
# Equity vs random (preflop)
# -----------------------------------------------------------------------------

def preflop_equity_vs_random(hole: List[Card], n_trials: int = 500) -> float:
    return _monte_carlo_equity(hole, [], n_trials)


# -----------------------------------------------------------------------------
# Session ID generation (PokerPro-compatible style)
# -----------------------------------------------------------------------------

def generate_session_id() -> str:
    return secrets.token_hex(32)


def session_id_to_bytes(session_id: str) -> bytes:
    h = session_id.replace("0x", "").zfill(64)[-64:]
    return bytes.fromhex(h)


# -----------------------------------------------------------------------------
# Feedback hash for PokerPro anchor
# -----------------------------------------------------------------------------

def compute_feedback_hash(hand_id: str, quality_band: int, ai_suggestion: str) -> str:
    return feedback_hash_for_contract(hand_id, quality_band, ai_suggestion)


def compute_hand_hash(hole: List[str], board: List[str], session_salt: str) -> str:
    return hand_hash_for_contract(hole, board, session_salt)


# -----------------------------------------------------------------------------
# Validation: duplicate cards
# -----------------------------------------------------------------------------

def has_duplicate_cards(cards: Sequence[Card]) -> bool:
    seen = set()
    for c in cards:
        idx = c.to_index()
        if idx in seen:
            return True
        seen.add(idx)
    return False


# -----------------------------------------------------------------------------
# Best hand from 7 cards (wrapper)
# -----------------------------------------------------------------------------

def best_hand_from_seven(cards: List[Card]) -> Tuple[List[Card], int, List[int]]:
    if len(cards) != 7:
