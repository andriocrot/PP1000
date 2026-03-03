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
