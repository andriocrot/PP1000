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
