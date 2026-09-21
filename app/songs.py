"""Nursery rhyme catalog for the 아기 동요 앱.

The lyrics bundled here are traditional Korean children's songs that are in the
public domain. Each entry is intentionally simple so the data layer stays easy
to test and extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class Song:
    """A single nursery rhyme."""

    id: str
    title: str
    emoji: str
    lyrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


SONGS: list[Song] = [
    Song(
        id="gom-se-mari",
        title="곰 세 마리",
        emoji="🐻",
        lyrics=[
            "곰 세 마리가 한 집에 있어",
            "아빠 곰 엄마 곰 애기 곰",
            "아빠 곰은 뚱뚱해",
            "엄마 곰은 날씬해",
            "애기 곰은 너무 귀여워",
            "으쓱으쓱 잘한다",
        ],
    ),
    Song(
        id="nabiya",
        title="나비야",
        emoji="🦋",
        lyrics=[
            "나비야 나비야 이리 날아 오너라",
            "노랑나비 흰나비 춤을 추며 오너라",
            "봄바람에 꽃잎도 방긋방긋 웃으며",
            "참새도 짹짹짹 노래하며 춤춘다",
        ],
    ),
    Song(
        id="banjjak-byeol",
        title="반짝반짝 작은 별",
        emoji="⭐",
        lyrics=[
            "반짝반짝 작은 별 아름답게 비추네",
            "동쪽 하늘에서도 서쪽 하늘에서도",
            "반짝반짝 작은 별 아름답게 비추네",
        ],
    ),
    Song(
        id="santokki",
        title="산토끼",
        emoji="🐰",
        lyrics=[
            "산토끼 토끼야 어디를 가느냐",
            "깡충깡충 뛰면서 어디를 가느냐",
            "산 고개 고개를 나 혼자 넘어서",
            "토실토실 알밤을 주워서 올 테야",
        ],
    ),
    Song(
        id="hakgyojong",
        title="학교종",
        emoji="🔔",
        lyrics=[
            "학교종이 땡땡땡 어서 모이자",
            "선생님이 우리를 기다리신다",
            "학교종이 땡땡땡 어서 모이자",
            "사이좋게 오늘도 공부 잘하자",
        ],
    ),
]


def list_songs() -> list[Song]:
    """Return every song in catalog order."""

    return list(SONGS)


def get_song(song_id: str) -> Optional[Song]:
    """Return a single song by id, or ``None`` when it does not exist."""

    for song in SONGS:
        if song.id == song_id:
            return song
    return None
