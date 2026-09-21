#!/usr/bin/env python3
"""Render 30 public-domain nursery rhymes to MP3 + timed bilingual lyrics."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "public" / "audio"
DATA_DIR = ROOT / "src" / "data"

NOTE_RE = re.compile(r"^(Rest|[A-G]#?\d)([qehwst]\.?)$")
BEAT = {"q": 1.0, "e": 0.5, "h": 2.0, "w": 4.0, "s": 0.25, "t": 1 / 3}
PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_of(token: str) -> int | None:
    if token == "Rest":
        return None
    name, octv = token[:-1], int(token[-1])
    sharp = name.endswith("#")
    base = PITCH[name[0]] + (1 if sharp else 0)
    return 12 * (octv + 1) + base


def parse_notes(spec: str) -> list[tuple[int | None, float]]:
    notes = []
    for tok in spec.split():
        m = NOTE_RE.match(tok)
        if not m:
            raise ValueError(f"Bad note token: {tok}")
        dur = BEAT[m.group(2)[0]] * (1.5 if m.group(2).endswith(".") else 1.0)
        notes.append((midi_of(m.group(1)), dur))
    return notes


def ads(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    env = np.zeros(n, dtype=np.float64)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    ns = max(0, n - na - nd - nr)
    i = 0
    if na:
        env[i : i + na] = np.linspace(0, 1, na, endpoint=False)
        i += na
    if nd:
        env[i : i + nd] = np.linspace(1, s, nd, endpoint=False)
        i += nd
    if ns:
        env[i : i + ns] = s
        i += ns
    if nr and i < n:
        env[i:] = np.linspace(env[i - 1] if i else s, 0, n - i)
    return env


def tone(freq: float, n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    if kind == "glock":
        decay = np.exp(-t * 6.4)
        sig = (
            np.sin(2 * math.pi * freq * t)
            + 0.55 * np.sin(2 * math.pi * freq * 2.01 * t)
            + 0.22 * np.sin(2 * math.pi * freq * 3.02 * t)
            + 0.08 * np.sin(2 * math.pi * freq * 5.04 * t)
        )
        ping = np.exp(-t * 90) * np.sin(2 * math.pi * freq * 6.8 * t) * 0.12
        return (sig * decay + ping) * ads(n, 0.002, 0.03, 0.35, 0.1)
    if kind == "xylo":
        decay = np.exp(-t * 7.2)
        sig = np.sin(2 * math.pi * freq * t) + 0.32 * np.sin(2 * math.pi * freq * 3.02 * t)
        return sig * decay * ads(n, 0.001, 0.025, 0.28, 0.08)
    if kind == "toy":
        decay = np.exp(-t * 4.6)
        sig = (
            np.sin(2 * math.pi * freq * t)
            + 0.28 * np.sin(2 * math.pi * freq * 2 * t)
            + 0.1 * np.sin(2 * math.pi * freq * 4.02 * t)
        )
        return sig * decay * ads(n, 0.004, 0.04, 0.42, 0.1)
    if kind == "pad":
        sig = 0.55 * np.sin(2 * math.pi * freq * t) + 0.2 * np.sin(2 * math.pi * freq * 2 * t)
        return sig * ads(n, 0.02, 0.06, 0.35, 0.16)
    decay = np.exp(-t * 3.8)
    sig = np.sin(2 * math.pi * freq * t) + 0.2 * np.sin(2 * math.pi * freq * 2 * t)
    return sig * decay * ads(n, 0.004, 0.05, 0.4, 0.12)


def woodblock(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    click = np.sin(2 * math.pi * 980 * t) * np.exp(-t * 55)
    noise = rng.normal(0, 1, n) * np.exp(-t * 90)
    return (click * 0.35 + noise * 0.12)


def hz(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def render_song(song: dict) -> tuple[np.ndarray, list[dict], float]:
    notes = parse_notes(song["notes"])
    bpm = max(int(round(song["bpm"] * 1.22)), 112)
    beat_sec = 60.0 / bpm
    lead = 0.28
    tail = 0.55
    transpose = 7
    total_beats = sum(d for _, d in notes)
    duration = lead + total_beats * beat_sec + tail
    n_total = int(duration * SR) + 1
    mix = np.zeros(n_total, dtype=np.float64)
    rng = np.random.default_rng(abs(hash(song["id"])) % (2**32))
    perc = True

    beat_pos = 0.0
    events: list[tuple[float, float, int | None]] = []
    for midi, dur in notes:
        start = lead + beat_pos * beat_sec
        end = start + dur * beat_sec
        events.append((start, end, midi))
        beat_pos += dur
        if midi is None:
            continue
        midi = min(midi + transpose, 98)
        n = int((end - start + 0.1) * SR)
        i0 = int(start * SR)
        i1 = min(n_total, i0 + n)
        n = i1 - i0
        if n <= 0:
            continue
        melody = tone(hz(midi), n, "glock", rng)
        sparkle = tone(hz(midi), n, "xylo", rng)
        toy = tone(hz(midi), n, "toy", rng)
        bass = tone(hz(max(midi - 12, 40)), n, "pad", rng) * 0.22
        third = 4 if (midi % 12) in (0, 5, 7) else 3
        harm = tone(hz(min(midi + third, 100)), n, "glock", rng) * 0.16
        mix[i0:i1] += 0.4 * melody + 0.34 * sparkle + 0.22 * toy + bass + harm

        if abs(beat_pos - round(beat_pos)) < 0.08:
            pn = min(int(0.06 * SR), n_total - i0)
            mix[i0 : i0 + pn] += woodblock(pn, rng)

    peak = np.max(np.abs(mix)) or 1.0
    mix = mix / peak * 0.9
    fade = int(0.04 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)

    lyrics = []
    lines = song["lines"]
    last_span = 4.0
    if len(lines) >= 2:
        last_span = max(4.0, float(lines[-1]["beat"] - lines[-2]["beat"]))
    needed = float(lines[-1]["beat"]) + last_span
    scale = total_beats / needed if needed > total_beats else 1.0
    for i, line in enumerate(lines):
        start_beat = float(line["beat"]) * scale
        if i + 1 < len(lines):
            end_beat = float(lines[i + 1]["beat"]) * scale
        else:
            end_beat = total_beats
        start = lead + start_beat * beat_sec
        end = lead + end_beat * beat_sec
        lyrics.append(
            {
                "start": round(start, 3),
                "end": round(min(max(end, start + 0.4), duration - 0.15), 3),
                "ko": line["ko"],
                "en": line["en"],
            }
        )
    return mix, lyrics, duration


def wav_to_mp3(wav: np.ndarray, dest: Path) -> None:
    raw = np.clip(wav, -1, 1)
    pcm = (raw * 32767).astype("<i2").tobytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".wav")
    header = _wav_header(len(pcm))
    tmp.write_bytes(header + pcm)
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "5",
            "-ac",
            "1",
            str(dest),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp.unlink()


def _wav_header(data_bytes: int) -> bytes:
    import struct

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        SR,
        SR * 2,
        2,
        16,
        b"data",
        data_bytes,
    )


SONGS = [
    {
        "id": "twinkle",
        "titleKo": "반짝반짝 작은별",
        "titleEn": "Twinkle Twinkle Little Star",
        "emoji": "⭐",
        "korean": True,
        "category": "play",
        "color": "#F7D6E8",
        "accent": "#E891B5",
        "bpm": 88,
        "style": "lullaby",
        "notes": (
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "G4q G4q F4q F4q E4q E4q D4h "
            "G4q G4q F4q F4q E4q E4q D4h "
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "반짝반짝 작은별", "en": "Twinkle, twinkle, little star"},
            {"beat": 8, "ko": "아름답게 비치네", "en": "How I wonder what you are"},
            {"beat": 16, "ko": "서쪽 하늘 위에서", "en": "Up above the world so high"},
            {"beat": 24, "ko": "반짝반짝 작은별", "en": "Like a diamond in the sky"},
            {"beat": 32, "ko": "반짝반짝 작은별", "en": "Twinkle, twinkle, little star"},
            {"beat": 40, "ko": "아름답게 비치네", "en": "How I wonder what you are"},
            {"beat": 48, "ko": "반짝반짝 작은별", "en": "Twinkle, twinkle, little star"},
            {"beat": 56, "ko": "아름답게 비치네", "en": "How I wonder what you are"},
        ],
    },
    {
        "id": "butterfly",
        "titleKo": "나비야",
        "titleEn": "Nabiya (Butterfly)",
        "emoji": "🦋",
        "korean": True,
        "category": "animal",
        "color": "#E4D7FA",
        "accent": "#B79BE8",
        "bpm": 96,
        "style": "play",
        "perc": True,
        "notes": (
            "E4e G4e G4e A4q G4e E4q. "
            "E4e G4e G4e A4q G4e E4q. "
            "E4e G4e G4e A4q G4e C5q. "
            "A4e G4q E4h Restq "
            "E4e G4e G4e A4q G4e E4q. "
            "E4e G4e G4e A4q G4e E4q. "
            "E4e G4e G4e A4q G4e C5q. "
            "A4e G4q E4h Restq"
        ),
        "lines": [
            {"beat": 0, "ko": "나비야 나비야", "en": "Butterfly, butterfly"},
            {"beat": 6, "ko": "이리 날아오너라", "en": "Come flying over here"},
            {"beat": 12, "ko": "노랑나비 흰나비", "en": "Yellow one, white one"},
            {"beat": 18, "ko": "춤을 추며 오너라", "en": "Come dancing through the air"},
            {"beat": 26, "ko": "봄바람에 꽃잎도", "en": "Flower petals in the breeze"},
            {"beat": 32, "ko": "방긋방긋 웃으며", "en": "Smile and bloom so sweetly"},
            {"beat": 38, "ko": "참새도 짹짹짹", "en": "Sparrows chirp tweet-tweet"},
            {"beat": 44, "ko": "노래하며 춤춘다", "en": "Singing as they dance along"},
        ],
    },
    {
        "id": "forsythia",
        "titleKo": "나리나리 개나리",
        "titleEn": "Nari Nari Forsythia",
        "emoji": "🌼",
        "korean": True,
        "category": "play",
        "color": "#FBE7A8",
        "accent": "#E8C85A",
        "bpm": 100,
        "style": "play",
        "perc": True,
        "notes": (
            "G4e A4e G4e E4q G4e A4e G4e E4q "
            "C5e C5e D5e E5q D5e C5q. "
            "G4e A4e G4e E4q G4e A4e G4e E4q "
            "C4e D4e E4q D4e C4h "
            "G4e A4e G4e E4q G4e A4e G4e E4q "
            "C5e C5e D5e E5q D5e C5q. "
            "G4e A4e G4e E4q G4e A4e G4e E4q "
            "C4e D4e E4q D4e C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "나리나리 개나리", "en": "Nari nari, forsythia"},
            {"beat": 6, "ko": "입에 따다 물고요", "en": "Pick a bloom and hold it close"},
            {"beat": 12, "ko": "병아리떼 하영하영", "en": "Little chicks go yellow-yellow"},
            {"beat": 18, "ko": "봄나들이 가요", "en": "Off we go for a springtime walk"},
            {"beat": 26, "ko": "나리나리 개나리", "en": "Nari nari, forsythia"},
            {"beat": 32, "ko": "노란 꽃이 피었네", "en": "Yellow blossoms everywhere"},
            {"beat": 38, "ko": "병아리떼 하영하영", "en": "Little chicks go yellow-yellow"},
            {"beat": 44, "ko": "봄나들이 가요", "en": "Off we go for a springtime walk"},
        ],
    },
    {
        "id": "brother-john",
        "titleKo": "학교종",
        "titleEn": "Frère Jacques / Are You Sleeping",
        "emoji": "🔔",
        "korean": True,
        "category": "daily",
        "color": "#D5EAFB",
        "accent": "#7FB6E8",
        "bpm": 92,
        "style": "play",
        "notes": (
            "C4q D4q E4q C4q C4q D4q E4q C4q "
            "E4q F4q G4h E4q F4q G4h "
            "G4e A4e G4e F4e E4q C4q G4e A4e G4e F4e E4q C4q "
            "C4q G3q C4h C4q G3q C4h "
            "C4q D4q E4q C4q C4q D4q E4q C4q "
            "E4q F4q G4h E4q F4q G4h "
            "G4e A4e G4e F4e E4q C4q G4e A4e G4e F4e E4q C4q "
            "C4q G3q C4h C4q G3q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "학교종이 땡땡땡", "en": "Are you sleeping, are you sleeping"},
            {"beat": 8, "ko": "어서 모이자", "en": "Brother John, Brother John"},
            {"beat": 16, "ko": "선생님이 우리를", "en": "Morning bells are ringing"},
            {"beat": 24, "ko": "기다리신다", "en": "Ding ding dong, ding ding dong"},
            {"beat": 32, "ko": "학교종이 땡땡땡", "en": "Are you sleeping, are you sleeping"},
            {"beat": 40, "ko": "어서 모이자", "en": "Brother John, Brother John"},
            {"beat": 48, "ko": "선생님이 우리를", "en": "Morning bells are ringing"},
            {"beat": 56, "ko": "기다리신다", "en": "Ding ding dong, ding ding dong"},
        ],
    },
    {
        "id": "fox",
        "titleKo": "여우야 여우야",
        "titleEn": "Fox, Fox",
        "emoji": "🦊",
        "korean": True,
        "category": "play",
        "color": "#FFD7B8",
        "accent": "#F0A05A",
        "bpm": 118,
        "style": "play",
        "perc": True,
        "notes": (
            "G4e G4e A4q G4e E4q. "
            "G4e G4e A4q G4e E4q. "
            "C5e C5e D5q C5e A4q. "
            "G4e A4q G4e E4h "
            "G4e G4e A4q G4e E4q. "
            "G4e G4e A4q G4e E4q. "
            "C5e C5e D5q C5e A4q. "
            "G4e A4q G4e E4h"
        ),
        "lines": [
            {"beat": 0, "ko": "여우야 여우야 뭐하니", "en": "Fox, fox, what are you doing"},
            {"beat": 6, "ko": "잠잔다", "en": "I'm sleeping"},
            {"beat": 12, "ko": "뭐 잡으러 왔니", "en": "What did you come to catch"},
            {"beat": 18, "ko": "토끼 잡으러 왔다", "en": "I came to catch a bunny"},
            {"beat": 26, "ko": "여우야 여우야 뭐하니", "en": "Fox, fox, what are you doing"},
            {"beat": 32, "ko": "잠잔다", "en": "I'm sleeping"},
            {"beat": 38, "ko": "뭐 잡으러 왔니", "en": "What did you come to catch"},
            {"beat": 44, "ko": "토끼 잡으러 왔다", "en": "I came to catch a bunny"},
        ],
    },
    {
        "id": "moon",
        "titleKo": "달아달아 밝은달아",
        "titleEn": "Bright Moon",
        "emoji": "🌕",
        "korean": True,
        "category": "play",
        "color": "#FFF3B0",
        "accent": "#E8C84A",
        "bpm": 112,
        "style": "play",
        "perc": True,
        "notes": (
            "C5q C5q A4q A4q G4q E4q G4h "
            "C5q C5q A4q A4q G4q E4q C4h "
            "G4q A4q G4q E4q G4q A4q G4h "
            "C5q A4q G4q E4q D4q E4q C4h "
            "C5q C5q A4q A4q G4q E4q G4h "
            "C5q C5q A4q A4q G4q E4q C4h "
            "G4q A4q G4q E4q G4q A4q G4h "
            "C5q A4q G4q E4q D4q E4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "달아달아 밝은 달아", "en": "Moon, moon, bright moon"},
            {"beat": 8, "ko": "이태백이 놀던 달아", "en": "The moon where poets played"},
            {"beat": 16, "ko": "저기 저기 저 달 속에", "en": "Look, look inside that moon"},
            {"beat": 24, "ko": "계수나무 박혔으니", "en": "A cinnamon tree is growing"},
            {"beat": 32, "ko": "달아달아 밝은 달아", "en": "Moon, moon, bright moon"},
            {"beat": 40, "ko": "방긋방긋 웃어라", "en": "Smile a great big smile"},
            {"beat": 48, "ko": "아기랑 손뼉 치며", "en": "Clap along with baby"},
            {"beat": 56, "ko": "둥실둥실 떠요", "en": "Float up, float up high"},
        ],
    },
    {
        "id": "mary-lamb",
        "titleKo": "메리의 어린 양",
        "titleEn": "Mary Had a Little Lamb",
        "emoji": "🐑",
        "category": "animal",
        "color": "#F8E6D4",
        "accent": "#E8B48A",
        "bpm": 100,
        "style": "play",
        "perc": True,
        "notes": (
            "E4q D4q C4q D4q E4q E4q E4h "
            "D4q D4q D4h E4q G4q G4h "
            "E4q D4q C4q D4q E4q E4q E4q E4q "
            "D4q D4q E4q D4q C4w "
            "E4q D4q C4q D4q E4q E4q E4h "
            "D4q D4q D4h E4q G4q G4h "
            "E4q D4q C4q D4q E4q E4q E4q E4q "
            "D4q D4q E4q D4q C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "메리에게 어린 양이", "en": "Mary had a little lamb"},
            {"beat": 8, "ko": "눈처럼 하얗고", "en": "Its fleece was white as snow"},
            {"beat": 16, "ko": "메리가 가는 곳마다", "en": "And everywhere that Mary went"},
            {"beat": 24, "ko": "양이 따라가요", "en": "The lamb was sure to go"},
            {"beat": 32, "ko": "어느 날 학교까지", "en": "He followed her to school one day"},
            {"beat": 40, "ko": "따라가 버렸죠", "en": "That was against the rule"},
            {"beat": 48, "ko": "친구들이 깔깔 웃고", "en": "It made the children laugh and play"},
            {"beat": 56, "ko": "양을 보았어요", "en": "To see a lamb at school"},
        ],
    },
    {
        "id": "black-sheep",
        "titleKo": "검은 양아",
        "titleEn": "Baa Baa Black Sheep",
        "emoji": "🖤",
        "category": "animal",
        "color": "#E8E0F4",
        "accent": "#A894C8",
        "bpm": 90,
        "style": "play",
        "notes": (
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "G4q G4q F4q F4q E4q E4q D4h "
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "검은 양아 검은 양아", "en": "Baa, baa, black sheep"},
            {"beat": 8, "ko": "털이 있니?", "en": "Have you any wool?"},
            {"beat": 16, "ko": "네, 있어요 세 주머니", "en": "Yes sir, yes sir, three bags full"},
            {"beat": 24, "ko": "주인 아저씨 한 주머니", "en": "One for the master"},
            {"beat": 32, "ko": "안주인께 한 주머니", "en": "And one for the dame"},
            {"beat": 40, "ko": "골목 아기에게도", "en": "And one for the little boy"},
            {"beat": 48, "ko": "따뜻한 털이에요", "en": "Who lives down the lane"},
        ],
    },
    {
        "id": "row-boat",
        "titleKo": "노 저어라",
        "titleEn": "Row, Row, Row Your Boat",
        "emoji": "🚣",
        "category": "play",
        "color": "#C9E8F7",
        "accent": "#6BB3D8",
        "bpm": 96,
        "style": "play",
        "perc": True,
        "notes": (
            "C4q. C4e C4q. D4e E4h "
            "E4q. D4e E4q. F4e G4h "
            "C5e C5e C5e G4e G4e G4e E4e E4e E4e C4e C4e C4e "
            "G4q. F4e E4q. D4e C4h "
            "C4q. C4e C4q. D4e E4h "
            "E4q. D4e E4q. F4e G4h "
            "C5e C5e C5e G4e G4e G4e E4e E4e E4e C4e C4e C4e "
            "G4q. F4e E4q. D4e C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "노를 저어라 노를 저어라", "en": "Row, row, row your boat"},
            {"beat": 8, "ko": "강을 따라가요", "en": "Gently down the stream"},
            {"beat": 16, "ko": "즐겁게 즐겁게 즐겁게 즐겁게", "en": "Merrily, merrily, merrily, merrily"},
            {"beat": 22, "ko": "인생은 꿈같아요", "en": "Life is but a dream"},
            {"beat": 30, "ko": "노를 저어라 노를 저어라", "en": "Row, row, row your boat"},
            {"beat": 38, "ko": "강을 따라가요", "en": "Gently down the stream"},
            {"beat": 46, "ko": "즐겁게 즐겁게 즐겁게 즐겁게", "en": "Merrily, merrily, merrily, merrily"},
            {"beat": 52, "ko": "인생은 꿈같아요", "en": "Life is but a dream"},
        ],
    },
    {
        "id": "london-bridge",
        "titleKo": "런던 다리",
        "titleEn": "London Bridge",
        "emoji": "🌉",
        "category": "play",
        "color": "#F8D5C8",
        "accent": "#E89A7A",
        "bpm": 98,
        "style": "play",
        "perc": True,
        "notes": (
            "G4q A4q G4q F4q E4q F4q G4h "
            "D4q E4q F4h E4q F4q G4h "
            "G4q A4q G4q F4q E4q F4q G4h "
            "D4h G4h E4q D4q C4h "
            "G4q A4q G4q F4q E4q F4q G4h "
            "D4q E4q F4h E4q F4q G4h "
            "G4q A4q G4q F4q E4q F4q G4h "
            "D4h G4h E4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "런던 다리가 무너져요", "en": "London Bridge is falling down"},
            {"beat": 8, "ko": "무너져요 무너져요", "en": "Falling down, falling down"},
            {"beat": 16, "ko": "런던 다리가 무너져요", "en": "London Bridge is falling down"},
            {"beat": 24, "ko": "나의 아름다운 아가씨", "en": "My fair lady"},
            {"beat": 32, "ko": "나무와 점토로 지어요", "en": "Build it up with wood and clay"},
            {"beat": 40, "ko": "지어요 지어요", "en": "Wood and clay, wood and clay"},
            {"beat": 48, "ko": "나무와 점토로 지어요", "en": "Build it up with wood and clay"},
            {"beat": 56, "ko": "나의 아름다운 아가씨", "en": "My fair lady"},
        ],
    },
    {
        "id": "old-macdonald",
        "titleKo": "올드 맥도널드",
        "titleEn": "Old MacDonald Had a Farm",
        "emoji": "🐔",
        "category": "animal",
        "color": "#F7E0B8",
        "accent": "#E0B45A",
        "bpm": 108,
        "style": "play",
        "perc": True,
        "notes": (
            "G4q G4q G4q D4q E4q E4q D4h "
            "B4q B4q A4q A4q G4h Restq "
            "D4e D4e G4q D4e D4e G4q "
            "G4e G4e G4e G4e G4e G4e G4q "
            "G4q G4q G4q D4q E4q E4q D4h "
            "B4q B4q A4q A4q G4h Restq "
            "G4q G4q G4q D4q E4q E4q D4h "
            "B4q B4q A4q A4q G4h"
        ),
        "lines": [
            {"beat": 0, "ko": "올드 맥도널드 농장에", "en": "Old MacDonald had a farm"},
            {"beat": 8, "ko": "이아이오", "en": "E-I-E-I-O"},
            {"beat": 14, "ko": "그 농장에 병아리가", "en": "And on that farm he had a chick"},
            {"beat": 22, "ko": "짹짹짹짹 이아이오", "en": "E-I-E-I-O"},
            {"beat": 30, "ko": "짹짹 여기 짹짹 저기", "en": "With a chick-chick here and there"},
            {"beat": 38, "ko": "여기저기 짹짹짹", "en": "Here a chick, there a chick"},
            {"beat": 46, "ko": "올드 맥도널드 농장에", "en": "Old MacDonald had a farm"},
            {"beat": 54, "ko": "이아이오", "en": "E-I-E-I-O"},
        ],
    },
    {
        "id": "bingo",
        "titleKo": "빙고",
        "titleEn": "Bingo",
        "emoji": "🐶",
        "category": "animal",
        "color": "#F8D4C0",
        "accent": "#E8A078",
        "bpm": 104,
        "style": "play",
        "perc": True,
        "notes": (
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "G4q G4q G4q G4q G4h "
            "G4q G4q G4q G4q G4h "
            "G4q G4q G4q G4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "농부에게 강아지가", "en": "There was a farmer had a dog"},
            {"beat": 8, "ko": "이름은 바로 빙고", "en": "And Bingo was his name-o"},
            {"beat": 16, "ko": "비 아이 엔 지 오", "en": "B-I-N-G-O"},
            {"beat": 22, "ko": "비 아이 엔 지 오", "en": "B-I-N-G-O"},
            {"beat": 28, "ko": "비 아이 엔 지 오", "en": "B-I-N-G-O"},
            {"beat": 34, "ko": "이름은 바로 빙고", "en": "And Bingo was his name-o"},
            {"beat": 42, "ko": "농부에게 강아지가", "en": "There was a farmer had a dog"},
            {"beat": 50, "ko": "이름은 바로 빙고", "en": "And Bingo was his name-o"},
        ],
    },
    {
        "id": "itsy-spider",
        "titleKo": "아기 거미",
        "titleEn": "The Itsy Bitsy Spider",
        "emoji": "🕷️",
        "category": "animal",
        "color": "#E0D8F8",
        "accent": "#A090D0",
        "bpm": 94,
        "style": "play",
        "notes": (
            "G4e C5e C5e C5e D5e E5q E5q "
            "E5e D5e C5e D5e E5q C5h "
            "G5q E5q C5q G4h "
            "G4e C5e C5e C5e D5e E5q E5q "
            "E5e D5e C5e D5e E5q C5h "
            "G4e C5e C5e C5e D5e E5q E5q "
            "E5e D5e C5e D5e E5q C5h "
            "G5q E5q C5q G4h "
            "G4e C5e C5e C5e D5e E5q E5q "
            "E5e D5e C5e D5e E5q C5h"
        ),
        "lines": [
            {"beat": 0, "ko": "아기 거미가 줄 타고 올라가요", "en": "The itsy bitsy spider climbed up the spout"},
            {"beat": 8, "ko": "비가 와서 거미를 씻겨 내렸죠", "en": "Down came the rain and washed the spider out"},
            {"beat": 16, "ko": "해가 나와 빗물을 말리고", "en": "Out came the sun and dried up all the rain"},
            {"beat": 22, "ko": "아기 거미 다시 올라가요", "en": "And the itsy bitsy spider climbed up again"},
            {"beat": 30, "ko": "아기 거미가 줄 타고 올라가요", "en": "The itsy bitsy spider climbed up the spout"},
            {"beat": 38, "ko": "비가 와서 거미를 씻겨 내렸죠", "en": "Down came the rain and washed the spider out"},
            {"beat": 46, "ko": "해가 나와 빗물을 말리고", "en": "Out came the sun and dried up all the rain"},
            {"beat": 52, "ko": "아기 거미 다시 올라가요", "en": "And the itsy bitsy spider climbed up again"},
        ],
    },
    {
        "id": "bird",
        "titleKo": "새야새야",
        "titleEn": "Bird, Bird",
        "emoji": "🐦",
        "korean": True,
        "category": "animal",
        "color": "#C8F0FF",
        "accent": "#5EC4E8",
        "bpm": 116,
        "style": "play",
        "perc": True,
        "notes": (
            "G4q A4q G4q E4q G4q A4q G4h "
            "C5q A4q G4q E4q D4q E4q C4h "
            "G4q A4q G4q E4q G4q A4q G4h "
            "C5q A4q G4q E4q D4q E4q C4h "
            "G4q A4q G4q E4q G4q A4q G4h "
            "C5q A4q G4q E4q D4q E4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "새야새야 파랑새야", "en": "Bird, bird, little blue bird"},
            {"beat": 8, "ko": "녹두밭에 앉지 마라", "en": "Don't sit in the bean field"},
            {"beat": 16, "ko": "새야새야 노랑새야", "en": "Bird, bird, little yellow bird"},
            {"beat": 24, "ko": "노래하며 날아가자", "en": "Sing and fly away with me"},
            {"beat": 32, "ko": "짹짹짹짹 노래해요", "en": "Tweet tweet, sing a song"},
            {"beat": 40, "ko": "하늘 높이 날아가요", "en": "Fly up high into the sky"},
        ],
    },
    {
        "id": "little-dog",
        "titleKo": "강아지는 어디에",
        "titleEn": "Oh Where Has My Little Dog Gone",
        "emoji": "🐾",
        "category": "animal",
        "color": "#F7E6C8",
        "accent": "#D4B07A",
        "bpm": 92,
        "style": "play",
        "notes": (
            "G4q E4q C4q G4h E4q C4h "
            "A4q F4q D4q A4h F4q D4h "
            "G4q G4q A4q G4q E4q C4h "
            "D4q E4q D4q B3q C4w "
            "G4q E4q C4q G4h E4q C4h "
            "A4q F4q D4q A4h F4q D4h "
            "G4q G4q A4q G4q E4q C4h "
            "D4q E4q D4q B3q C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "강아지는 어디에 갔을까", "en": "Oh where, oh where has my little dog gone"},
            {"beat": 8, "ko": "어디에 있을까", "en": "Oh where, oh where can he be"},
            {"beat": 16, "ko": "짧은 귀와 긴 꼬리", "en": "With his ears cut short and his tail cut long"},
            {"beat": 24, "ko": "어디에 있을까", "en": "Oh where, oh where can he be"},
            {"beat": 32, "ko": "강아지는 어디에 갔을까", "en": "Oh where, oh where has my little dog gone"},
            {"beat": 40, "ko": "어디에 있을까", "en": "Oh where, oh where can he be"},
            {"beat": 48, "ko": "짧은 귀와 긴 꼬리", "en": "With his ears cut short and his tail cut long"},
            {"beat": 56, "ko": "어디에 있을까", "en": "Oh where, oh where can he be"},
        ],
    },
    {
        "id": "circle",
        "titleKo": "둥글게 둥글게",
        "titleEn": "Round and Round",
        "emoji": "⭕",
        "korean": True,
        "category": "play",
        "color": "#FFE0F0",
        "accent": "#F08AB8",
        "bpm": 120,
        "style": "play",
        "perc": True,
        "notes": (
            "C5q C5q A4q A4q G4q E4q G4h "
            "C5q C5q A4q A4q G4q E4q C4h "
            "G4q G4q E4q E4q G4q A4q G4h "
            "C5q A4q G4q E4q D4q E4q C4h "
            "C5q C5q A4q A4q G4q E4q G4h "
            "C5q C5q A4q A4q G4q E4q C4h "
            "G4q G4q E4q E4q G4q A4q G4h "
            "C5q A4q G4q E4q D4q E4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "둥글게 둥글게", "en": "Round and round we go"},
            {"beat": 8, "ko": "둥글게 둥글게", "en": "Round and round we go"},
            {"beat": 16, "ko": "빙글빙글 돌아가며", "en": "Spinning, spinning, dancing"},
            {"beat": 24, "ko": "춤을 춥시다", "en": "Let's dance together"},
            {"beat": 32, "ko": "손뼉을 치면서", "en": "Clap your hands with me"},
            {"beat": 40, "ko": "발도 구르면서", "en": "Stamp your feet with me"},
            {"beat": 48, "ko": "다 같이 웃으며", "en": "Everybody smiling"},
            {"beat": 56, "ko": "돌아가요", "en": "Round we go"},
        ],
    },
    {
        "id": "jack-jill",
        "titleKo": "잭과 질",
        "titleEn": "Jack and Jill",
        "emoji": "🪣",
        "category": "story",
        "color": "#D4EEF8",
        "accent": "#7AB8D4",
        "bpm": 96,
        "style": "play",
        "notes": (
            "C4q C4q C4q D4q E4h E4h "
            "D4q E4q F4q D4q E4h C4h "
            "G4q G4q F4q E4q D4h G3h "
            "C4q E4q D4q B3q C4w "
            "C4q C4q C4q D4q E4h E4h "
            "D4q E4q F4q D4q E4h C4h "
            "G4q G4q F4q E4q D4h G3h "
            "C4q E4q D4q B3q C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "잭과 질이 언덕을 올라", "en": "Jack and Jill went up the hill"},
            {"beat": 8, "ko": "물을 길으러 갔어요", "en": "To fetch a pail of water"},
            {"beat": 16, "ko": "잭이 넘어져 머리가 아프고", "en": "Jack fell down and broke his crown"},
            {"beat": 24, "ko": "질도 따라 굴렀죠", "en": "And Jill came tumbling after"},
            {"beat": 32, "ko": "잭과 질이 언덕을 올라", "en": "Jack and Jill went up the hill"},
            {"beat": 40, "ko": "물을 길으러 갔어요", "en": "To fetch a pail of water"},
            {"beat": 48, "ko": "잭이 넘어져 머리가 아프고", "en": "Jack fell down and broke his crown"},
            {"beat": 56, "ko": "질도 따라 굴렀죠", "en": "And Jill came tumbling after"},
        ],
    },
    {
        "id": "hickory",
        "titleKo": "째깍째깍 시계",
        "titleEn": "Hickory Dickory Dock",
        "emoji": "⏰",
        "category": "story",
        "color": "#E4F0D4",
        "accent": "#A3C47A",
        "bpm": 100,
        "style": "play",
        "perc": True,
        "notes": (
            "C4q C4e D4e C4q E4h "
            "C4q C4e D4e C4q F4h "
            "G4q E4q C4q A4q "
            "G4e F4e E4e D4e C4h "
            "C4q C4e D4e C4q E4h "
            "C4q C4e D4e C4q F4h "
            "G4q E4q C4q A4q "
            "G4e F4e E4e D4e C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "째깍째깍 시계", "en": "Hickory dickory dock"},
            {"beat": 6, "ko": "생쥐가 시계를 타고", "en": "The mouse ran up the clock"},
            {"beat": 12, "ko": "시계가 한 시를 치자", "en": "The clock struck one"},
            {"beat": 16, "ko": "생쥐가 내려왔죠", "en": "The mouse ran down"},
            {"beat": 20, "ko": "째깍째깍 시계", "en": "Hickory dickory dock"},
            {"beat": 26, "ko": "생쥐가 시계를 타고", "en": "The mouse ran up the clock"},
            {"beat": 32, "ko": "시계가 한 시를 치자", "en": "The clock struck one"},
            {"beat": 36, "ko": "생쥐가 내려왔죠", "en": "The mouse ran down"},
        ],
    },
    {
        "id": "clap",
        "titleKo": "손뼉짝짝짝",
        "titleEn": "Clap Clap Clap",
        "emoji": "👏",
        "korean": True,
        "category": "play",
        "color": "#FFE6C8",
        "accent": "#F0A05A",
        "bpm": 122,
        "style": "play",
        "perc": True,
        "notes": (
            "C5q C5q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "G4q G4q F4q F4q E4q E4q D4h "
            "C5q C5q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "손뼉 짝짝짝", "en": "Clap clap clap"},
            {"beat": 8, "ko": "발도 쿵쿵쿵", "en": "Stomp stomp stomp"},
            {"beat": 16, "ko": "고개를 끄덕끄덕", "en": "Nod your little head"},
            {"beat": 24, "ko": "손뼉 짝짝짝", "en": "Clap clap clap"},
            {"beat": 32, "ko": "웃으며 따라해요", "en": "Smile and follow me"},
        ],
    },
    {
        "id": "sun",
        "titleKo": "햇님 방긋",
        "titleEn": "Smiling Sun",
        "emoji": "☀️",
        "korean": True,
        "category": "play",
        "color": "#FFF4A8",
        "accent": "#F0C43A",
        "bpm": 118,
        "style": "play",
        "perc": True,
        "notes": (
            "C5q D5q E5q C5q G4q A4q G4h "
            "C5q D5q E5q C5q G4q E4q C4h "
            "G4q G4q A4q G4q E4q D4q C4h "
            "C5q D5q E5q C5q G4q A4q G4h "
            "C5q D5q E5q C5q G4q E4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "햇님 햇님 방긋방긋", "en": "Sunshine, sunshine, smile at me"},
            {"beat": 8, "ko": "따뜻하게 비춰줘요", "en": "Warm and bright today"},
            {"beat": 16, "ko": "아기야 일어나요", "en": "Wake up little baby"},
            {"beat": 24, "ko": "햇님 햇님 방긋방긋", "en": "Sunshine, sunshine, smile at me"},
            {"beat": 32, "ko": "밖으로 놀러가요", "en": "Let's go out and play"},
        ],
    },
    {
        "id": "mulberry",
        "titleKo": "뽕나무 덤불",
        "titleEn": "Here We Go Round the Mulberry Bush",
        "emoji": "🌿",
        "category": "play",
        "color": "#D4F0DC",
        "accent": "#78C494",
        "bpm": 102,
        "style": "play",
        "perc": True,
        "notes": (
            "C5q C5q C5e C5e G4q A4q A4q G4h "
            "E4q G4q F4q D4q C4h Restq "
            "C5q C5q C5e C5e G4q A4q A4q G4h "
            "E4q G4q F4q D4q C4w "
            "C5q C5q C5e C5e G4q A4q A4q G4h "
            "E4q G4q F4q D4q C4h Restq "
            "C5q C5q C5e C5e G4q A4q A4q G4h "
            "E4q G4q F4q D4q C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "뽕나무 덤불을 돌아요", "en": "Here we go round the mulberry bush"},
            {"beat": 8, "ko": "덤불을 돌아요", "en": "The mulberry bush"},
            {"beat": 16, "ko": "뽕나무 덤불을 돌아요", "en": "Here we go round the mulberry bush"},
            {"beat": 24, "ko": "추운 아침이에요", "en": "On a cold and frosty morning"},
            {"beat": 32, "ko": "손을 이렇게 씻어요", "en": "This is the way we wash our hands"},
            {"beat": 40, "ko": "손을 씻어요", "en": "Wash our hands"},
            {"beat": 48, "ko": "손을 이렇게 씻어요", "en": "This is the way we wash our hands"},
            {"beat": 56, "ko": "추운 아침이에요", "en": "On a cold and frosty morning"},
        ],
    },
    {
        "id": "hello",
        "titleKo": "안녕안녕",
        "titleEn": "Hello Hello",
        "emoji": "👋",
        "korean": True,
        "category": "play",
        "color": "#FFE4C8",
        "accent": "#F0A05A",
        "bpm": 124,
        "style": "play",
        "perc": True,
        "notes": (
            "C5q C5q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "G4q G4q F4q F4q E4q E4q D4h "
            "C5q C5q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "안녕 안녕 안녕", "en": "Hello, hello, hello"},
            {"beat": 8, "ko": "친구야 안녕", "en": "Hello my friend"},
            {"beat": 16, "ko": "손을 흔들며 안녕", "en": "Wave your hand and say hello"},
            {"beat": 24, "ko": "안녕 안녕 안녕", "en": "Hello, hello, hello"},
            {"beat": 32, "ko": "우리 모두 안녕", "en": "Hello everyone"},
        ],
    },
    {
        "id": "jump",
        "titleKo": "깡충깡충",
        "titleEn": "Hop Hop Hop",
        "emoji": "🐰",
        "korean": True,
        "category": "play",
        "color": "#E4F4C8",
        "accent": "#88C46A",
        "bpm": 128,
        "style": "play",
        "perc": True,
        "notes": (
            "E5q E5q C5q C5q D5q D5q G4h "
            "E5q E5q C5q C5q D5q B4q C5h "
            "G4q A4q G4q E4q G4q A4q G4h "
            "E5q E5q C5q C5q D5q B4q C5h "
            "E5q E5q C5q C5q D5q D5q G4h "
            "E5q E5q C5q C5q D5q B4q C5h"
        ),
        "lines": [
            {"beat": 0, "ko": "깡충깡충 토끼가", "en": "Hop hop hop goes bunny"},
            {"beat": 8, "ko": "깡충깡충 뛰어요", "en": "Hop hop hop so high"},
            {"beat": 16, "ko": "앞발로 깡충", "en": "Front paws hop hop"},
            {"beat": 24, "ko": "뒷발로 깡충", "en": "Back paws hop hop"},
            {"beat": 32, "ko": "깡충깡충 뛰어요", "en": "Hop hop hop so high"},
            {"beat": 40, "ko": "아기랑 같이 뛰어요", "en": "Baby hop along with me"},
        ],
    },
    {
        "id": "muffin-man",
        "titleKo": "머핀 아저씨",
        "titleEn": "The Muffin Man",
        "emoji": "🧁",
        "category": "daily",
        "color": "#F8D8E4",
        "accent": "#E090B0",
        "bpm": 98,
        "style": "play",
        "notes": (
            "C4q D4q E4q F4q G4h E4h "
            "G4e A4e G4e F4e E4q C4q D4h G3h "
            "C4q D4q E4q F4q G4h E4h "
            "G4q G4q D4q E4q C4w "
            "C4q D4q E4q F4q G4h E4h "
            "G4e A4e G4e F4e E4q C4q D4h G3h "
            "C4q D4q E4q F4q G4h E4h "
            "G4q G4q D4q E4q C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "머핀 아저씨 아시나요", "en": "Do you know the muffin man"},
            {"beat": 8, "ko": "드루리 레인에 사는", "en": "The muffin man, the muffin man"},
            {"beat": 16, "ko": "머핀 아저씨 아시나요", "en": "Do you know the muffin man"},
            {"beat": 24, "ko": "드루리 레인에 살죠", "en": "Who lives on Drury Lane"},
            {"beat": 32, "ko": "네 알아요 머핀 아저씨", "en": "Yes I know the muffin man"},
            {"beat": 40, "ko": "드루리 레인에 사는", "en": "The muffin man, the muffin man"},
            {"beat": 48, "ko": "네 알아요 머핀 아저씨", "en": "Yes I know the muffin man"},
            {"beat": 56, "ko": "드루리 레인에 살죠", "en": "Who lives on Drury Lane"},
        ],
    },
    {
        "id": "pat-a-cake",
        "titleKo": "케이크를 만들자",
        "titleEn": "Pat-a-cake",
        "emoji": "🎂",
        "category": "daily",
        "color": "#FBE0D0",
        "accent": "#E8A888",
        "bpm": 100,
        "style": "play",
        "perc": True,
        "notes": (
            "C4q C4q E4q G4q G4q E4q C4h "
            "D4q D4q B3q G3q C4h Restq "
            "C4q C4q E4q G4q A4q G4q E4q C4q "
            "D4q F4q E4q D4q C4w "
            "C4q C4q E4q G4q G4q E4q C4h "
            "D4q D4q B3q G3q C4h Restq "
            "C4q C4q E4q G4q A4q G4q E4q C4q "
            "D4q F4q E4q D4q C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "톡톡 케이크 빵집 아저씨", "en": "Pat-a-cake, pat-a-cake, baker's man"},
            {"beat": 8, "ko": "빨리빨리 만들어 주세요", "en": "Bake me a cake as fast as you can"},
            {"beat": 16, "ko": "콕 찌르고 표시하고", "en": "Pat it and prick it and mark it with B"},
            {"beat": 24, "ko": "아기와 내가 나눠 먹어요", "en": "And put it in the oven for baby and me"},
            {"beat": 32, "ko": "톡톡 케이크 빵집 아저씨", "en": "Pat-a-cake, pat-a-cake, baker's man"},
            {"beat": 40, "ko": "빨리빨리 만들어 주세요", "en": "Bake me a cake as fast as you can"},
            {"beat": 48, "ko": "콕 찌르고 표시하고", "en": "Pat it and prick it and mark it with B"},
            {"beat": 56, "ko": "아기와 내가 나눠 먹어요", "en": "And put it in the oven for baby and me"},
        ],
    },
    {
        "id": "polly-kettle",
        "titleKo": "폴리는 주전자를",
        "titleEn": "Polly Put the Kettle On",
        "emoji": "🫖",
        "category": "daily",
        "color": "#D8ECF8",
        "accent": "#84B8D4",
        "bpm": 104,
        "style": "play",
        "notes": (
            "G4q E4q C4q G4q E4q C4h Restq "
            "G4q E4q C4q D4q B3q C4h Restq "
            "G4q E4q C4q G4q E4q C4h Restq "
            "D4q D4q B3q G3q C4w "
            "G4q E4q C4q G4q E4q C4h Restq "
            "G4q E4q C4q D4q B3q C4h Restq "
            "G4q E4q C4q G4q E4q C4h Restq "
            "D4q D4q B3q G3q C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "폴리야 주전자를 올리렴", "en": "Polly put the kettle on"},
            {"beat": 8, "ko": "폴리야 주전자를 올리렴", "en": "Polly put the kettle on"},
            {"beat": 16, "ko": "폴리야 주전자를 올리렴", "en": "Polly put the kettle on"},
            {"beat": 24, "ko": "다 같이 차를 마셔요", "en": "We'll all have tea"},
            {"beat": 32, "ko": "수키야 주전자를 내리렴", "en": "Sukey take it off again"},
            {"beat": 40, "ko": "수키야 주전자를 내리렴", "en": "Sukey take it off again"},
            {"beat": 48, "ko": "수키야 주전자를 내리렴", "en": "Sukey take it off again"},
            {"beat": 56, "ko": "모두 돌아가 버렸죠", "en": "They've all gone away"},
        ],
    },
    {
        "id": "alphabet",
        "titleKo": "알파벳 노래",
        "titleEn": "Alphabet Song",
        "emoji": "🔤",
        "category": "daily",
        "color": "#DCE8FB",
        "accent": "#7EA4E0",
        "bpm": 88,
        "style": "play",
        "notes": (
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h "
            "G4q G4q F4q F4q E4q E4q D4h "
            "G4q G4q F4q F4q E4q E4q D4h "
            "C4q C4q G4q G4q A4q A4q G4h "
            "F4q F4q E4q E4q D4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "에이 비 시 디 이 에프 지", "en": "A B C D E F G"},
            {"beat": 8, "ko": "에이치 아이 제이 케이 엘 엠 엔 오 피", "en": "H I J K L M N O P"},
            {"beat": 16, "ko": "큐 알 에스 티 유 브이", "en": "Q R S T U V"},
            {"beat": 24, "ko": "더블유 엑스 와이 앤드 지", "en": "W X Y and Z"},
            {"beat": 32, "ko": "이제 알파벳을 알아요", "en": "Now I know my ABCs"},
            {"beat": 40, "ko": "다음에도 함께 불러요", "en": "Next time won't you sing with me"},
        ],
    },
    {
        "id": "rain-away",
        "titleKo": "비야 비야",
        "titleEn": "Rain, Rain, Go Away",
        "emoji": "🌧️",
        "category": "play",
        "color": "#D0E4F8",
        "accent": "#7AA8D8",
        "bpm": 90,
        "style": "lullaby",
        "notes": (
            "G4q E4q G4h G4q E4q G4h "
            "A4q G4q F4q E4q D4q E4q C4h "
            "G4q E4q G4h G4q E4q G4h "
            "A4q G4q F4q E4q D4q D4q C4h "
            "G4q E4q G4h G4q E4q G4h "
            "A4q G4q F4q E4q D4q E4q C4h "
            "G4q E4q G4h G4q E4q G4h "
            "A4q G4q F4q E4q D4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "비야 비야 오지 마라", "en": "Rain, rain, go away"},
            {"beat": 8, "ko": "다른 날에 다시 오렴", "en": "Come again another day"},
            {"beat": 16, "ko": "아기랑 놀고 싶은데", "en": "Little baby wants to play"},
            {"beat": 24, "ko": "비야 비야 가지 마라", "en": "Rain, rain, go away"},
            {"beat": 32, "ko": "비야 비야 오지 마라", "en": "Rain, rain, go away"},
            {"beat": 40, "ko": "다른 날에 다시 오렴", "en": "Come again another day"},
            {"beat": 48, "ko": "아기랑 놀고 싶은데", "en": "Little baby wants to play"},
            {"beat": 56, "ko": "비야 비야 가지 마라", "en": "Rain, rain, go away"},
        ],
    },
    {
        "id": "skip-lou",
        "titleKo": "루에게로",
        "titleEn": "Skip to My Lou",
        "emoji": "💃",
        "category": "play",
        "color": "#F8D0E0",
        "accent": "#E888B0",
        "bpm": 112,
        "style": "play",
        "perc": True,
        "notes": (
            "C5q C5q A4q A4q C5q C5q A4h "
            "B4q B4q G4q G4q B4q B4q G4h "
            "C5q C5q A4q A4q C5q C5q A4h "
            "G4q A4q B4q G4q C5w "
            "C5q C5q A4q A4q C5q C5q A4h "
            "B4q B4q G4q G4q B4q B4q G4h "
            "C5q C5q A4q A4q C5q C5q A4h "
            "G4q A4q B4q G4q C5w"
        ),
        "lines": [
            {"beat": 0, "ko": "루에게로 건너뛰자", "en": "Skip, skip, skip to my Lou"},
            {"beat": 8, "ko": "루에게로 건너뛰자", "en": "Skip, skip, skip to my Lou"},
            {"beat": 16, "ko": "루에게로 건너뛰자", "en": "Skip, skip, skip to my Lou"},
            {"beat": 24, "ko": "나의 사랑하는 루", "en": "Skip to my Lou, my darling"},
            {"beat": 32, "ko": "파리가 사탕에 앉았네", "en": "Flies in the buttermilk, shoo fly shoo"},
            {"beat": 40, "ko": "파리가 사탕에 앉았네", "en": "Flies in the buttermilk, shoo fly shoo"},
            {"beat": 48, "ko": "파리가 사탕에 앉았네", "en": "Flies in the buttermilk, shoo fly shoo"},
            {"beat": 56, "ko": "나의 사랑하는 루", "en": "Skip to my Lou, my darling"},
        ],
    },
    {
        "id": "hide",
        "titleKo": "꼭꼭 숨어라",
        "titleEn": "Hide and Seek",
        "emoji": "🙈",
        "korean": True,
        "category": "play",
        "color": "#F6D0D4",
        "accent": "#E08090",
        "bpm": 120,
        "style": "play",
        "perc": True,
        "notes": (
            "C5q C5q A4q A4q G4q E4q G4h "
            "C5q C5q A4q A4q G4q E4q C4h "
            "G4q A4q G4q E4q G4q A4q G4h "
            "C5q A4q G4q E4q D4q E4q C4h "
            "C5q C5q A4q A4q G4q E4q G4h "
            "C5q C5q A4q A4q G4q E4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "꼭꼭 숨어라", "en": "Hide, hide, hide away"},
            {"beat": 8, "ko": "머리카락 보인다", "en": "I can see your hair"},
            {"beat": 16, "ko": "하나 둘 셋", "en": "One, two, three"},
            {"beat": 24, "ko": "내가 찾았다", "en": "I found you"},
            {"beat": 32, "ko": "꼭꼭 숨어라", "en": "Hide, hide, hide away"},
            {"beat": 40, "ko": "다시 숨어요", "en": "Hide again with me"},
        ],
    },
]


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog = []
    for i, song in enumerate(SONGS, 1):
        wav, lyrics, duration = render_song(song)
        dest = AUDIO_DIR / f"{song['id']}.mp3"
        wav_to_mp3(wav, dest)
        catalog.append(
            {
                "id": song["id"],
                "no": i,
                "titleKo": song["titleKo"],
                "titleEn": song["titleEn"],
                "emoji": song["emoji"],
                "category": song["category"],
                "korean": bool(song.get("korean")),
                "color": song["color"],
                "accent": song["accent"],
                "bpm": max(int(round(song["bpm"] * 1.22)), 112),
                "audio": f"/audio/{song['id']}.mp3",
                "duration": round(duration, 2),
                "lyrics": lyrics,
                "source": "전래·퍼블릭 도메인 동요 · 밝은 실로폰 편곡",
            }
        )
        print(f"{i:02d} {song['id']:16s} {duration:6.1f}s  {song['titleKo']}")
    (DATA_DIR / "songs.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", len(catalog), "songs")


if __name__ == "__main__":
    main()
