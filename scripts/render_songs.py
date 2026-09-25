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


def hz(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def felt_piano(freq: float, n: int, velocity: float = 1.0) -> np.ndarray:
    """Round felt-piano partials. No hammer click."""
    t = np.arange(n) / SR
    sig = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate((1.0, 0.34, 0.12, 0.045, 0.012), start=1):
        stretch = 1.0 + 0.00012 * k * k
        partial = freq * k * stretch
        if partial > 7600:
            break
        sig += amp * np.sin(2 * math.pi * partial * t)
    # A whisper of a second string, a few cents sharp, for width.
    sig += 0.16 * np.sin(2 * math.pi * freq * 1.003 * t)
    decay = np.exp(-t * (0.95 + freq / 1400.0))
    rel = min(0.2, max(0.045, n / SR * 0.42))
    return sig * decay * ads(n, 0.014, 0.16, 0.62, rel) * velocity


def warm_pad(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    sig = (
        np.sin(2 * math.pi * freq * t)
        + np.sin(2 * math.pi * freq * 1.004 * t)
        + 0.45 * np.sin(2 * math.pi * freq * 0.5 * t)
    )
    return sig * ads(n, 0.12, 0.22, 0.72, 0.4)


def lowpass(sig: np.ndarray, cutoff: float) -> np.ndarray:
    if len(sig) < 16:
        return sig
    k = max(3, int(SR / cutoff))
    kernel = np.exp(-np.arange(k) * 2.8 / k)
    kernel /= kernel.sum()
    return np.convolve(sig, kernel, mode="same")


METER3 = {"brahms", "fox", "arirang"}
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)
MAJOR_TRIADS = ((0, 4, 7), (5, 9, 0), (7, 11, 2), (9, 0, 4))
MINOR_TRIADS = ((0, 3, 7), (5, 8, 0), (3, 7, 10), (8, 0, 3))


def song_bpm(song: dict) -> int:
    style = song.get("style", "play")
    bpm = int(song["bpm"])
    if style == "lullaby":
        return max(66, min(bpm, 86))
    return max(88, min(bpm, 110))


def infer_key(notes: list[tuple[int | None, float]]) -> tuple[int, str]:
    pitched = [(m, d) for m, d in notes if m is not None]
    tonic = pitched[-1][0] % 12
    pcs = {m % 12 for m, _ in pitched}
    major = {(tonic + s) % 12 for s in MAJOR_STEPS}
    minor = {(tonic + s) % 12 for s in MINOR_STEPS}
    mode = "minor" if len(pcs - minor) < len(pcs - major) else "major"
    return tonic, mode


def bar_triad(group: list[tuple[int, float]], tonic: int, mode: str) -> tuple[int, int, int]:
    triads = MINOR_TRIADS if mode == "minor" else MAJOR_TRIADS
    best = triads[0]
    best_score = -1.0
    for index, triad in enumerate(triads):
        tones = {(tonic + step) % 12 for step in triad}
        score = sum(d for m, d in group if m % 12 in tones)
        # Prefer the home chord when the bar fits more than one triad.
        score += (0.35, 0.15, 0.05, 0.2)[index]
        if score > best_score:
            best, best_score = triad, score
    return best


def place(bus: np.ndarray, i0: int, sig: np.ndarray, gain: float) -> None:
    if gain == 0 or i0 >= len(bus) or len(sig) == 0:
        return
    i1 = min(len(bus), i0 + len(sig))
    n = i1 - i0
    if n > 0:
        bus[i0:i1] += sig[:n] * gain


def voice_midi(pc: int, low: int, high: int) -> int:
    midi = low + (pc - (low % 12)) % 12
    while midi < low:
        midi += 12
    while midi > high:
        midi -= 12
    return midi


def render_song(song: dict) -> tuple[np.ndarray, list[dict], float]:
    notes = parse_notes(song["notes"])
    style = song.get("style", "play")
    bpm = song_bpm(song)
    beat_sec = 60.0 / bpm
    lead = 0.32
    tail = 1.15
    total_beats = sum(d for _, d in notes)
    duration = lead + total_beats * beat_sec + tail
    n_total = int(duration * SR) + 1
    melody = np.zeros(n_total, dtype=np.float64)
    harmony = np.zeros(n_total, dtype=np.float64)
    meter = 3 if song["id"] in METER3 else 4
    tonic, mode = infer_key(notes)

    beat_pos = 0.0
    timed: list[tuple[float, int | None, float]] = []
    for midi, dur in notes:
        timed.append((beat_pos, midi, dur))
        beat_pos += dur

    for start_beat, midi, dur in timed:
        if midi is None:
            continue
        midi = min(max(midi, 48), 84)
        start = lead + start_beat * beat_sec
        # Notes overlap a little so the line sings instead of pecking.
        hold = dur * beat_sec + 0.07
        n = int(max(hold, 0.05) * SR)
        i0 = int(start * SR)
        place(melody, i0, felt_piano(hz(midi), n, 0.92), 0.7)
        if midi - 12 >= 40:
            place(melody, i0, felt_piano(hz(midi - 12), n, 0.55), 0.16)

    # One soft chord per bar, held under the melody.
    bar = 0
    while bar * meter < total_beats - 0.05:
        bar_start = bar * meter
        bar_end = min(total_beats, bar_start + meter)
        group = [
            (m, d)
            for b, m, d in timed
            if m is not None and bar_start - 0.01 <= b < bar_end - 0.01
        ]
        if group:
            triad = bar_triad(group, tonic, mode)
            i0 = int((lead + bar_start * beat_sec) * SR)
            n = int(((bar_end - bar_start) * beat_sec + 0.28) * SR)
            gains = (0.11, 0.07, 0.08) if style == "lullaby" else (0.13, 0.075, 0.09)
            for step, gain in zip(triad, gains):
                pc = (tonic + step) % 12
                rootish = step == triad[0]
                voiced = voice_midi(pc, 46 if rootish else 58, 64 if rootish else 74)
                place(harmony, i0, warm_pad(hz(voiced), n), gain)
        bar += 1

    # Gentle left hand on the first beat of the bar. Lullabies stay pad-only.
    if style != "lullaby":
        bass_pc = tonic
        bass_midi = voice_midi(bass_pc, 43, 52)
        beat = 0.0
        while beat < total_beats - 0.05:
            i0 = int((lead + beat * beat_sec) * SR)
            n = int(min(meter * 0.55, 1.6) * beat_sec * SR)
            place(harmony, i0, felt_piano(hz(bass_midi), n, 0.7), 0.2)
            beat += meter

    melody = lowpass(melody, 3400 if style == "lullaby" else 4200)
    harmony = lowpass(harmony, 1600)
    send = lowpass(melody * 0.42 + harmony, 2400)
    room = np.zeros(n_total, dtype=np.float64)
    for delay, wet in ((0.031, 0.22), (0.047, 0.16), (0.061, 0.11), (0.083, 0.07), (0.109, 0.045)):
        shift = int(delay * SR)
        if shift < n_total:
            room[shift:] += send[:-shift] * wet
    room = lowpass(room, 2800)

    left = melody + harmony + room
    right = melody + harmony + room
    harm_shift = int(0.013 * SR)
    if harm_shift < n_total:
        right[harm_shift:] += harmony[:-harm_shift] * 0.35
        left[harm_shift // 2 :] += harmony[: n_total - harm_shift // 2] * 0.22
    stereo = np.stack((left, right), axis=1)
    stereo = np.tanh(stereo * 1.15)
    peak = float(np.max(np.abs(stereo)) or 1.0)
    stereo = stereo / peak * 0.8
    fade = int(0.06 * SR)
    stereo[:fade] *= np.linspace(0, 1, fade)[:, None]
    stereo[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mix = stereo

    lines = song["lines"]
    lyrics = []
    for i, line in enumerate(lines):
        start_beat = float(line["beat"])
        if start_beat >= total_beats:
            raise ValueError(f"{song['id']} lyric beat {start_beat} past {total_beats} beats")
        end_beat = float(lines[i + 1]["beat"]) if i + 1 < len(lines) else total_beats
        start = lead + start_beat * beat_sec
        end = lead + end_beat * beat_sec
        lyrics.append(
            {
                "start": round(start, 3),
                "end": round(min(max(end, start + 0.4), duration - 0.12), 3),
                "ko": line["ko"].strip(),
                "en": line["en"].strip(),
            }
        )
    return mix, lyrics, duration


def wav_to_mp3(wav: np.ndarray, dest: Path) -> None:
    raw = np.clip(wav, -1, 1)
    if raw.ndim == 1:
        raw = np.stack((raw, raw), axis=1)
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
            "2",
            "-ac",
            "2",
            str(dest),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp.unlink()


def _wav_header(data_bytes: int) -> bytes:
    import struct

    channels = 2
    bits = 16
    block = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        SR,
        SR * block,
        block,
        bits,
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
        "category": "lullaby",
        "color": "#F7D6E8",
        "accent": "#E891B5",
        "bpm": 80,
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
        # 독일 민요. 계이름: 솔미미 파레레 도레미파 솔솔솔 … (음절과 1:1)
        "notes": (
            "G4q E4q E4q F4q D4q D4q C4q D4q E4q F4q G4q G4q G4q "
            "G4q E4q E4q E4q F4q D4q D4q C4q E4q G4q G4q E4q E4q E4q "
            "D4q D4q D4q D4q D4q E4q F4q E4q E4q E4q E4q E4q F4q G4q "
            "G4q E4q E4q F4q D4q D4q C4q E4q G4q G4q C4q C4q C4q "
            "G4q E4q E4q F4q D4q D4q C4q D4q E4q F4q G4q G4q G4q "
            "G4q E4q E4q E4q F4q D4q D4q C4q E4q G4q G4q E4q E4q E4q "
            "D4q D4q D4q D4q D4q E4q F4q E4q E4q E4q E4q E4q F4q G4q "
            "G4q E4q E4q F4q D4q D4q C4q E4q G4q G4q C4q C4q C4q"
        ),
        "lines": [
            {"beat": 0, "ko": "나비야 나비야 이리 날아오너라", "en": "Butterfly, butterfly, come flying over"},
            {"beat": 13, "ko": "노랑나비 흰나비 춤을 추며 오너라", "en": "Yellow butterfly, white butterfly, come dancing"},
            {"beat": 27, "ko": "봄바람에 꽃잎도 방긋방긋 웃으며", "en": "Petals in the spring breeze, smiling"},
            {"beat": 41, "ko": "참새도 짹짹짹 노래하며 춤춘다", "en": "Sparrows tweet and dance along"},
            {"beat": 54, "ko": "나비야 나비야 이리 날아오너라", "en": "Butterfly, butterfly, come flying over"},
            {"beat": 67, "ko": "노랑나비 흰나비 춤을 추며 오너라", "en": "Yellow butterfly, white butterfly, come dancing"},
            {"beat": 81, "ko": "봄바람에 꽃잎도 방긋방긋 웃으며", "en": "Petals in the spring breeze, smiling"},
            {"beat": 95, "ko": "참새도 짹짹짹 노래하며 춤춘다", "en": "Sparrows tweet and dance along"},
        ],
    },
    {
        "id": "arirang",
        "titleKo": "아리랑",
        "titleEn": "Arirang",
        "emoji": "⛰️",
        "korean": True,
        "category": "story",
        "color": "#FBE7A8",
        "accent": "#E8C85A",
        "bpm": 84,
        "style": "play",
        "notes": (
            "G4q E4q G4q A4q G4q E4q G4q E4q D4q C4h. "
            "G4q E4q G4q A4q G4q E4q D4q C4q A3q C4h. "
            "A3q C4q D4q E4q G4q A4q G4q E4q D4q C4h. "
            "G4q E4q G4q A4q G4q E4q D4q C4q A3q C4h. "
            "G4q E4q G4q A4q G4q E4q G4q E4q D4q C4h. "
            "G4q E4q G4q A4q G4q E4q D4q C4q A3q C4h."
        ),
        "lines": [
            {"beat": 0, "ko": "아리랑 아리랑 아라리요", "en": "Arirang, arirang, arariyo"},
            {"beat": 12, "ko": "아리랑 고개로 넘어간다", "en": "Going over Arirang hill"},
            {"beat": 24, "ko": "나를 버리고 가시는 님은", "en": "My love, you are leaving me"},
            {"beat": 36, "ko": "십리도 못 가서 발병 난다", "en": "Your feet will hurt before ten li"},
            {"beat": 48, "ko": "아리랑 아리랑 아라리요", "en": "Arirang, arirang, arariyo"},
            {"beat": 60, "ko": "아리랑 고개로 넘어간다", "en": "Going over Arirang hill"},
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
        "bpm": 96,
        "style": "play",
        "perc": True,
        # 질문 소절 계이름: 미라라 라라솔 미라솔라
        "notes": (
            "E4q A4q A4q A4q A4q G4q E4q A4q G4e A4e "
            "A4q A4q G4q E4e A4e G4e A4e Restq "
            "E4q A4q A4q A4q A4q G4q E4q A4q G4e A4e "
            "E4e A4e G4e A4e Restq A4q A4q G4q "
            "E4q A4q A4q A4q A4q G4q E4q A4q G4e A4e "
            "A4q A4q G4q E4e A4e G4e A4e Restq"
        ),
        "lines": [
            {"beat": 0, "ko": "여우야 여우야 뭐하니", "en": "Fox, fox, what are you doing"},
            {"beat": 9, "ko": "잠잔다", "en": "I'm sleeping"},
            {"beat": 12, "ko": "잠꾸러기", "en": "Sleepyhead"},
            {"beat": 15, "ko": "여우야 여우야 뭐하니", "en": "Fox, fox, what are you doing"},
            {"beat": 24, "ko": "세수한다", "en": "Washing my face"},
            {"beat": 27, "ko": "멋쟁이", "en": "Looking fancy"},
            {"beat": 30, "ko": "여우야 여우야 뭐하니", "en": "Fox, fox, what are you doing"},
            {"beat": 39, "ko": "잠잔다", "en": "I'm sleeping"},
            {"beat": 42, "ko": "잠꾸러기", "en": "Sleepyhead"},
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
        "bpm": 92,
        "style": "play",
        "perc": True,
        # 8음절 한 줄. 새야새야와 같은 전래 선율 뼈대: 솔미솔미 라솔미도 / 솔미라솔 미레도도
        "notes": (
            "G4q E4q G4q E4q A4q G4q E4q C4q "
            "G4q E4q A4q G4q E4q D4q C4q C4q "
            "G4q E4q G4q E4q A4q G4q E4q C4q "
            "G4q E4q A4q G4q E4q D4q C4q C4q "
            "G4q E4q G4q E4q A4q G4q E4q C4q "
            "G4q E4q A4q G4q E4q D4q C4q C4q"
        ),
        "lines": [
            {"beat": 0, "ko": "달아 달아 밝은 달아", "en": "Moon, moon, bright moon"},
            {"beat": 8, "ko": "이태백이 놀던 달아", "en": "The moon where Li Bai played"},
            {"beat": 16, "ko": "저기 저기 저 달 속에", "en": "There, there, inside that moon"},
            {"beat": 24, "ko": "계수나무 박혔으니", "en": "A cinnamon tree is growing"},
            {"beat": 32, "ko": "옥도끼로 찍어 내어", "en": "Chop it out with a jade axe"},
            {"beat": 40, "ko": "금도끼로 다듬어서", "en": "Trim it with a golden axe"},
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
            {"beat": 6, "ko": "강을 따라가요", "en": "Gently down the stream"},
            {"beat": 12, "ko": "즐겁게 즐겁게 즐겁게 즐겁게", "en": "Merrily, merrily, merrily, merrily"},
            {"beat": 18, "ko": "인생은 꿈같아요", "en": "Life is but a dream"},
            {"beat": 24, "ko": "노를 저어라 노를 저어라", "en": "Row, row, row your boat"},
            {"beat": 30, "ko": "강을 따라가요", "en": "Gently down the stream"},
            {"beat": 36, "ko": "즐겁게 즐겁게 즐겁게 즐겁게", "en": "Merrily, merrily, merrily, merrily"},
            {"beat": 42, "ko": "인생은 꿈같아요", "en": "Life is but a dream"},
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
            "B4q B4q A4q A4q G4w "
            "G4q G4q G4q D4q E4q E4q D4h "
            "B4q B4q A4q A4q G4w "
            "G4q G4q G4q D4q E4q E4q D4h "
            "B4q B4q A4q A4q G4w "
            "G4q G4q G4q D4q E4q E4q D4h "
            "B4q B4q A4q A4q G4w"
        ),
        "lines": [
            {"beat": 0, "ko": "올드 맥도널드 농장에", "en": "Old MacDonald had a farm"},
            {"beat": 8, "ko": "이아이오", "en": "E-I-E-I-O"},
            {"beat": 16, "ko": "그 농장에 병아리가", "en": "And on that farm he had a chick"},
            {"beat": 24, "ko": "이아이오", "en": "E-I-E-I-O"},
            {"beat": 32, "ko": "짹짹 여기 짹짹 저기", "en": "With a chick-chick here and there"},
            {"beat": 40, "ko": "여기저기 짹짹짹", "en": "Here a chick, there a chick"},
            {"beat": 48, "ko": "올드 맥도널드 농장에", "en": "Old MacDonald had a farm"},
            {"beat": 56, "ko": "이아이오", "en": "E-I-E-I-O"},
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
            "C4q C4q C4q D4q E4q D4q C4h "
            "E4q D4q C4q D4q E4q C4h Restq "
            "G4q E4q C4q G3q C4w "
            "C4q C4q C4q D4q E4q D4q C4h "
            "C4q C4q C4q D4q E4q D4q C4h "
            "E4q D4q C4q D4q E4q C4h Restq "
            "G4q E4q C4q G3q C4w "
            "C4q C4q C4q D4q E4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "아기 거미가 줄 타고 올라가요", "en": "The itsy bitsy spider climbed up the spout"},
            {"beat": 8, "ko": "비가 와서 거미를 씻겨 내렸죠", "en": "Down came the rain and washed the spider out"},
            {"beat": 16, "ko": "해가 나와 빗물을 말리고", "en": "Out came the sun and dried up all the rain"},
            {"beat": 24, "ko": "아기 거미 다시 올라가요", "en": "And the itsy bitsy spider climbed up again"},
            {"beat": 32, "ko": "아기 거미가 줄 타고 올라가요", "en": "The itsy bitsy spider climbed up the spout"},
            {"beat": 40, "ko": "비가 와서 거미를 씻겨 내렸죠", "en": "Down came the rain and washed the spider out"},
            {"beat": 48, "ko": "해가 나와 빗물을 말리고", "en": "Out came the sun and dried up all the rain"},
            {"beat": 56, "ko": "아기 거미 다시 올라가요", "en": "And the itsy bitsy spider climbed up again"},
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
        "bpm": 96,
        "style": "play",
        "perc": True,
        "notes": (
            "G4q E4q G4q E4q A4q G4q E4q C4q "
            "G4q E4q A4q G4q E4q D4q C4q C4q "
            "G4q E4q G4q E4q A4q G4q E4q C4q "
            "G4q E4q A4q G4q E4q D4q C4q C4q "
            "G4q E4q G4q E4q A4q G4q E4q C4q "
            "G4q E4q A4q G4q E4q D4q C4q C4q"
        ),
        "lines": [
            {"beat": 0, "ko": "새야 새야 파랑새야", "en": "Bird, bird, little blue bird"},
            {"beat": 8, "ko": "녹두밭에 앉지 마라", "en": "Don't sit in the bean field"},
            {"beat": 16, "ko": "녹두꽃이 떨어지면", "en": "If the mung flowers fall"},
            {"beat": 24, "ko": "청포장수 울고 간다", "en": "The jelly seller starts to cry"},
            {"beat": 32, "ko": "새야 새야 파랑새야", "en": "Bird, bird, little blue bird"},
            {"beat": 40, "ko": "녹두밭에 앉지 마라", "en": "Don't sit in the bean field"},
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
        "id": "ganggang",
        "titleKo": "강강술래",
        "titleEn": "Ganggangsullae",
        "emoji": "🌕",
        "korean": True,
        "category": "play",
        "color": "#FFE0F0",
        "accent": "#F08AB8",
        "bpm": 96,
        "style": "play",
        "perc": True,
        # 메기고 받는 4음. 라솔미도
        "notes": (
            "A4q G4q E4q C4q A4q G4q E4q A3q "
            "A4q G4q E4q C4q A4q G4q E4q A3q "
            "A4q G4q E4q C4q A4q G4q E4q A3q "
            "A4q G4q E4q C4q A4q G4q E4q A3q "
            "A4q G4q E4q C4q A4q G4q E4q A3q "
            "A4q G4q E4q C4q A4q G4q E4q A3q"
        ),
        "lines": [
            {"beat": 0, "ko": "달 떠온다", "en": "The moon is rising"},
            {"beat": 4, "ko": "강강술래", "en": "Ganggangsullae"},
            {"beat": 8, "ko": "동해 동천", "en": "From the eastern sky"},
            {"beat": 12, "ko": "강강술래", "en": "Ganggangsullae"},
            {"beat": 16, "ko": "이태백이", "en": "Where Li Bai"},
            {"beat": 20, "ko": "강강술래", "en": "Ganggangsullae"},
            {"beat": 24, "ko": "놀던 달아", "en": "Used to play"},
            {"beat": 28, "ko": "강강술래", "en": "Ganggangsullae"},
            {"beat": 32, "ko": "달 떠온다", "en": "The moon is rising"},
            {"beat": 36, "ko": "강강술래", "en": "Ganggangsullae"},
            {"beat": 40, "ko": "강강술래", "en": "Ganggangsullae"},
            {"beat": 44, "ko": "강강술래", "en": "Ganggangsullae"},
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
            "G4e F4e E4e D4e C4h Restq Restq"
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
        "id": "three-mice",
        "titleKo": "생쥐 세 마리",
        "titleEn": "Three Blind Mice",
        "emoji": "🐭",
        "category": "animal",
        "color": "#F5D0D8",
        "accent": "#E08AA0",
        "bpm": 86,
        "style": "play",
        "notes": (
            "E4q. D4e C4h E4q. D4e C4h "
            "G4q G4q G4e F4e E4q. D4e C4h "
            "G4q G4q G4e F4e E4q. D4e C4h "
            "E4q. D4e C4h E4q. D4e C4h "
            "G4q G4q G4e F4e E4q. D4e C4h "
            "G4q G4q G4e F4e E4q. D4e C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "생쥐 세 마리", "en": "Three blind mice"},
            {"beat": 4, "ko": "생쥐 세 마리", "en": "Three blind mice"},
            {"beat": 8, "ko": "어떻게 달려가나요", "en": "See how they run"},
            {"beat": 15, "ko": "어떻게 달려가나요", "en": "See how they run"},
            {"beat": 22, "ko": "생쥐 세 마리", "en": "Three blind mice"},
            {"beat": 26, "ko": "생쥐 세 마리", "en": "Three blind mice"},
            {"beat": 30, "ko": "어떻게 달려가나요", "en": "See how they run"},
            {"beat": 37, "ko": "어떻게 달려가나요", "en": "See how they run"},
        ],
    },
    {
        "id": "humpty",
        "titleKo": "험프티 덤프티",
        "titleEn": "Humpty Dumpty",
        "emoji": "🥚",
        "category": "story",
        "color": "#F8E8C0",
        "accent": "#E2C56A",
        "bpm": 90,
        "style": "play",
        "notes": (
            "G4q G4q G4q E4q G4q G4q G4q E4q "
            "G4q C5q B4q A4q G4h Restq "
            "A4q A4q A4q F4q A4q A4q A4q F4q "
            "G4q A4q G4q F4q E4h Restq "
            "G4q G4q G4q E4q G4q G4q G4q E4q "
            "G4q C5q B4q A4q G4h Restq "
            "A4q A4q A4q F4q A4q A4q A4q F4q "
            "G4q A4q G4q F4q E4w"
        ),
        "lines": [
            {"beat": 0, "ko": "험프티 덤프티 담장 위에", "en": "Humpty Dumpty sat on a wall"},
            {"beat": 8, "ko": "험프티 덤프티 떨어졌네", "en": "Humpty Dumpty had a great fall"},
            {"beat": 15, "ko": "임금님의 말과 병사들이", "en": "All the king's horses and all the king's men"},
            {"beat": 23, "ko": "다시 붙일 수 없었죠", "en": "Couldn't put Humpty together again"},
            {"beat": 29, "ko": "험프티 덤프티 담장 위에", "en": "Humpty Dumpty sat on a wall"},
            {"beat": 37, "ko": "험프티 덤프티 떨어졌네", "en": "Humpty Dumpty had a great fall"},
            {"beat": 44, "ko": "임금님의 말과 병사들이", "en": "All the king's horses and all the king's men"},
            {"beat": 52, "ko": "다시 붙일 수 없었죠", "en": "Couldn't put Humpty together again"},
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
        "id": "bo-peep",
        "titleKo": "꼬마 보핍",
        "titleEn": "Little Bo-Peep",
        "emoji": "🎀",
        "category": "story",
        "color": "#F6D8EC",
        "accent": "#E090C0",
        "bpm": 88,
        "style": "lullaby",
        "notes": (
            "G4q. A4e G4q E4q C4q D4q E4h "
            "G4q G4q A4q G4q E4h Restq "
            "G4q. A4e G4q E4q C4q D4q E4h "
            "D4q E4q D4q B3q C4h Restq "
            "G4q. A4e G4q E4q C4q D4q E4h "
            "G4q G4q A4q G4q E4h Restq "
            "G4q. A4e G4q E4q C4q D4q E4h "
            "D4q E4q D4q B3q C4w"
        ),
        "lines": [
            {"beat": 0, "ko": "꼬마 보핍이 양을 잃었어요", "en": "Little Bo-Peep has lost her sheep"},
            {"beat": 8, "ko": "어디에 있는지 몰라요", "en": "And can't tell where to find them"},
            {"beat": 15, "ko": "그냥 두렴 집으로 올 거야", "en": "Leave them alone and they'll come home"},
            {"beat": 23, "ko": "꼬리를 흔들며 온대요", "en": "Wagging their tails behind them"},
            {"beat": 30, "ko": "꼬마 보핍이 양을 잃었어요", "en": "Little Bo-Peep has lost her sheep"},
            {"beat": 38, "ko": "어디에 있는지 몰라요", "en": "And can't tell where to find them"},
            {"beat": 45, "ko": "그냥 두렴 집으로 올 거야", "en": "Leave them alone and they'll come home"},
            {"beat": 53, "ko": "꼬리를 흔들며 온대요", "en": "Wagging their tails behind them"},
        ],
    },
    {
        "id": "this-old-man",
        "titleKo": "이 할아버지",
        "titleEn": "This Old Man",
        "emoji": "🥁",
        "category": "play",
        "color": "#F8DCC8",
        "accent": "#E0A07A",
        "bpm": 108,
        "style": "play",
        "perc": True,
        "notes": (
            "G4q E4q G4h G4q E4q G4h "
            "A4q G4q F4q E4q D4q E4q F4h "
            "E4q F4q G4q C4q C4q C4q "
            "D4q E4q D4q C4q G4q E4q D4q C4h "
            "G4q E4q G4h G4q E4q G4h "
            "A4q G4q F4q E4q D4q E4q F4h "
            "E4q F4q G4q C4q C4q C4q "
            "D4q E4q D4q C4q G4q E4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "이 할아버지 하나", "en": "This old man, he played one"},
            {"beat": 8, "ko": "엄지 손가락으로 북을 치네", "en": "He played knick-knack on my thumb"},
            {"beat": 16, "ko": "닉낵 패디왁 북을 치며", "en": "With a knick-knack paddywhack"},
            {"beat": 22, "ko": "강아지에게 뼈다귀", "en": "Give a dog a bone"},
            {"beat": 31, "ko": "이 할아버지 둘", "en": "This old man, he played two"},
            {"beat": 39, "ko": "신발 위에서 북을 치네", "en": "He played knick-knack on my shoe"},
            {"beat": 47, "ko": "닉낵 패디왁 북을 치며", "en": "With a knick-knack paddywhack"},
            {"beat": 53, "ko": "이 할아버지 집으로", "en": "This old man came rolling home"},
        ],
    },
    {
        "id": "brahms",
        "titleKo": "브람스 자장가",
        "titleEn": "Brahms' Lullaby",
        "emoji": "🌙",
        "category": "lullaby",
        "color": "#D9D4F7",
        "accent": "#9B90D8",
        "bpm": 72,
        "style": "lullaby",
        "notes": (
            # Wiegenlied, 3/4. 여린내박 미미 + 점4분·8분·4분 세 마디 = 12박.
            "E4e E4e G4q. E4e E4q G4q. E4e G4q C5q. B4e A4q Resth "
            "D4e D4e F4q. D4e D4q F4q. D4e F4q B4q. A4e G4q Resth "
            "E4e E4e G4q. E4e E4q G4q. E4e G4q C5q. B4e A4q Resth "
            "A4e F4e A4q. F4e D4q G4q. E4e C4q D4q. B3e C4q Resth "
            "E4e E4e G4q. E4e E4q G4q. E4e G4q C5q. B4e A4q Resth "
            "D4e D4e F4q. D4e D4q F4q. D4e F4q B4q. A4e G4q Resth "
            "E4e E4e G4q. E4e E4q G4q. E4e G4q C5q. B4e A4q Resth "
            "A4e F4e A4q. F4e D4q G4q. E4e C4q D4q. B3e C4q Resth"
        ),
        "lines": [
            {"beat": 0, "ko": "잘 자렴 아기야", "en": "Lullaby, and good night"},
            {"beat": 12, "ko": "엄마 품에서", "en": "With roses bedight"},
            {"beat": 24, "ko": "별이 반짝이면", "en": "With lilies o'er spread"},
            {"beat": 36, "ko": "꿈나라로 가요", "en": "Is baby's wee bed"},
            {"beat": 48, "ko": "잘 자렴 아기야", "en": "Lullaby, and good night"},
            {"beat": 60, "ko": "포근한 자장가", "en": "Thy mother's delight"},
            {"beat": 72, "ko": "달빛 아래 살며시", "en": "Lay thee down now and rest"},
            {"beat": 84, "ko": "눈을 감아요", "en": "May thy slumber be blessed"},
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
        "id": "farmer-dell",
        "titleKo": "농부 아저씨",
        "titleEn": "The Farmer in the Dell",
        "emoji": "🚜",
        "category": "play",
        "color": "#E4F4C8",
        "accent": "#A8CC6A",
        "bpm": 100,
        "style": "play",
        "perc": True,
        "notes": (
            "G4q G4q G4e A4e G4q E4h "
            "G4q G4q G4e A4e G4q E4h "
            "C5q C5q C5q A4h "
            "G4q G4q G4e A4e G4q E4q D4q C4h "
            "G4q G4q G4e A4e G4q E4h "
            "G4q G4q G4e A4e G4q E4h "
            "C5q C5q C5q A4h "
            "G4q G4q G4e A4e G4q E4q D4q C4h"
        ),
        "lines": [
            {"beat": 0, "ko": "농부 아저씨가 들에", "en": "The farmer in the dell"},
            {"beat": 6, "ko": "농부 아저씨가 들에", "en": "The farmer in the dell"},
            {"beat": 12, "ko": "하이호 체리오", "en": "Hi-ho the derry-o"},
            {"beat": 17, "ko": "농부 아저씨가 들에", "en": "The farmer in the dell"},
            {"beat": 25, "ko": "농부가 부인을 데려와요", "en": "The farmer takes a wife"},
            {"beat": 31, "ko": "농부가 부인을 데려와요", "en": "The farmer takes a wife"},
            {"beat": 37, "ko": "하이호 체리오", "en": "Hi-ho the derry-o"},
            {"beat": 42, "ko": "농부가 부인을 데려와요", "en": "The farmer takes a wife"},
        ],
    },
]


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog = []
    for i, song in enumerate(SONGS, 1):
        notes = parse_notes(song["notes"])
        total_beats = sum(d for _, d in notes)
        beats = [float(line["beat"]) for line in song["lines"]]
        if beats != sorted(beats) or beats[0] != 0:
            raise ValueError(f"{song['id']} lyric beats must start at 0 and increase")
        if beats[-1] >= total_beats:
            raise ValueError(f"{song['id']} last lyric {beats[-1]} >= {total_beats} beats")
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
                "bpm": song_bpm(song),
                "audio": f"/audio/{song['id']}.mp3",
                "duration": round(duration, 2),
                "lyrics": lyrics,
                "source": "전래·퍼블릭 도메인 계이름 · 펠트 피아노와 따뜻한 화음",
            }
        )
        print(f"{i:02d} {song['id']:16s} {duration:6.1f}s  {song['titleKo']}")
    (DATA_DIR / "songs.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", len(catalog), "songs")


if __name__ == "__main__":
    main()
