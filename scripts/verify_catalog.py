#!/usr/bin/env python3
"""Catalog checks for the 30-song baby nursery app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src" / "data" / "songs.json"
AUDIO = ROOT / "public" / "audio"


def main() -> int:
    songs = json.loads(CATALOG.read_text(encoding="utf-8"))
    errors: list[str] = []
    if len(songs) != 30:
        errors.append(f"expected 30 songs, got {len(songs)}")
    ids = [s["id"] for s in songs]
    if len(ids) != len(set(ids)):
        errors.append("duplicate song ids")
    for song in songs:
        path = ROOT / "public" / song["audio"].lstrip("/")
        if not path.exists():
            errors.append(f"missing audio {path.name}")
        lyrics = song["lyrics"]
        if len(lyrics) < 4:
            errors.append(f"{song['id']} has too few lyric lines")
        prev = -1.0
        for line in lyrics:
            if line["end"] <= line["start"]:
                errors.append(f"{song['id']} lyric end <= start")
            if line["start"] < prev:
                errors.append(f"{song['id']} lyrics are not ordered")
            prev = line["start"]
            if not line["ko"] or not line["en"]:
                errors.append(f"{song['id']} missing bilingual lyric")
        if lyrics[-1]["end"] > song["duration"] + 1.5:
            errors.append(f"{song['id']} lyrics run past duration")
    if errors:
        print("FAIL")
        print("\n".join(errors))
        return 1
    print(f"OK  {len(songs)} songs, {sum(len(s['lyrics']) for s in songs)} lyric lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
