"""Build themed YAMNet label catalog from the bundled class map."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "models" / "yamnet_class_map.csv"
DEFAULT_OUT = REPO_ROOT / "config" / "yamnet_label_catalog.json"

UPSTREAM_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)

THEME_RULES: list[tuple[str, list[str]]] = [
    (
        "Human voice",
        [
            "speech",
            "conversation",
            "narration",
            "babbling",
            "shout",
            "bellow",
            "whoop",
            "yell",
            "screaming",
            "whisper",
            "children shouting",
            "speech synthesizer",
        ],
    ),
    (
        "Human non-speech",
        [
            "laughter",
            "giggle",
            "snicker",
            "chuckle",
            "crying",
            "sobbing",
            "whimper",
            "wail",
            "moan",
            "sigh",
            "breathing",
            "wheeze",
            "snoring",
            "cough",
            "sneeze",
            "sniff",
            "throat clearing",
            "hiccup",
            "fart",
            "hands",
            "clapping",
            "finger snapping",
            "footsteps",
            "walk",
            "run",
            "crowd",
            "hubbub",
            "chatter",
            "boo",
            "applause",
            "cheer",
            "children playing",
            "groan",
            "grunt",
            "whistling",
        ],
    ),
    (
        "Music",
        [
            "music",
            "song",
            "singing",
            "choir",
            "yodeling",
            "chant",
            "mantra",
            "rapping",
            "humming",
            "guitar",
            "piano",
            "drum",
            "organ",
            "violin",
            "cello",
            "flute",
            "trumpet",
            "saxophone",
            "harmonica",
            "accordion",
            "banjo",
            "harp",
            "gong",
            "marimba",
            "xylophone",
            "orchestra",
            "brass",
            "woodwind",
            "keyboard",
            "percussion",
            "cymbal",
            "hi-hat",
            "snare",
            "bass drum",
            "tabla",
            "tambourine",
            "shaker",
            "mallet",
            "pluck",
            "bowed",
            "techno",
            "disco",
            "funk",
            "hip hop",
            "pop music",
            "rock music",
            "punk",
            "heavy metal",
            "reggae",
            "country",
            "swing",
            "bluegrass",
            "classical music",
            "opera",
            "lullaby",
            "salsa",
            "flamenco",
            "blues",
            "folk music",
            "jazz",
            "rhythm and blues",
            "soul music",
            "gospel",
            "electronic music",
        ],
    ),
    (
        "Animals",
        [
            "dog",
            "bark",
            "howl",
            "bow-wow",
            "growling",
            "cat",
            "purr",
            "meow",
            "hiss",
            "caterwaul",
            "livestock",
            "horse",
            "clip-clop",
            "neigh",
            "cattle",
            "moo",
            "cowbell",
            "pig",
            "oink",
            "goat",
            "bleat",
            "sheep",
            "fowl",
            "chicken",
            "rooster",
            "turkey",
            "duck",
            "goose",
            "wild animals",
            "roaring cats",
            "bird",
            "birdsong",
            "chirp",
            "squawk",
            "pigeon",
            "crow",
            "owl",
            "insect",
            "cricket",
            "mosquito",
            "fly, housefly",
            "bee",
            "wasp",
            "frog",
            "toad",
            "snake",
            "whale vocalization",
        ],
    ),
    (
        "Vehicles",
        [
            "vehicle",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "scooter",
            "moped",
            "bicycle",
            "skateboard",
            "traffic",
            "engine",
            "accelerat",
            "idling",
            "reverse beeps",
            "ice cream truck",
            "emergency vehicle",
            "police car",
            "ambulance",
            "fire engine",
            "motorboat",
            "ship",
            "sailboat",
            "rowboat",
            "submarine",
            "aircraft",
            "airplane",
            "helicopter",
            "train",
            "rail transport",
            "railroad car",
            "subway",
            "steam whistle",
        ],
    ),
    (
        "Alarms and signals",
        [
            "siren",
            "civil defense siren",
            "buzzer",
            "smoke detector",
            "fire alarm",
            "foghorn",
            "whistle",
            "air horn",
            "car horn",
            "vehicle horn",
            "toot",
            "beep",
            "ding",
            "telephone",
            "ringtone",
            "dial tone",
            "busy signal",
            "alarm clock",
            "chime",
        ],
    ),
    (
        "Tools and industry",
        [
            "tool",
            "hammer",
            "jackhammer",
            "sawing",
            "cutting",
            "filing",
            "sanding",
            "power tool",
            "drill",
            "explosion",
            "gunshot",
            "machine gun",
            "artillery",
            "fireworks",
            "firecracker",
            "boom",
            "bang",
            "smash",
            "crash",
            "breaking",
            "chop",
            "tear",
            "slam",
            "thump",
            "thud",
            "knock",
            "tap",
            "click",
            "clink",
            "jingle",
            "squeak",
            "creak",
            "rustle",
            "scrape",
            "crush",
            "whoosh",
            "vibration",
            "typing",
            "printer",
            "sewing machine",
            "vacuum cleaner",
            "dishwasher",
            "blender",
            "microwave oven",
            "hair dryer",
            "mechanical fan",
            "air conditioning",
            "laundry",
        ],
    ),
    (
        "Domestic / interior",
        [
            "door",
            "doorbell",
            "cupboard",
            "drawer",
            "dishes",
            "cutlery",
            "chopping (food)",
            "keys jangling",
            "coin (dropping)",
            "scissors",
            "toothbrush",
            "bathtub",
            "shower",
            "sink (filling or washing)",
            "water tap",
            "toilet flush",
            "writing",
        ],
    ),
    (
        "Nature and weather",
        [
            "rain",
            "raindrop",
            "stream",
            "waterfall",
            "ocean",
            "waves",
            "steam",
            "fire",
            "crackle",
            "wind",
            "rustling leaves",
            "thunderstorm",
            "thunder",
        ],
    ),
    (
        "Environments / noise",
        [
            "silence",
            "noise",
            "pink noise",
            "white noise",
            "static",
            "hum",
            "environmental noise",
            "sound effect",
            "field recording",
            "outside",
            "inside",
            "echo",
            "reverberation",
            "cacophony",
        ],
    ),
]

THEME_ORDER = [name for name, _ in THEME_RULES] + ["Other"]


def theme_for(display_name: str) -> str:
    lowered = display_name.lower()
    for theme, keys in THEME_RULES:
        for key in keys:
            if key in lowered:
                return theme
    return "Other"


def load_class_map(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "index": int(row["index"]),
                    "mid": row["mid"],
                    "display_name": row["display_name"].strip().strip('"'),
                }
            )
        return rows


def build_catalog(class_rows: list[dict], *, source: str) -> dict:
    labels = [
        {**row, "theme": theme_for(row["display_name"])} for row in class_rows
    ]
    counts = Counter(item["theme"] for item in labels)
    themes = [theme for theme in THEME_ORDER if counts[theme]]
    return {
        "labels": labels,
        "themes": themes,
        "source": source,
        "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def sync_from_upstream(dest_csv: Path) -> Path:
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UPSTREAM_CLASS_MAP_URL, timeout=30) as response:
        dest_csv.write_bytes(response.read())
    return dest_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--from-upstream",
        action="store_true",
        help="Download the official YAMNet class map before building.",
    )
    args = parser.parse_args(argv)

    try:
        source = str(args.csv.relative_to(REPO_ROOT))
    except ValueError:
        source = str(args.csv)
    if args.from_upstream:
        sync_from_upstream(args.csv)
        source = f"{UPSTREAM_CLASS_MAP_URL} -> {source}"

    catalog = build_catalog(load_class_map(args.csv), source=source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    counts = Counter(item["theme"] for item in catalog["labels"])
    print(f"Wrote {len(catalog['labels'])} labels to {args.out}")
    print(
        "Themes:",
        ", ".join(f"{theme}={counts[theme]}" for theme in catalog["themes"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
