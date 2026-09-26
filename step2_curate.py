"""
step2_curate.py — inspect and clean your generated pool.

Run it in three modes:

    python step2_curate.py stats      Show what you have. Start here.
    python step2_curate.py dupes      Find repeated and near-identical lines.
    python step2_curate.py review     Read them one by one, keep or delete.

Input:  exchanges.json
Output: exchanges_curated.json   (only what you kept)

At 500 exchanges the real problem is repetition, not individual bad
lines. Run 'stats' and 'dupes' first — they will tell you what to look
for before you start reading.
"""

import json
import os
import re
import sys
from collections import Counter

INPUT_FILE = "exchanges.json"
OUTPUT_FILE = "exchanges_curated.json"


def load(path):
    if not os.path.exists(path):
        print(f"Cannot find {path}. Run this from your project folder.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def all_texts(pool):
    return [t["text"] for ex in pool for t in ex["transmissions"]]


def normalise(text):
    """Strip callsigns, numbers and punctuation so we can compare the
    underlying phrasing rather than the specific details."""
    t = text.lower()
    t = re.sub(r'\b(alpha|bravo|charlie|delta|echo|foxtrot)[- ]?\d*\b', 'X', t)
    t = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|zero|niner)\b', 'N', t)
    t = re.sub(r'\d+', 'N', t)
    t = re.sub(r'[^\w\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


# ---------------------------------------------------------------------------

def show_stats(pool):
    texts = all_texts(pool)

    print(f"Exchanges:     {len(pool)}")
    print(f"Transmissions: {len(texts)}")
    print()

    print("By scenario:")
    for name, count in Counter(ex["scenario"] for ex in pool).most_common():
        bar = "#" * int(count / max(1, len(pool)) * 40)
        print(f"  {name:20s} {count:4d}  {bar}")
    print()

    lengths = Counter(len(ex["transmissions"]) for ex in pool)
    print("Exchange length:")
    for n in sorted(lengths):
        print(f"  {n} lines: {lengths[n]}")
    print()

    words = [len(t.split()) for t in texts]
    print(f"Words per transmission: avg {sum(words)/len(words):.1f}, "
          f"min {min(words)}, max {max(words)}")
    print()

    # Variety is the number that matters most
    unique = len(set(normalise(t) for t in texts))
    variety = unique / len(texts) * 100
    print(f"Unique phrasings: {unique} of {len(texts)}  ({variety:.0f}% variety)")
    if variety < 60:
        print("  ^ LOW. The model is repeating itself heavily.")
        print("    Add more scenarios and vary the example lines in your prompt.")
    elif variety < 80:
        print("  ^ Acceptable, but there is room to improve.")
    else:
        print("  ^ Good.")
    print()

    print("Most repeated openings:")
    openings = Counter(" ".join(t.split()[:4]).lower() for t in texts)
    for phrase, count in openings.most_common(10):
        print(f"  {count:4d}x  {phrase}...")
    print()

    print("Grid references used:")
    grids = Counter()
    for t in texts:
        for m in re.findall(r'grid[\s\w]{0,40}', t.lower()):
            grids[m.strip()] += 1
    if grids:
        for grid, count in grids.most_common(8):
            print(f"  {count:4d}x  {grid}")
        print()
        print("  Check these are consistent. If your patrols are scattered")
        print("  across unrelated invented grids, the session will not hold")
        print("  together geographically. Consider rewriting them from a")
        print("  fixed fictional list.")
    else:
        print("  none found")


def show_dupes(pool):
    texts = all_texts(pool)
    groups = {}
    for t in texts:
        groups.setdefault(normalise(t), []).append(t)

    repeated = {k: v for k, v in groups.items() if len(v) > 2}
    repeated = dict(sorted(repeated.items(), key=lambda x: -len(x[1])))

    print(f"Phrasings appearing 3+ times: {len(repeated)}")
    print()

    for i, (_, variants) in enumerate(list(repeated.items())[:25], 1):
        print(f"{i}. appears {len(variants)}x")
        for v in list(dict.fromkeys(variants))[:3]:
            print(f'     "{v}"')
        print()

    print("Some repetition is realistic — radio procedure IS repetitive.")
    print("Worry when the same full sentence appears many times with")
    print("nothing changed but the callsign.")


def review(pool):
    kept = []
    print(f"Reviewing {len(pool)} exchanges.")
    print("  k = keep      d = delete      q = quit and save")
    print()

    for i, ex in enumerate(pool, 1):
        print("=" * 58)
        print(f"[{i}/{len(pool)}]  {ex['scenario']}   (kept so far: {len(kept)})")
        print("-" * 58)
        for t in ex["transmissions"]:
            print(f'  {t["callsign"]}: "{t["text"]}"')
        print("-" * 58)

        while True:
            choice = input("  k / d / q > ").strip().lower()
            if choice in ("k", "d", "q", ""):
                break

        if choice == "q":
            print("\nStopping early.")
            break
        if choice in ("k", ""):
            kept.append(ex)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print()
    print(f"Kept {len(kept)}. Saved to {OUTPUT_FILE}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stats"
    data = load(INPUT_FILE)

    if mode == "stats":
        show_stats(data)
    elif mode == "dupes":
        show_dupes(data)
    elif mode == "review":
        review(data)
    else:
        print("Use: stats, dupes, or review")