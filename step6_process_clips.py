"""
step6_process_clips.py — Phase 3. Apply radio character to every rendered clip.

Reads:  manifest.json + clips\\*.wav
Writes: processed\\*.wav
        manifest_processed.json

Requires step5_radio_fx.py in the same folder.

    python step6_process_clips.py

Fast — this is plain DSP, no AI. A few hundred clips take a couple of
minutes on any machine.

---------------------------------------------------------------------------
WHY EACH CALLSIGN GETS A DIFFERENT PROFILE

This does more perceptual work than the voices do.

A base station and a handheld three kilometres out genuinely sound
different over the air — different bandwidth, different noise floor,
different signal stability. Reproducing that separates your stations far
more effectively than voice timbre, and it is physically correct rather
than a trick.

It also covers for having only two source speakers: radio flattens voices
in real life, which is part of why identifying people over a poor link is
hard.
---------------------------------------------------------------------------
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

try:
    from step5_radio_fx import PROFILES, process
except ImportError:
    print("Cannot find step5_radio_fx.py — it must be in this folder.")
    sys.exit(1)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

MANIFEST_IN = "manifest.json"
MANIFEST_OUT = "manifest_processed.json"
CLIPS_DIR = "clips"
OUT_DIR = "processed"

LIMIT = 0            # 0 = all clips
OVERWRITE = False    # True = reprocess clips that already exist

# ---------------------------------------------------------------------------
# PROFILE ASSIGNMENT
#
# Available: base, handheld, distant, air
#
#   base      strong, clean, authoritative — command posts, fixed sites
#   handheld  the default patrol sound
#   distant   weak, noisy, drifting — use SPARINGLY
#   air       bright and clean with rumble underneath
#
# Use 'distant' for one station only. One rough signal makes the others
# sound solid by contrast; several rough signals just sound broken.
# ---------------------------------------------------------------------------

CALLSIGN_PROFILES = {
    "ALPHA-1":   "base",       # company command post
    "BRAVO-6":   "handheld",   # forward patrol
    "CHARLIE-3": "handheld",
    "DELTA-2":   "handheld",   # vehicle patrol
    "ECHO-4":    "handheld",
    "GOLF-7":    "distant",    # static OP, poor position — the weak one
    "FOXTROT-5": "base",       # logistics, fixed location
    "HOTEL-9":   "base",       # battalion, strong signal
}

DEFAULT_PROFILE = "handheld"

# Deterministic processing: the same clip always gets the same noise and
# fading. Set to None for different randomness on every run.
SEED_BASE = 1000


# ===========================================================================

def main():
    if not os.path.exists(MANIFEST_IN):
        print(f"Cannot find {MANIFEST_IN}. Run step4_render_voices.py first.")
        sys.exit(1)

    with open(MANIFEST_IN, encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    if LIMIT:
        manifest = manifest[:LIMIT]

    print(f"{len(manifest)} clips to process")
    print()
    print("Profiles:")
    for callsign in sorted(CALLSIGN_PROFILES):
        print(f"  {callsign:12s} {CALLSIGN_PROFILES[callsign]}")
    print()

    out_manifest = []
    processed = 0
    skipped = 0
    failed = 0

    for i, entry in enumerate(manifest, 1):
        in_path = entry["file"].replace("/", os.sep)
        if not os.path.exists(in_path):
            print(f"  missing source: {in_path}")
            failed += 1
            continue

        callsign = entry["callsign"]
        profile_name = CALLSIGN_PROFILES.get(callsign, DEFAULT_PROFILE)
        profile = PROFILES[profile_name]

        filename = os.path.basename(in_path)
        out_path = os.path.join(OUT_DIR, filename)

        if os.path.exists(out_path) and not OVERWRITE:
            skipped += 1
        else:
            try:
                audio, sr = sf.read(in_path)
                seed = None if SEED_BASE is None else SEED_BASE + i
                out = process(audio, sr, profile, seed=seed)
                sf.write(out_path, out, sr)
                processed += 1
            except Exception as e:
                print(f"  FAILED {filename}: {type(e).__name__}: {e}")
                failed += 1
                continue

        duration = 0.0
        try:
            info = sf.info(out_path)
            duration = info.frames / info.samplerate
        except Exception:
            pass

        out_manifest.append({
            **entry,
            "file": out_path.replace("\\", "/"),
            "source_file": entry["file"],
            "profile": profile_name,
            "duration": round(duration, 2),
        })

        if i % 50 == 0:
            print(f"  {i}/{len(manifest)}", flush=True)

    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    print()
    print("=" * 58)
    print(f"Processed {processed}   Skipped {skipped}   Failed {failed}")

    total = sum(m["duration"] for m in out_manifest)
    print(f"Total audio: {total/60:.1f} minutes across {len(out_manifest)} clips")

    print()
    print("By profile:")
    by_profile = {}
    for m in out_manifest:
        by_profile.setdefault(m["profile"], []).append(m["duration"])
    for name in sorted(by_profile):
        d = by_profile[name]
        print(f"  {name:10s} {len(d):5d} clips  {sum(d)/60:6.1f} min")

    print()
    print(f"Manifest written to {MANIFEST_OUT}")
    print()
    print("LISTEN NOW — play a clip from processed\\ against the same file")
    print("in clips\\. The difference should be obvious: band-limited,")
    print("compressed, noise floor, key click and squelch tail.")
    print()
    print("Then play an ALPHA-1 clip against a GOLF-7 clip. They should")
    print("sound like two different radios, not just two voices.")


if __name__ == "__main__":
    main()