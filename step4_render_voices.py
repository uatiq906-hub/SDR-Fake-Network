"""
step4_render_voices.py — read your curated exchanges and render them all to audio.

Put this file in the same folder as exchanges_curated.json and run:

    python step4_render_voices.py

It will:
    1. Read exchanges_curated.json (or exchanges.json if that is absent)
    2. Render every transmission to its own WAV in clips\\
    3. Write manifest.json — the index Phase 3 and 4 read
    4. Report timing and an ETA as it goes

Safe to stop with Ctrl+C. Already-rendered files are skipped on restart,
so you can run it in sessions.

---------------------------------------------------------------------------
REFERENCE VOICES ARE OPTIONAL

If a refs\\ folder exists with recordings in it, each callsign gets its own
cloned voice. If not, everything is rendered with Chatterbox's built-in
voice — which works, but every station will sound like the same American
speaker.

Without refs this is a pipeline test, not a deliverable. Useful for
calibrating settings and for building the Phase 4 orchestrator against
real audio while you organise recording sessions.
---------------------------------------------------------------------------
"""

import json
import os
import signal
import sys
import time

import torch
import torchaudio as ta

# ===========================================================================
# CONFIGURATION
# ===========================================================================

INPUT_FILE = "exchanges_curated.json"
FALLBACK_FILE = "exchanges.json"
CLIPS_DIR = "clips"
REFS_DIR = "refs"
MANIFEST_FILE = "manifest.json"

LIMIT = 0             # 0 = all exchanges. Start at 3, listen, then set 0.
OVERWRITE = True      # True = re-render clips that already exist.
                      # Keep True while calibrating, set False for the
                      # full run so you can stop and resume.

# --- Delivery settings ---------------------------------------------------
#
# exaggeration: how expressive. Engine default is 0.5. Radio operators are
#               FLAT, so we sit below it. Higher sounds like acting.
# cfg_weight:   how closely it follows reference pacing. Lower is more
#               deliberate.
#
# These are the two settings worth calibrating before a large render.
# Render a few clips at 0.25 / 0.35 / 0.45, listen, then commit.

EXAGGERATION = 0.35
CFG_WEIGHT = 0.5

# --- Reference recordings ------------------------------------------------
# Filenames inside refs\. Ignored entirely if the folder is missing.
#
# This mapping is built from two recordings plus their pitch variants
# (make_variants.py). It is arranged so that NO TWO UNITS WHO TALK TO EACH
# OTHER share a voice — otherwise an exchange sounds like one person
# talking to himself, which is immediately obvious.
#
# Do not reshuffle it casually. Every pair in the scenario library was
# checked against this assignment.

VOICE_REFS = {
    "ALPHA-1":   {"ref": "voice_a.wav",      "exaggeration": 0.30},
    "BRAVO-6":   {"ref": "voice_b.wav",      "exaggeration": 0.40},
    "CHARLIE-3": {"ref": "voice_b_low.wav",  "exaggeration": 0.35},
    "DELTA-2":   {"ref": "voice_b_high.wav", "exaggeration": 0.33},
    "ECHO-4":    {"ref": "voice_a_high.wav", "exaggeration": 0.38},
    "GOLF-7":    {"ref": "voice_a_low.wav",  "exaggeration": 0.25},
    "FOXTROT-5": {"ref": "voice_b_low.wav",  "exaggeration": 0.28},
    "HOTEL-9":   {"ref": "voice_b.wav",      "exaggeration": 0.35},
}


# ===========================================================================

def load_exchanges():
    path = INPUT_FILE if os.path.exists(INPUT_FILE) else FALLBACK_FILE
    if not os.path.exists(path):
        print(f"Cannot find {INPUT_FILE} or {FALLBACK_FILE} in this folder.")
        print(f"Current folder: {os.getcwd()}")
        print("Move this script next to your exchanges file, or cd there first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Read {len(data)} exchanges from {path}")
    return data


def find_refs():
    """Return {callsign: settings} for refs that actually exist, or {}."""
    if not os.path.isdir(REFS_DIR):
        return {}

    found = {}
    for callsign, settings in VOICE_REFS.items():
        path = os.path.join(REFS_DIR, settings["ref"])
        if os.path.exists(path):
            found[callsign] = {**settings, "path": path}

    return found


def load_model(device):
    """Prefer Nano (fast on CPU), fall back to Turbo, then standard."""
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print("Loading Chatterbox Nano...")
        return ChatterboxTurboTTS.from_pretrained(device=device, nano=True), "nano"
    except Exception:
        pass

    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print("Loading Chatterbox Turbo...")
        return ChatterboxTurboTTS.from_pretrained(device=device), "turbo"
    except Exception:
        pass

    from chatterbox.tts import ChatterboxTTS
    print("Loading Chatterbox (standard)...")
    return ChatterboxTTS.from_pretrained(device=device), "standard"


def clean_for_speech(text):
    """Say callsigns as words. 'ALPHA-1' read literally comes out wrong."""
    spoken = {
        "ALPHA-1": "Alpha One", "BRAVO-6": "Bravo Six",
        "CHARLIE-3": "Charlie Three", "DELTA-2": "Delta Two",
        "ECHO-4": "Echo Four", "FOXTROT-5": "Foxtrot Five",
        "GOLF-7": "Golf Seven", "HOTEL-9": "Hotel Nine",
        "Alpha-1": "Alpha One", "Bravo-6": "Bravo Six",
        "Charlie-3": "Charlie Three", "Delta-2": "Delta Two",
        "Echo-4": "Echo Four", "Foxtrot-5": "Foxtrot Five",
        "Golf-7": "Golf Seven", "Hotel-9": "Hotel Nine",
    }
    for a, b in spoken.items():
        text = text.replace(a, b)
    return text.strip()


def safe_name(text):
    """Make a string safe for a Windows filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


# ===========================================================================

def main():
    exchanges = load_exchanges()
    os.makedirs(CLIPS_DIR, exist_ok=True)

    if LIMIT:
        exchanges = exchanges[:LIMIT]
        print(f"LIMIT set: rendering first {LIMIT} exchanges only")

    total_lines = sum(len(ex["transmissions"]) for ex in exchanges)

    refs = find_refs()
    if refs:
        print(f"Reference voices found for: {', '.join(sorted(refs))}")
        missing = sorted(set(VOICE_REFS) - set(refs))
        if missing:
            print(f"  no reference for {', '.join(missing)} "
                  f"— those will use the built-in voice")
    else:
        print("No refs\\ folder found — using the built-in voice for everything.")
        print("  Every callsign will sound like the same speaker.")
        print("  Fine for testing the pipeline; not a deliverable.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print()
    print(f"torch {torch.__version__}   device: {device}")
    if device == "cpu":
        print("  CPU only. This will be slow — that is expected.")
    print()

    model, engine = load_model(device)
    sample_rate = model.sr
    print(f"Engine: {engine}   sample rate: {sample_rate} Hz")
    print(f"{len(exchanges)} exchanges, {total_lines} transmissions to render")
    print("Ctrl+C is safe — finished clips are kept and skipped on restart.")
    print()

    manifest = []
    rendered = 0
    skipped = 0
    failed = 0
    times = []
    started = time.time()

    def write_manifest():
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def on_interrupt(signum, frame):
        print("\n\nInterrupted.")
        write_manifest()
        print(f"Rendered {rendered} this session. Manifest saved.")
        print("Run again to continue where this left off.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_interrupt)

    for index, ex in enumerate(exchanges, 1):
        ex_id = ex.get("id", index)
        scenario = safe_name(ex.get("scenario", "unknown"))

        for pos, t in enumerate(ex["transmissions"], 1):
            callsign = str(t["callsign"]).strip().upper()
            settings = refs.get(callsign)

            filename = f"{ex_id:04d}_{scenario}_{pos:02d}_{safe_name(callsign)}.wav"
            out_path = os.path.join(CLIPS_DIR, filename)

            if os.path.exists(out_path) and not OVERWRITE:
                skipped += 1
            else:
                kwargs = {
                    "exaggeration": (settings or {}).get("exaggeration", EXAGGERATION),
                    "cfg_weight": (settings or {}).get("cfg_weight", CFG_WEIGHT),
                }
                if settings:
                    kwargs["audio_prompt_path"] = settings["path"]

                start = time.time()
                try:
                    wav = model.generate(clean_for_speech(t["text"]), **kwargs)
                    ta.save(out_path, wav, sample_rate)
                except Exception as e:
                    print(f"  FAILED {filename}: {type(e).__name__}: {e}")
                    failed += 1
                    continue

                times.append(time.time() - start)
                rendered += 1

                # Progress with ETA, based on measured speed so far
                done = rendered + skipped
                avg = sum(times) / len(times)
                eta_min = (total_lines - done) * avg / 60
                print(f"  {done}/{total_lines}  {callsign:11s} "
                      f"{times[-1]:5.1f}s  ETA {eta_min:5.0f} min", flush=True)

            duration = 0.0
            try:
                info = ta.info(out_path)
                duration = info.num_frames / info.sample_rate
            except Exception:
                pass

            manifest.append({
                "file": out_path.replace("\\", "/"),
                "exchange_id": ex_id,
                "scenario": ex.get("scenario", "unknown"),
                "thread_id": ex.get("thread_id"),
                "thread_stage": ex.get("thread_stage"),
                "thread_gap_seconds": ex.get("thread_gap_seconds"),
                "position": pos,
                "is_last": pos == len(ex["transmissions"]),
                "callsign": callsign,
                "text": t["text"],
                "reference": settings["ref"] if settings else None,
                "duration": round(duration, 2),
                "sample_rate": sample_rate,
                "conditions": ex.get("conditions", {}),
            })

            if len(manifest) % 25 == 0:
                write_manifest()

    write_manifest()

    # -----------------------------------------------------------------------
    elapsed = time.time() - started
    print()
    print("=" * 60)
    print(f"Rendered {rendered}   Skipped {skipped}   Failed {failed}")
    print(f"Elapsed: {elapsed/60:.1f} minutes")

    if times:
        avg = sum(times) / len(times)
        print(f"Average: {avg:.1f} s per transmission on {device}")
        print()
        print("Projected for a larger pool at this speed:")
        for n in (100, 200, 400):
            clips = n * 4
            print(f"   {n:4d} exchanges (~{clips} clips): "
                  f"{clips * avg / 3600:5.1f} hours")

    total_audio = sum(m["duration"] for m in manifest)
    print()
    print(f"Total speech: {total_audio/60:.1f} minutes across {len(manifest)} clips")
    if manifest:
        print(f"Average transmission: {total_audio/len(manifest):.1f} seconds")

    print()
    print("Per callsign:")
    by_callsign = {}
    for m in manifest:
        by_callsign.setdefault(m["callsign"], []).append(m["duration"])
    for callsign in sorted(by_callsign):
        d = by_callsign[callsign]
        print(f"  {callsign:12s} {len(d):5d} clips  {sum(d)/60:6.1f} min")

    print()
    print(f"Manifest written to {MANIFEST_FILE}")
    print()
    print("NOW LISTEN to a few clips in clips\\ and check:")
    print("  - delivery is flat and functional, not acted")
    print("  - nothing is clipped at the start or end")
    print("  - callsigns say 'Alpha One', not spelled out oddly")
    print()
    print("Too dramatic? Lower EXAGGERATION and re-run with OVERWRITE = True.")
    print("Next: python step5_radio_fx.py clips\\<somefile>.wav test.wav --profile base")


if __name__ == "__main__":
    main()