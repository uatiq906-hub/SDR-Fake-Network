"""
step3b_import_recorded.py — bring peacetime recordings into the pipeline.

Messages can now come from two places:

    RECORDED    Real operators read the script aloud in peacetime. The
                recording IS the transmission. No synthesis involved, so
                these skip step 4 entirely and go straight to radio
                processing. Best quality by a wide margin.

    GENERATED   The model writes them, step 4 speaks them in cloned voices.
                Unlimited variety, no recording session needed.

This script takes recordings captured in the web console and folds them into
the same manifest the rest of the pipeline already understands, so nothing
downstream needs to know where a clip came from.

WHERE THE FILES COME FROM

The console writes session_config.json and saves each recorded line as a
separate audio file. Put them here:

    recorded\\
        session_config.json
        R1a2b3c_01_ALPHA-1.webm
        R1a2b3c_02_BRAVO-6.webm
        ...

Then:

    python step3b_import_recorded.py

It converts each file to WAV at the project sample rate, writes them to
clips\\, and merges them into manifest.json with source="recorded".

WHY RECORD ANYTHING AT ALL

Record the traffic that repeats constantly — radio checks, position reports,
nothing-to-report sitreps. Those three make up a large share of any session,
so a handful of real takes lifts the whole thing noticeably. Generate the
long tail, where variety matters more than perfection.
"""

import json
import os
import subprocess
import sys

CONFIG_FILE = os.path.join("recorded", "session_config.json")
RECORDED_DIR = "recorded"
CLIPS_DIR = "clips"
MANIFEST = "manifest.json"

TARGET_SR = 24000       # must match what step4_render_voices.py produces


def check_ffmpeg():
    """Browser recordings are webm/opus; ffmpeg converts them to WAV."""
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def convert(src, dst, sample_rate):
    """Convert to mono WAV at the project sample rate."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        dst,
    ], check=True)


def duration(path):
    import wave
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"Cannot find {CONFIG_FILE}")
        print()
        print("Export the settings from the web console (step 4, 'Save the")
        print("settings') and put the file in a folder called 'recorded'")
        print("alongside the audio files.")
        sys.exit(1)

    if not check_ffmpeg():
        print("ffmpeg is not installed, and browser recordings need it.")
        print("Install it with:  winget install ffmpeg")
        print("Then close and reopen PowerShell.")
        sys.exit(1)

    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = json.load(f)

    takes = cfg.get("recorded_takes", [])
    if not takes:
        print("No recorded takes in the config. Nothing to import.")
        print("Record some messages in step 2 of the console first.")
        sys.exit(0)

    os.makedirs(CLIPS_DIR, exist_ok=True)

    # Existing manifest, if the generated side has already been rendered
    manifest = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)
        # Drop any previous import so re-running is safe
        before = len(manifest)
        manifest = [m for m in manifest if m.get("source") != "recorded"]
        if before != len(manifest):
            print(f"Replacing {before - len(manifest)} previously imported clips")

    print(f"Importing {len(takes)} recorded exchanges")
    print()

    imported = 0
    missing = 0
    next_id = max([m.get("exchange_id", 0) for m in manifest] or [0]) + 1

    for take in takes:
        ex_id = next_id
        next_id += 1
        situation = take.get("situation", "recorded")

        for pos, line in enumerate(take.get("lines", []), 1):
            callsign = str(line.get("callsign", "UNKNOWN")).upper()

            # The console names files <take_id>_<NN>_<CALLSIGN>.<ext>
            stem = f"{take['id']}_{pos:02d}_{callsign}"
            src = None
            for ext in (".webm", ".ogg", ".wav", ".mp4", ".m4a"):
                candidate = os.path.join(RECORDED_DIR, stem + ext)
                if os.path.exists(candidate):
                    src = candidate
                    break

            if src is None:
                print(f"  missing audio for {stem}")
                missing += 1
                continue

            out_name = f"{ex_id:04d}_{situation}_{pos:02d}_{callsign}.wav"
            out_path = os.path.join(CLIPS_DIR, out_name)

            try:
                convert(src, out_path, TARGET_SR)
            except subprocess.CalledProcessError:
                print(f"  could not convert {src}")
                missing += 1
                continue

            secs = duration(out_path)

            manifest.append({
                "file": out_path.replace("\\", "/"),
                "exchange_id": ex_id,
                "scenario": situation,
                "thread_id": None,
                "thread_stage": None,
                "thread_gap_seconds": None,
                "position": pos,
                "is_last": pos == len(take["lines"]),
                "callsign": callsign,
                "text": line.get("text", ""),
                "source": "recorded",       # <- the flag everything downstream reads
                "reference": None,          # no voice cloning involved
                "duration": round(secs, 2),
                "sample_rate": TARGET_SR,
                "conditions": {},
            })
            imported += 1

    # Anything already in the manifest without a source is generated
    for m in manifest:
        m.setdefault("source", "generated")

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    print()
    print("=" * 58)
    print(f"Imported {imported} clips   Missing {missing}")

    rec = [m for m in manifest if m["source"] == "recorded"]
    gen = [m for m in manifest if m["source"] == "generated"]
    print()
    print(f"Manifest now holds {len(manifest)} clips:")
    print(f"  recorded   {len(rec):5d}  "
          f"({sum(m['duration'] for m in rec)/60:.1f} min)")
    print(f"  generated  {len(gen):5d}  "
          f"({sum(m['duration'] for m in gen)/60:.1f} min)")

    if rec:
        print()
        print("Recorded situations:")
        from collections import Counter
        for name, count in Counter(m["scenario"] for m in rec).most_common():
            print(f"  {count:4d}  {name}")

    print()
    print("Next: python step6_process_clips.py")
    print()
    print("Recorded clips get the same radio processing as generated ones —")
    print("that part is identical either way. Only the speech step differs.")


if __name__ == "__main__":
    main()
