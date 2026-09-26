"""
step3_prep.py — turn a phone recording into a clean reference WAV.

Takes almost any audio file (m4a, mp3, wav, ogg) and produces the mono
WAV that voice cloning wants: single channel, correct sample rate, silence
trimmed off the ends, level normalised.

Uses librosa, which you already have as a Chatterbox dependency.

    python step3_prep.py myvoice.m4a refs\\myvoice.wav

Or convert everything in a folder:

    python step3_prep.py recordings\\ refs\\

It also reports on the recording so you can judge whether it is usable
before you build 400 clips on top of it.
"""

import os
import sys

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 24000      # comfortable for cloning; downsampled internally anyway
TRIM_DB = 30           # silence threshold at the ends
TARGET_PEAK = 0.95


def prep(in_path, out_path):
    print(f"\n{os.path.basename(in_path)}")

    try:
        audio, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  could not read: {type(e).__name__}: {e}")
        print("  if this is an m4a, you may need ffmpeg installed")
        return False

    original_len = len(audio) / sr

    # Trim silence from the start and end only — not the middle, since
    # natural pauses between sentences are useful reference material.
    audio, _ = librosa.effects.trim(audio, top_db=TRIM_DB)
    trimmed_len = len(audio) / sr

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio * (TARGET_PEAK / peak)

    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, audio, sr, subtype="PCM_16")

    print(f"  {original_len:.1f}s -> {trimmed_len:.1f}s after trimming silence")
    print(f"  mono, {sr} Hz, 16-bit -> {out_path}")

    # --- Judgement on whether this is good enough to build on ---
    warnings = []

    if trimmed_len < 8:
        warnings.append(f"only {trimmed_len:.1f}s of speech — aim for 20-40s")
    elif trimmed_len > 90:
        warnings.append(f"{trimmed_len:.1f}s is longer than needed; 20-40s is plenty")

    if peak > 0.995:
        warnings.append("recording is clipped (peaked at maximum) — "
                        "re-record further from the mic")
    elif peak < 0.1:
        warnings.append("recording is very quiet — re-record closer to the mic")

    if rms < 0.02:
        warnings.append("very low average level; likely too far from the mic")

    # Rough noise estimate: level of the quietest 10% of short frames
    frame = int(sr * 0.05)
    if len(audio) > frame * 20:
        frames = audio[:len(audio) // frame * frame].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        floor = float(np.percentile(frame_rms, 10))
        if rms > 0 and floor / rms > 0.15:
            warnings.append("noticeable background noise — a quieter room "
                            "would improve the clone significantly")

    if warnings:
        print("  ATTENTION:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  looks good")

    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    if os.path.isdir(src):
        exts = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus")
        files = [f for f in sorted(os.listdir(src)) if f.lower().endswith(exts)]
        if not files:
            print(f"No audio files found in {src}")
            sys.exit(1)
        print(f"Converting {len(files)} files from {src}")
        for f in files:
            base = os.path.splitext(f)[0] + ".wav"
            prep(os.path.join(src, f), os.path.join(dst, base))
    else:
        prep(src, dst)

    print("\nDone. Point VOICE_REFS in step4_render_voices.py at these filenames.")


if __name__ == "__main__":
    main()