"""
step7_session_build.py — Phase 4. Turn your clip library into a live-sounding net.

Reads:  manifest_processed.json + processed\\*.wav
Writes: session_command.wav      one file per channel
        session_admin.wav
        session_mixed.wav        all channels together
        session_log.csv          timestamped log of every transmission
        session_log.json         same, machine readable

    python step7_session_build.py

---------------------------------------------------------------------------
THIS IS THE PHASE THAT DECIDES WHETHER IT SOUNDS REAL

A net is not a playlist. Five behaviours separate a convincing net from a
sequence of clips, and all five are modelled here:

1. REPLY LATENCY — 1 to 4 seconds, varying. Longer after "wait, out",
   longer when the station has to look something up. Evenly spaced
   transmissions read as fake instantly.

2. SILENCE — real nets are quiet for long stretches, then busy around an
   event, then quiet again. Continuous chatter is the single clearest
   giveaway. Target occupancy here is around 20-30% of airtime.

3. NET DISCIPLINE — one station transmits at a time. Occasional doubling
   (two stations keying together, both garbled) is a strong realism cue at
   roughly one occurrence in twenty exchanges. More than that sounds
   broken.

4. SCENARIO STATE — event threads play at their recorded offsets, so a
   report at T+0 gets a follow-up at T+4min and a resolution at T+10min,
   with unrelated routine traffic in between.

5. MULTIPLE NETS — a command net and an admin net, different tempo,
   different traffic.
---------------------------------------------------------------------------
"""

import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

# ===========================================================================
# CONFIGURATION
# ===========================================================================

MANIFEST = "manifest_processed.json"
FALLBACK_MANIFEST = "manifest.json"

SESSION_MINUTES = 10          # length of the session to build
SEED = 42                     # None for a different session every run

# Target proportion of airtime carrying traffic. Real routine nets are
# quiet. Above about 0.4 it starts sounding like a busy operations room.
TARGET_OCCUPANCY = 0.25

# --- Channels -------------------------------------------------------------
# An exchange is placed on the first channel that contains both its units.
# Anything unmatched goes on the first channel.

CHANNELS = {
    "command": ["ALPHA-1", "BRAVO-6", "CHARLIE-3", "DELTA-2", "ECHO-4", "GOLF-7"],
    "admin":   ["ALPHA-1", "FOXTROT-5", "HOTEL-9", "DELTA-2", "ECHO-4"],
}

# --- Timing ---------------------------------------------------------------

REPLY_GAP = (1.0, 4.0)        # seconds between transmissions in an exchange
THINKING_GAP = (5.0, 12.0)    # after "wait, out" or a question needing lookup

ACTIVE_GAP = (12.0, 45.0)     # between exchanges during a busy period
QUIET_GAP = (60.0, 210.0)     # between exchanges during a lull

ACTIVE_RUN = (2, 5)           # exchanges before switching to a lull
QUIET_RUN = (1, 2)            # lulls before becoming busy again

THREAD_PROBABILITY = 0.35     # chance of starting an event thread
DOUBLING_PROBABILITY = 0.05   # chance an exchange gets stepped on
DOUBLING_GAIN = 0.55          # level of the interfering station


# ===========================================================================

def load_manifest():
    path = MANIFEST if os.path.exists(MANIFEST) else FALLBACK_MANIFEST
    if not os.path.exists(path):
        print(f"Cannot find {MANIFEST} or {FALLBACK_MANIFEST}.")
        print("Run step4_render_voices.py then step6_process_clips.py first.")
        sys.exit(1)
    if path == FALLBACK_MANIFEST:
        print(f"NOTE: using {FALLBACK_MANIFEST} — this is unprocessed audio.")
        print("      Run step6_process_clips.py for radio-processed clips.")
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def group_exchanges(manifest):
    """Collect manifest rows into exchanges, ordered by position."""
    grouped = defaultdict(list)
    for row in manifest:
        grouped[row["exchange_id"]].append(row)

    exchanges = {}
    for ex_id, rows in grouped.items():
        rows.sort(key=lambda r: r["position"])
        exchanges[ex_id] = {
            "id": ex_id,
            "scenario": rows[0].get("scenario", "unknown"),
            "thread_id": rows[0].get("thread_id"),
            "thread_stage": rows[0].get("thread_stage"),
            "thread_gap": rows[0].get("thread_gap_seconds"),
            "units": sorted({r["callsign"] for r in rows}),
            "rows": rows,
        }
    return exchanges


def build_threads(exchanges):
    """Group threaded exchanges by thread_id, ordered by stage."""
    threads = defaultdict(list)
    for ex in exchanges.values():
        if ex["thread_id"]:
            threads[ex["thread_id"]].append(ex)
    for stages in threads.values():
        stages.sort(key=lambda e: e["thread_stage"] or 0)
    return dict(threads)


def pick_channel(units):
    for name, members in CHANNELS.items():
        if all(u in members for u in units):
            return name
    return next(iter(CHANNELS))


def exchange_duration(ex, rng):
    """How long this exchange occupies the air, including internal gaps."""
    total = 0.0
    rows = ex["rows"]
    for i, row in enumerate(rows):
        total += row.get("duration", 2.0)
        if i < len(rows) - 1:
            text = row.get("text", "").lower()
            if "wait" in text or "say again" in text:
                total += rng.uniform(*THINKING_GAP)
            else:
                total += rng.uniform(*REPLY_GAP)
    return total


# ===========================================================================
# SCHEDULING
# ===========================================================================

def schedule(exchanges, threads, rng):
    """Produce a list of scheduled transmissions across all channels."""

    session_end = SESSION_MINUTES * 60

    singles = [e for e in exchanges.values() if not e["thread_id"]]
    rng.shuffle(singles)
    thread_ids = list(threads)
    rng.shuffle(thread_ids)

    used_singles = set()
    used_threads = set()
    events = []                      # (start_time, channel, exchange)
    channel_busy = defaultdict(list)  # channel -> [(start, end)]

    def air_used(channel):
        return sum(e - s for s, e in channel_busy[channel])

    def place(ex, at, allow_overlap=False):
        """Place an exchange at a time, shifting later if the channel is busy."""
        channel = pick_channel(ex["units"])
        dur = exchange_duration(ex, rng)

        if not allow_overlap:
            for s, e in sorted(channel_busy[channel]):
                if at < e and at + dur > s:
                    at = e + rng.uniform(*REPLY_GAP)

        if at + dur > session_end:
            return None

        channel_busy[channel].append((at, at + dur))
        events.append((at, channel, ex))
        return at + dur

    # --- Walk the session, alternating busy periods and lulls ---
    t = rng.uniform(5, 30)
    mode = "active"
    remaining_in_mode = rng.randint(*ACTIVE_RUN)

    while t < session_end:
        # Stop adding traffic once we are at target occupancy
        occupancy = max(air_used(c) for c in CHANNELS) / max(t, 1)
        if occupancy > TARGET_OCCUPANCY * 1.3:
            t += rng.uniform(*QUIET_GAP)
            continue

        placed = False

        # Start an event thread?
        if (thread_ids and rng.random() < THREAD_PROBABILITY
                and len(used_threads) < len(thread_ids)):
            for tid in thread_ids:
                if tid in used_threads:
                    continue
                used_threads.add(tid)
                base = t
                for stage in threads[tid]:
                    gap = stage["thread_gap"] or 0
                    place(stage, base + gap)
                placed = True
                break

        # Otherwise a standalone exchange
        if not placed:
            for ex in singles:
                if ex["id"] in used_singles:
                    continue
                used_singles.add(ex["id"])

                # Occasionally have another station step on this one
                double = rng.random() < DOUBLING_PROBABILITY
                place(ex, t)

                if double:
                    for other in singles:
                        if other["id"] in used_singles:
                            continue
                        used_singles.add(other["id"])
                        place(other, t + rng.uniform(0.5, 2.0),
                              allow_overlap=True)
                        break
                placed = True
                break

        if not placed:
            break        # ran out of content

        if mode == "active":
            t += rng.uniform(*ACTIVE_GAP)
        else:
            t += rng.uniform(*QUIET_GAP)

        remaining_in_mode -= 1
        if remaining_in_mode <= 0:
            mode = "quiet" if mode == "active" else "active"
            remaining_in_mode = rng.randint(
                *(QUIET_RUN if mode == "quiet" else ACTIVE_RUN))

    events.sort(key=lambda e: e[0])
    return events


def expand(events, rng):
    """Turn scheduled exchanges into individual timed transmissions."""
    out = []
    for start, channel, ex in events:
        t = start
        rows = ex["rows"]
        for i, row in enumerate(rows):
            out.append({
                "time": round(t, 2),
                "channel": channel,
                "callsign": row["callsign"],
                "text": row.get("text", ""),
                "file": row["file"],
                "duration": row.get("duration", 0.0),
                "scenario": ex["scenario"],
                "thread_id": ex["thread_id"],
                "exchange_id": ex["id"],
            })
            t += row.get("duration", 2.0)
            if i < len(rows) - 1:
                low = row.get("text", "").lower()
                if "wait" in low or "say again" in low:
                    t += rng.uniform(*THINKING_GAP)
                else:
                    t += rng.uniform(*REPLY_GAP)
    out.sort(key=lambda x: x["time"])
    return out


# ===========================================================================
# MIXDOWN
# ===========================================================================

def mixdown(transmissions, sample_rate, length_seconds):
    """Render each channel to its own buffer, plus a combined mix."""
    n = int(length_seconds * sample_rate)
    buffers = {name: np.zeros(n, dtype=np.float64) for name in CHANNELS}
    cache = {}

    for tx in transmissions:
        path = tx["file"].replace("/", os.sep)
        if path not in cache:
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != sample_rate:
                    print(f"  sample rate mismatch in {path} "
                          f"({sr} vs {sample_rate}) — skipped")
                    cache[path] = None
                else:
                    cache[path] = audio
            except Exception as e:
                print(f"  could not read {path}: {e}")
                cache[path] = None

        audio = cache[path]
        if audio is None:
            continue

        start = int(tx["time"] * sample_rate)
        end = min(start + len(audio), n)
        if start >= n:
            continue

        buf = buffers[tx["channel"]]
        segment = audio[:end - start]
        # Overlapping transmissions sum, which is what doubling sounds like
        buf[start:end] += segment

    for name, buf in buffers.items():
        peak = np.max(np.abs(buf))
        if peak > 0.95:
            buffers[name] = buf * (0.95 / peak)

    mixed = np.zeros(n, dtype=np.float64)
    for buf in buffers.values():
        mixed += buf
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed *= 0.95 / peak

    return buffers, mixed


# ===========================================================================

def main():
    manifest, source = load_manifest()
    print(f"Read {len(manifest)} clips from {source}")

    rates = {row.get("sample_rate") for row in manifest if row.get("sample_rate")}
    if len(rates) > 1:
        print(f"ERROR: mixed sample rates in manifest: {sorted(rates)}")
        print("All clips must share a sample rate. Re-render consistently.")
        sys.exit(1)
    sample_rate = rates.pop() if rates else 24000

    exchanges = group_exchanges(manifest)
    threads = build_threads(exchanges)
    print(f"{len(exchanges)} exchanges, {len(threads)} event threads")
    print(f"Channels: {', '.join(CHANNELS)}")
    print(f"Building a {SESSION_MINUTES}-minute session at {sample_rate} Hz")
    print()

    rng = random.Random(SEED)

    events = schedule(exchanges, threads, rng)
    transmissions = expand(events, rng)
    print(f"Scheduled {len(events)} exchanges, {len(transmissions)} transmissions")

    length = SESSION_MINUTES * 60
    buffers, mixed = mixdown(transmissions, sample_rate, length)

    for name, buf in buffers.items():
        out = f"session_{name}.wav"
        sf.write(out, buf, sample_rate)
        airtime = float(np.sum(np.abs(buf) > 1e-4)) / sample_rate
        print(f"  {out:24s} {airtime/60:5.1f} min of traffic "
              f"({airtime/length*100:4.1f}% occupancy)")

    sf.write("session_mixed.wav", mixed, sample_rate)
    print(f"  session_mixed.wav")

    # --- Logs ---
    with open("session_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "mm:ss", "channel", "callsign", "text",
                    "scenario", "thread_id"])
        for tx in transmissions:
            mm, ss = divmod(int(tx["time"]), 60)
            w.writerow([tx["time"], f"{mm:02d}:{ss:02d}", tx["channel"],
                        tx["callsign"], tx["text"], tx["scenario"],
                        tx["thread_id"] or ""])

    with open("session_log.json", "w", encoding="utf-8") as f:
        json.dump(transmissions, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    print()
    print("=" * 62)
    per_channel = defaultdict(int)
    for tx in transmissions:
        per_channel[tx["channel"]] += 1
    print("Transmissions per channel:")
    for name in CHANNELS:
        print(f"  {name:10s} {per_channel[name]:4d}")

    gaps = []
    by_channel = defaultdict(list)
    for tx in transmissions:
        by_channel[tx["channel"]].append(tx)
    for rows in by_channel.values():
        for a, b in zip(rows, rows[1:]):
            gap = b["time"] - (a["time"] + a["duration"])
            if gap > 0:
                gaps.append(gap)
    if gaps:
        gaps.sort()
        print()
        print(f"Gaps between transmissions: median {gaps[len(gaps)//2]:.1f}s, "
              f"longest {gaps[-1]:.0f}s")

    print()
    print("Logs written: session_log.csv, session_log.json")
    print()
    print("LISTEN TO session_command.wav ALL THE WAY THROUGH.")
    print()
    print("What to judge:")
    print("  - Do the quiet stretches feel right, or too long or too short?")
    print("  - Does traffic cluster around events, or drone evenly?")
    print("  - Do replies come back at believable delays?")
    print("  - Can you follow an event thread developing across the session?")
    print()
    print("Adjust TARGET_OCCUPANCY, ACTIVE_GAP and QUIET_GAP, then re-run.")
    print("Change SEED for a completely different session from the same pool.")


if __name__ == "__main__":
    main()