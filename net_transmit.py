"""
net_transmit.py — run a session across two radios as a live conversation.

THE PROBLEM THIS SOLVES

Sending radio_A.wav then radio_B.wav gives you two monologues: everything
Alpha says, then everything Bravo says. A third radio listening in hears
neither a conversation nor anything that sounds like a net.

This walks the session timeline instead. At each moment it keys whichever
radio is supposed to be talking, sends that one transmission, drops the
carrier, and waits for the next. A monitor on the same frequency hears

    Alpha ... Bravo ... Alpha ... Bravo ...

which is what a net actually sounds like.

    python net_transmit.py --plan output/transmit_plan_20260812_143022.json
    python net_transmit.py --plan <file> --dry-run     no radios, timing only
    python net_transmit.py --plan <file> --monitor rx.wav

---------------------------------------------------------------------------
BEFORE TRANSMITTING

This puts a signal on an antenna. Confirm all of the following, because the
software cannot:

  - the frequency is assigned to your organisation for exercise traffic
  - somebody is controlling the exercise and can stop it immediately
  - the exercise marking is in the audio, so anyone who hears it knows

The last one this file does check — it refuses to run on a plan whose text
carries no marking unless you pass --unmarked, and says why.

Keep the power low. TX_GAIN of -60 or below is plenty for a room, and a
Pluto at its default will be heard across a building.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("soundfile is needed:  pip install soundfile")
    sys.exit(1)

try:
    import pluto_link
except ImportError:
    print("pluto_link.py must be in this folder")
    sys.exit(1)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

RADIO_URIS = {
    "A": "ip:192.168.2.1",
    "B": "ip:192.168.3.1",
}

CARRIER_MHZ = 488.0        # your assigned frequency
TX_GAIN = -60              # dB. Low. Raise only if the link genuinely fails.
RX_GAIN = 40

GUARD_BEFORE = 0.15        # carrier up this long before speech
GUARD_AFTER = 0.25         # and held this long after, for the squelch tail
MAX_GAP = 6.0              # cap on dead air, so a bench run is watchable


# ===========================================================================
# THE PLAN
# ===========================================================================

def load_plan(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tx = data.get("transmissions") or []
    if not tx:
        raise ValueError("the plan has no transmissions in it")

    out = []
    for t in tx:
        f = t.get("file")
        if not f or not os.path.exists(f.replace("/", os.sep)):
            print(f"  missing clip, skipping: {f}")
            continue
        out.append({
            "t": float(t.get("t", 0)),
            "radio": (t.get("radio") or "A").upper(),
            "cs": t.get("cs", ""),
            "text": t.get("text", ""),
            "file": f.replace("/", os.sep),
            "dur": float(t.get("dur", 0)),
        })

    out.sort(key=lambda x: x["t"])
    return out, data.get("duration_s", 0)


def check_marking(plan):
    """Is this traffic identifiable as an exercise?

    Not a formality. The audio is realistic contact and casualty traffic; a
    station that picks it up mid-session and cannot tell it is practice is
    the whole reason marking exists.
    """
    if not plan:
        return False, 0
    marked = sum(1 for t in plan if "exercise" in t["text"].lower())
    return marked > len(plan) * 0.5, marked


# ===========================================================================
# THE RADIOS
# ===========================================================================

class Radio:
    """One transmitter, opened once and reused.

    Opening a Pluto takes a second or so. Doing that per transmission would
    put a gap in the middle of the conversation that has nothing to do with
    the schedule.
    """

    def __init__(self, key, uri, carrier_hz, dry=False):
        self.key = key
        self.uri = uri
        self.dry = dry
        self.sdr = None
        self.sent = 0

        if dry:
            return

        pluto_link.TX_GAIN = TX_GAIN
        self.sdr = pluto_link.open_tx(uri, carrier=int(carrier_hz))
        print(f"  radio {key} ready on {uri}  gain {TX_GAIN} dB")

    def send(self, audio):
        if self.dry:
            time.sleep(len(audio) / pluto_link.AUDIO_RATE)
            self.sent += 1
            return
        iq = pluto_link.nbfm_modulate(audio)
        pluto_link.transmit(self.sdr, iq)
        self.sent += 1

    def close(self):
        if self.sdr is not None:
            del self.sdr
            self.sdr = None


class Monitor:
    """A third receiver, recording what the other two put on the air.

    This is the check that matters: if the monitor recording sounds like a
    conversation, the interleaving worked. If it sounds like one long
    monologue followed by another, it did not.
    """

    def __init__(self, uri, carrier_hz, out_path):
        self.out_path = out_path
        self.chunks = []
        self.running = False
        pluto_link.RX_GAIN = RX_GAIN
        self.sdr = pluto_link.open_rx(uri, carrier=int(carrier_hz))
        self.thread = None

    def start(self):
        self.running = True

        def loop():
            while self.running:
                try:
                    self.chunks.append(self.sdr.rx())
                except Exception:
                    break

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)

        if not self.chunks:
            print("  monitor recorded nothing")
            return None

        iq = np.concatenate(self.chunks)
        audio = pluto_link.nbfm_demodulate(iq)
        sf.write(self.out_path, audio, pluto_link.AUDIO_RATE)

        level = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        print(f"  monitor wrote {self.out_path}  "
              f"{len(audio)/pluto_link.AUDIO_RATE:.1f}s  level {level:.0f}")
        del self.sdr
        return self.out_path


# ===========================================================================
# RUNNING THE NET
# ===========================================================================

def run(plan, duration, carrier_mhz, dry=False, monitor_uri=None,
        monitor_out=None, max_gap=MAX_GAP, speed=1.0):
    """Walk the timeline, keying whichever radio is due to talk."""

    carrier_hz = carrier_mhz * 1e6
    used = sorted({t["radio"] for t in plan})

    print()
    print(f"  {len(plan)} transmissions over {duration:.0f}s")
    print(f"  radios in use: {', '.join(used)}")
    print(f"  carrier {carrier_mhz} MHz")
    if dry:
        print("  DRY RUN — no hardware, timing only")
    print()

    radios = {}
    for key in used:
        uri = RADIO_URIS.get(key)
        if uri is None:
            print(f"  no address configured for radio {key}, skipping it")
            continue
        try:
            radios[key] = Radio(key, uri, carrier_hz, dry=dry)
        except Exception as e:
            print(f"  could not open radio {key}: {type(e).__name__}: {e}")
            return False

    mon = None
    if monitor_uri and not dry:
        try:
            mon = Monitor(monitor_uri, carrier_hz, monitor_out or "monitor.wav")
            mon.start()
            print(f"  monitor listening on {monitor_uri}")
            print()
        except Exception as e:
            print(f"  monitor unavailable: {type(e).__name__}: {e}")

    # Preload the audio. Reading from disk mid-conversation would introduce
    # delays that have nothing to do with the schedule.
    print("  loading clips...")
    for t in plan:
        t["audio"] = pluto_link.load_audio(t["file"])

    print()
    print("  " + "-" * 56)
    started = time.time()
    skipped = 0.0      # dead air we chose not to wait out
    last_end = 0.0

    try:
        for i, t in enumerate(plan, 1):
            key = t["radio"]
            radio = radios.get(key)
            if radio is None:
                continue

            # Wait until this transmission is due, capped so a long silence
            # does not make a bench run unwatchable.
            #
            # Skipped time is tracked separately rather than by shifting the
            # start, because shifting accumulates and sends the reported
            # clock backwards.
            target = t["t"] / speed
            elapsed = (time.time() - started) + skipped
            gap = target - elapsed
            if gap > 0:
                wait = min(gap, max_gap)
                if wait > 0.05:
                    print(f"        ... {wait:.1f}s"
                          + (f"  (of {gap:.1f}s)" if gap > wait + 0.1 else ""))
                time.sleep(wait)
                skipped += (gap - wait)

            at = (time.time() - started) + skipped
            print(f"  {at:7.1f}s  RADIO {key}  {t['cs']:10s} "
                  f"{t['dur']:.1f}s  {t['text'][:44]}")

            # Carrier up, speech, carrier down. The guard either side is
            # what stops the first syllable being clipped and gives the
            # squelch tail somewhere to live.
            if GUARD_BEFORE > 0:
                time.sleep(GUARD_BEFORE)

            radio.send(t["audio"])

            if GUARD_AFTER > 0:
                time.sleep(GUARD_AFTER)

            last_end = (time.time() - started) + skipped

    except KeyboardInterrupt:
        print("\n  stopped by hand")

    finally:
        print("  " + "-" * 56)
        for r in radios.values():
            r.close()
        if mon:
            time.sleep(0.5)
            mon.stop()

    print()
    for key in sorted(radios):
        print(f"  radio {key} sent {radios[key].sent} transmissions")
    real = time.time() - started
    print(f"  session time {last_end:.0f}s"
          + (f"  (took {real:.0f}s, {skipped:.0f}s of silence skipped)"
             if skipped > 1 else ""))
    return True


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Run a session across two radios as a live conversation.")
    ap.add_argument("--plan", required=True,
                    help="transmit_plan_*.json from a session build")
    ap.add_argument("--carrier", type=float, default=CARRIER_MHZ,
                    help=f"MHz (default {CARRIER_MHZ})")
    ap.add_argument("--dry-run", action="store_true",
                    help="no hardware — check the ordering and timing first")
    ap.add_argument("--monitor", metavar="WAV",
                    help="record the air with a third receiver")
    ap.add_argument("--monitor-uri", default=None,
                    help="address of the monitoring radio")
    ap.add_argument("--max-gap", type=float, default=MAX_GAP,
                    help="longest silence to actually wait out")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="compress the timeline; 2 runs it twice as fast")
    ap.add_argument("--gain", type=int, default=None,
                    help="transmit gain in dB, lower is quieter (default -60)")
    ap.add_argument("--unmarked", action="store_true",
                    help="allow traffic with no exercise marking")
    args = ap.parse_args()

    global TX_GAIN
    if args.gain is not None:
        TX_GAIN = args.gain

    print()
    print("=" * 62)
    print("  TWO-RADIO NET")
    print("=" * 62)

    if not os.path.exists(args.plan):
        print(f"\n  cannot find {args.plan}")
        print("  Build a session in the console first — it writes one of these")
        print("  into your output folder.")
        sys.exit(1)

    plan, duration = load_plan(args.plan)

    marked, count = check_marking(plan)
    print()
    if marked:
        print(f"  exercise marking: present on {count} of {len(plan)} transmissions")
    else:
        print(f"  exercise marking: MISSING — only {count} of {len(plan)} "
              f"transmissions say this is an exercise")
        if not args.dry_run and not args.unmarked:
            print()
            print("  Not transmitting. This audio is realistic operational")
            print("  traffic, and anyone who hears it has no way of knowing")
            print("  it is a simulation.")
            print()
            print("  Rebuild with exercise marking turned on in Session")
            print("  settings, or pass --unmarked if you are certain nothing")
            print("  outside your setup can receive it.")
            print()
            sys.exit(1)

    by_radio = {}
    for t in plan:
        by_radio[t["radio"]] = by_radio.get(t["radio"], 0) + 1
    print(f"  transmissions per radio: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_radio.items())))

    # Is this actually going to sound like a conversation? Count how often
    # the talking radio changes.
    swaps = sum(1 for a, b in zip(plan, plan[1:]) if a["radio"] != b["radio"])
    print(f"  hand-overs between radios: {swaps}")
    if swaps < 2 and len(plan) > 3:
        print()
        print("  Almost every transmission is from the same radio, so this")
        print("  will not sound like a conversation. Check that the stations")
        print("  are split across both radios in the session build.")

    ok = run(plan, duration, args.carrier,
             dry=args.dry_run,
             monitor_uri=args.monitor_uri,
             monitor_out=args.monitor,
             max_gap=args.max_gap,
             speed=args.speed)

    print()
    if ok and not args.dry_run:
        print("  Listen to the monitor recording. If it sounds like two")
        print("  stations talking to each other, the interleaving worked.")
        print("  If it sounds like one long monologue then another, it did not.")
    print()


if __name__ == "__main__":
    main()
