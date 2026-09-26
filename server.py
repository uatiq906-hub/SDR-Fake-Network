"""
server.py — run the web console against your real pipeline.

WHAT THIS IS FOR

The browser on its own can only preview. It cannot clone voices and it
cannot write WAV files of synthesised speech — those are hard browser
limits, not shortcuts.

This server closes that gap. It serves the console, takes whatever you
record or compose in it, and runs the actual work on this machine using
the scripts you already have:

    step5_radio_fx.py   the DSP chain — bandpass, compression, noise,
                        squelch. Imported directly, so the console and the
                        command line produce identical audio.

    Chatterbox          voice cloning for computer-written messages, using
                        the reference recordings you make in the console.

Everything runs locally on your CPU. Nothing is uploaded anywhere.

---------------------------------------------------------------------------
RUNNING IT

    python server.py

That is the whole thing. It serves the console itself and opens your
browser, so there is no need for 'python -m http.server' — in fact having
one of those running on the same port is what stops this working, since it
hands out the files while none of the API exists.

If port 8000 is taken it quietly moves to 8001, 8002 and so on, and tells
you which address to use.

Put this file in your project folder, alongside step5_radio_fx.py and the
webui\\ folder. It needs:

    numpy, scipy, soundfile        (you already have these)
    ffmpeg on PATH                 for converting browser recordings
    torch, chatterbox-tts          only for computer-written messages

If Chatterbox is missing the recorded side still works fully — the server
says so at startup rather than failing later.
---------------------------------------------------------------------------
"""

import json
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Every path below is relative — REFS_DIR, OUT_DIR, WEBUI_DIR, and so on.
# Relative paths are resolved against the current working directory, which
# is wherever the terminal happened to be *before* "python server.py" was
# typed — not necessarily the folder this file lives in. Run from the right
# folder, that's a no-op; run from anywhere else (a shortcut, a different
# starting directory, "python C:\...\server.py" from elsewhere) and every
# relative path below silently points at the wrong place — usually an empty
# or nonexistent one, which looks exactly like "no files" rather than
# "wrong folder". Anchoring to this file's own location removes that
# failure mode entirely.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8000
WEBUI_DIR = "webui"

REFS_DIR = "refs"            # reference voices recorded in the console
RECORDED_DIR = "recorded"    # whole-take message recordings
CLIPS_DIR = "clips"          # synthesised speech, before radio processing
OUT_DIR = "output"           # finished session files
STATE_FILE = "session_state.json"   # what you were working on

SAMPLE_RATE = 24000          # everything is resampled to this


# ===========================================================================
# THE DSP CHAIN — imported from your own script, not reimplemented
# ===========================================================================

try:
    from step5_radio_fx import PROFILES, process as radio_process
    HAVE_FX = True
except ImportError:
    try:
        from radio_fx import PROFILES, process as radio_process
        HAVE_FX = True
    except ImportError:
        HAVE_FX = False
        PROFILES = {}
        radio_process = None


# The SDR link. Optional — everything else works without it.
try:
    import pluto_link
    HAVE_PLUTO = True
except Exception:
    pluto_link = None
    HAVE_PLUTO = False


def have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except Exception:
        return False


HAVE_FFMPEG = have_ffmpeg()


# ---------------------------------------------------------------------------
# Chatterbox is loaded lazily. It pulls in torch and takes a while, so there
# is no reason to pay that cost unless computer-written messages are asked
# for.
# ---------------------------------------------------------------------------

_tts = None
_tts_error = None


def get_tts():
    global _tts, _tts_error
    if _tts is not None or _tts_error is not None:
        return _tts

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"loading Chatterbox on {device} — first time takes a while")

        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            _tts = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
            log("Chatterbox Nano ready")
        except Exception:
            from chatterbox.tts import ChatterboxTTS
            _tts = ChatterboxTTS.from_pretrained(device=device)
            log("Chatterbox (standard) ready")
    except Exception as e:
        _tts_error = f"{type(e).__name__}: {e}"
        log(f"Chatterbox unavailable — {_tts_error}")

    return _tts


# ===========================================================================
# THE RADIOS
# --------------------------------------------------------------------------
# A board is found two ways: by trying addresses it commonly answers on,
# and by scanning USB contexts directly. USB scanning matters on Windows
# in particular — the Pluto's virtual network adapter needs a driver to be
# reachable by IP at all, and a board that hasn't gotten that driver will
# still show up over USB even though every ip:192.168.x.1 attempt times
# out. A board that answers under both is folded into a single entry by
# comparing its hardware serial number.
#
# Every address that was tried but failed keeps a short reason, so "no
# radios found" can say *why* instead of leaving that to be guessed at.
#
# Probing hardware takes a second or so, and the console asks often, so the
# answer is cached. Force a fresh look with ?refresh=1.
# ===========================================================================

_radios = {"checked": 0, "found": [], "problem": None, "attempts": []}
RADIO_CACHE_S = 20

# Addresses to try directly, in addition to whatever a USB scan turns up.
# Pluto ships on 192.168.2.1; a second board has to be renumbered onto its
# own subnet, and people pick various things, so several likely addresses
# are tried rather than assuming exactly two boards.
RADIO_URIS = [
    "ip:192.168.2.1",
    "ip:192.168.3.1",
    "ip:192.168.4.1",
    "ip:192.168.5.1",
    "ip:pluto.local",
]

# With one board and a cable between its own TX and RX ports, the same radio
# is both ends of the link. Everything except the hop between two separate
# boards is still exercised, which is most of what there is to verify.
ALLOW_LOOPBACK = True


def find_radios(force=False):
    """Probe every likely address plus a USB scan, and return the boards
    that answer.

    Only boards that actually respond are returned, numbered in the order
    found. Listing addresses that answered nothing was misleading — an
    operator seeing "SDR 1: not found" cannot tell whether a board is broken
    or was never there. Instead, if nothing answers at all, the reasons each
    address failed for are kept so that can be surfaced instead of a bare
    "nothing found".

    Two addresses can resolve to the same physical board (192.168.2.1 and
    pluto.local, for instance, or an IP address and a USB URI), so
    duplicates are collapsed by comparing the device's own hardware serial
    number rather than the address used to reach it.
    """
    now = time.time()
    if not force and now - _radios["checked"] < RADIO_CACHE_S:
        return _radios["found"]

    found = []
    problem = None
    attempts = []   # [(uri, reason)] for every candidate that failed

    if not HAVE_PLUTO:
        problem = "pluto_link.py is not in this folder"
        _radios.update(found=[], problem=problem, attempts=[], checked=now)
        return []

    try:
        import adi
    except ImportError:
        problem = "pyadi-iio is not installed"
        _radios.update(found=[], problem=problem, attempts=[], checked=now)
        return []

    # Candidates: the known addresses, plus whatever a USB scan finds. USB
    # scanning catches a board that answers over USB but has no working IP
    # route yet (typically a missing driver for Pluto's virtual network
    # adapter on Windows) — the two discovery paths are independent, so a
    # board only needs one of them to work to show up here.
    candidates = list(RADIO_URIS)
    try:
        import iio
        for uri in iio.scan_contexts():
            if uri not in candidates:
                candidates.append(uri)
    except ImportError:
        attempts.append(("usb scan", "iio module not installed — pyadi-iio "
                          "alone does not include it"))
    except Exception as e:
        attempts.append(("usb scan", f"{type(e).__name__}: {e}"))

    seen = {}
    for uri in candidates:
        dev = None
        try:
            dev = adi.Pluto(uri)
            rate = float(getattr(dev, "sample_rate", 0) or 0)
            rx_lo = float(getattr(dev, "rx_lo", 0) or 0)
            ctx_attrs = {}
            try:
                ctx_attrs = dev.ctx.attrs
            except Exception:
                pass
            serial = ctx_attrs.get("hw_serial")
            model = ctx_attrs.get("hw_model") or "ADALM-PLUTO SDR"

            key = serial or (round(rate), round(rx_lo))
            if key in seen:
                # Same board reached by a second address
                seen[key]["also"].append(uri)
                continue

            entry = {
                "n": len(found) + 1,
                "uri": uri,
                "also": [],
                "online": True,
                "serial": serial,
                "model": model,
                "sample_rate_mhz": round(rate / 1e6, 2),
                "rx_mhz": round(rx_lo / 1e6, 1),
                "detail": f"{rate/1e6:.1f} MS/s, rx {rx_lo/1e6:.0f} MHz",
            }
            entry["label"] = f"{model} {entry['n']}"
            seen[key] = entry
            found.append(entry)
        except Exception as e:
            attempts.append((uri, f"{type(e).__name__}: {e}"))
        finally:
            if dev is not None:
                try:
                    del dev
                except Exception:
                    pass

    if not found and attempts:
        # Nothing answered anywhere — say what each attempt actually hit
        # rather than a bare "no radios", since "timed out" and "no such
        # device" and "driver missing" all need different fixes.
        problem = "; ".join(f"{u}: {r}" for u, r in attempts)

    _radios["found"] = found
    _radios["problem"] = problem
    _radios["attempts"] = attempts
    _radios["checked"] = now
    return found


# ===========================================================================
# JOBS
# Rendering takes minutes on CPU, so work runs in a thread and the browser
# polls for progress rather than waiting on a request that would time out.
# ===========================================================================

jobs = {}
jobs_lock = threading.Lock()


def log(msg):
    print(f"  {msg}", flush=True)


def iso_ms(t=None):
    """Timestamp with millisecond precision, not just whole seconds.

    Whole-second timestamps make TX and RX for the same message look
    identical in a log even though they're running concurrently in two
    threads — the actual gap between them is a few milliseconds, not
    zero, and this is precise enough to actually show that overlap
    rather than hide it.
    """
    if t is None:
        t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + f".{ms:03d}"


def new_job():
    jid = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[jid] = {"state": "running", "step": "starting",
                     "done": 0, "total": 0, "files": [], "error": None,
                     "started": time.time()}
    return jid


def set_job(jid, **kw):
    with jobs_lock:
        if jid in jobs:
            jobs[jid].update(kw)


def get_job(jid):
    with jobs_lock:
        return dict(jobs.get(jid, {"state": "unknown"}))


def cancel_job(jid):
    """Ask a running job to stop. It's cooperative — the job's own loop
    has to actually check job_cancelled() between steps for this to do
    anything, same as any other stop-a-background-thread scheme without
    forcibly killing it mid-transmit."""
    with jobs_lock:
        if jid in jobs:
            jobs[jid]["cancel"] = True


def job_cancelled(jid):
    with jobs_lock:
        return bool(jobs.get(jid, {}).get("cancel"))


# ===========================================================================
# AUDIO HELPERS
# ===========================================================================

def to_wav(src, dst, sample_rate=SAMPLE_RATE):
    """Browser recordings arrive as webm/opus. ffmpeg makes them mono WAV."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src, "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", dst,
    ], check=True)


def read_mono(path, sample_rate=SAMPLE_RATE):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        # Linear resample. Good enough — everything downstream is band-limited
        # to 3.4 kHz anyway, so resampling artefacts are filtered out.
        n = int(len(audio) * sample_rate / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n),
            np.arange(len(audio)), audio)
    return audio.astype(np.float64)


def apply_radio(audio, profile_name, seed=None):
    """Run your own DSP chain, so console output and command-line output
    are byte-for-byte the same treatment."""
    if not HAVE_FX:
        return audio
    prof = PROFILES.get(profile_name) or PROFILES.get("handheld")
    return radio_process(audio, SAMPLE_RATE, prof, seed=seed)


def mix_into(buf, audio, at_seconds):
    start = int(at_seconds * SAMPLE_RATE)
    end = min(start + len(audio), len(buf))
    if start >= len(buf):
        return
    buf[start:end] += audio[:end - start]


def normalise(x, peak=0.92):
    m = np.max(np.abs(x)) if x.size else 0
    return x * (peak / m) if m > 1e-9 else x


# ===========================================================================
# RENDERING
# ===========================================================================

def render_session(jid, cfg):
    """Build the session WAVs from a schedule the console worked out.

    The browser already decided what happens when — this only has to
    produce the audio and place it. That keeps the timing model in one
    place rather than duplicated on both sides.
    """
    try:
        tx = cfg.get("transmissions", [])
        duration = float(cfg.get("duration_s", 600))
        channels = cfg.get("channels", {"COMMAND": [], "ADMIN": []})
        assignment = cfg.get("voice_assignment", {})
        profiles = cfg.get("profiles", {})

        if not tx:
            raise ValueError("nothing to render — the session is empty")

        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(CLIPS_DIR, exist_ok=True)

        set_job(jid, total=len(tx), step="preparing")

        n = int(duration * SAMPLE_RATE)
        buffers = {ch: np.zeros(n, dtype=np.float64) for ch in channels}
        if not buffers:
            buffers = {"COMMAND": np.zeros(n, dtype=np.float64)}

        # Two more buffers, split by which end of the link each station sits
        # on. Radios are physical; stations are callsigns, and there are more
        # of them than there are radios. This is what makes radio_A.wav and
        # radio_B.wav — one file per transmitter, not per channel.
        radio_of = cfg.get("radio_assignment") or {}
        radio_buf = {"A": np.zeros(n, dtype=np.float64),
                     "B": np.zeros(n, dtype=np.float64)}

        marking = cfg.get("exercise_marking", "both")

        # Slot periodic announcements into the timeline before rendering, so
        # they are spoken and placed like any other transmission.
        if marking != "off" and NET_ANNOUNCEMENT_EVERY > 0 and tx:
            base = (channels.get("COMMAND") or ["ALPHA-1"])[0]
            last_at = -NET_ANNOUNCEMENT_EVERY
            extra = []
            for t in tx:
                at = float(t.get("t", 0))
                if at - last_at >= NET_ANNOUNCEMENT_EVERY:
                    extra.append({
                        "t": max(0.0, at - 3.0),
                        "ch": t.get("ch", "COMMAND"),
                        "cs": base,
                        "text": net_announcement(base),
                        "src": "generated",
                        "_announcement": True,
                    })
                    last_at = at
            if extra:
                tx = sorted(tx + extra, key=lambda x: float(x.get("t", 0)))
                log(f"added {len(extra)} exercise announcements")

        need_tts = any(t.get("src") == "generated" for t in tx)
        model = None
        if need_tts:
            set_job(jid, step="loading the voice model")
            model = get_tts()
            if model is None:
                raise RuntimeError(
                    "Chatterbox is not installed, so computer-written "
                    "messages cannot be spoken. Install it with "
                    "'pip install chatterbox-tts', or use recorded messages.")

        cache = {}

        for i, t in enumerate(tx, 1):
            cs = t.get("cs", "")
            ch = t.get("ch", "COMMAND")
            if ch not in buffers:
                ch = list(buffers)[0]

            prof = profiles.get(cs, t.get("profile", "handheld"))
            audio = None

            if t.get("src") == "recorded":
                rid = t.get("recording_id")
                path = os.path.join(RECORDED_DIR, rid + ".wav") if rid else None
                if path and os.path.exists(path):
                    audio = read_mono(path)
                else:
                    log(f"missing recording {rid}")

            else:
                text = t.get("text", "")
                # Announcements carry their own marking already.
                if not t.get("_announcement"):
                    text = mark_text(text, marking,
                                     first=(i == 1), last=(i == len(tx)))
                    t["text"] = text
                ref = assignment.get(cs)
                ref_path = os.path.join(REFS_DIR, ref + ".wav") if ref else None
                key = (text, ref)

                if key in cache:
                    audio = cache[key]
                else:
                    set_job(jid, step=f"speaking {cs}", done=i - 1)
                    kwargs = {"exaggeration": 0.35, "cfg_weight": 0.5}
                    if ref_path and os.path.exists(ref_path):
                        kwargs["audio_prompt_path"] = ref_path

                    wav = model.generate(spoken(text), **kwargs)
                    arr = wav.squeeze().cpu().numpy().astype(np.float64)

                    if model.sr != SAMPLE_RATE:
                        m = int(len(arr) * SAMPLE_RATE / model.sr)
                        arr = np.interp(np.linspace(0, len(arr) - 1, m),
                                        np.arange(len(arr)), arr)
                    audio = arr
                    cache[key] = audio

            if audio is None or not len(audio):
                continue

            set_job(jid, step=f"radio processing {cs}", done=i)
            treated = apply_radio(audio, prof, seed=1000 + i)
            mix_into(buffers[ch], treated, float(t.get("t", 0)))

            # Default split: the command post is one end of the link, every
            # field station the other. That is the shape of a real patrol
            # net, and it means each radio file holds a coherent side of the
            # conversation rather than a random half.
            side = radio_of.get(cs)
            if side not in ("A", "B"):
                side = "A" if cs == (channels.get("COMMAND") or [cs])[0] else "B"
            t["_radio"] = side
            mix_into(radio_buf[side], treated, float(t.get("t", 0)))

            # Also keep the clip on its own. The mixed session is what you
            # listen to; the individual clips are what gets transmitted,
            # one keying at a time.
            clip_name = f"tx_{i:04d}_{cs.replace('/', '_')}.wav"
            clip_path = os.path.join(CLIPS_DIR, clip_name)
            try:
                sf.write(clip_path, normalise(treated), SAMPLE_RATE)
                t["_clip"] = clip_path.replace("\\", "/")
            except Exception:
                pass

        # --- Write the files ---
        set_job(jid, step="writing files", done=len(tx))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        files = []

        mixed = np.zeros(n, dtype=np.float64)
        for ch, buf in buffers.items():
            mixed += buf
            out = os.path.join(OUT_DIR, f"session_{ch.lower()}_{stamp}.wav")
            sf.write(out, normalise(buf), SAMPLE_RATE)
            files.append(os.path.basename(out))

        out = os.path.join(OUT_DIR, f"session_mixed_{stamp}.wav")
        sf.write(out, normalise(mixed), SAMPLE_RATE)
        files.append(os.path.basename(out))

        # One file per radio. These are what get sent to the hardware — each
        # holds only what that transmitter should key, and both share the
        # session timeline so they stay in step.
        for side, buf in radio_buf.items():
            if not np.any(np.abs(buf) > 1e-9):
                continue
            out = os.path.join(OUT_DIR, f"radio_{side}_{stamp}.wav")
            sf.write(out, normalise(buf), SAMPLE_RATE)
            files.append(os.path.basename(out))

        # Log alongside the audio — for a training exercise this is the
        # after-action record, not a debug file.
        # The plan the SDR transmitter reads: what to send, and when.
        plan = [
            {"t": float(t.get("t", 0)), "cs": t.get("cs", ""),
             "ch": t.get("ch", ""), "text": t.get("text", ""),
             "radio": t.get("_radio", "A"),
             "file": t.get("_clip"), "dur": float(t.get("dur", 0))}
            for t in tx if t.get("_clip")
        ]
        plan_path = os.path.join(OUT_DIR, f"transmit_plan_{stamp}.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump({"duration_s": duration, "transmissions": plan},
                      f, indent=2)
        files.append(os.path.basename(plan_path))

        csv_path = os.path.join(OUT_DIR, f"session_log_{stamp}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("time_s,mm:ss,channel,station,source,message\n")
            for t in tx:
                secs = float(t.get("t", 0))
                mm, ss = divmod(int(secs), 60)
                text = str(t.get("text", "")).replace('"', '""')
                f.write(f'{secs:.2f},{mm:02d}:{ss:02d},{t.get("ch","")},'
                        f'{t.get("cs","")},{t.get("src","")},"{text}"\n')
        files.append(os.path.basename(csv_path))

        set_job(jid, state="done", step="finished", files=files)
        log(f"job {jid} finished — {len(files)} files in {OUT_DIR}\\")

    except Exception as e:
        traceback.print_exc()
        set_job(jid, state="error", error=f"{type(e).__name__}: {e}")


# ===========================================================================
# EXERCISE MARKING
# --------------------------------------------------------------------------
# Simulated traffic is spoken as such. A station that hears a contact report
# has no way of knowing it is practice unless told, and this system produces
# exactly the kind of traffic somebody would act on.
#
# The marking goes into the AUDIO, not just the log. A note in a document
# protects nobody who is listening on the frequency.
# ===========================================================================

EXERCISE_WORD = "Exercise"
NET_ANNOUNCEMENT_EVERY = 180.0     # seconds; 0 disables


def mark_text(text, mode, first=False, last=False):
    """Wrap one transmission according to the chosen marking."""
    if mode == "off":
        return text

    if mode == "period":
        if first:
            return f"{EXERCISE_WORD}, {EXERCISE_WORD}, {EXERCISE_WORD}. {text}"
        if last:
            return f"{text} {EXERCISE_WORD} complete, {EXERCISE_WORD} complete."
        return text

    # "both" / "every"
    return f"{EXERCISE_WORD}. {text} {EXERCISE_WORD}."


def net_announcement(callsign):
    """Periodic reminder on the net. This is the one that protects a monitor
    who tunes in partway through and would otherwise hear nothing but what
    sounds like real traffic."""
    return (f"All stations, this is {callsign}. "
            f"{EXERCISE_WORD}, {EXERCISE_WORD}, {EXERCISE_WORD}. "
            f"This is an exercise transmission. Out.")


def spoken(text):
    """Say callsigns as words. 'ALPHA-1' read literally comes out wrong."""
    words = {"ALPHA": "Alpha", "BRAVO": "Bravo", "CHARLIE": "Charlie",
             "DELTA": "Delta", "ECHO": "Echo", "FOXTROT": "Foxtrot",
             "GOLF": "Golf", "HOTEL": "Hotel", "INDIA": "India",
             "JULIET": "Juliet", "KILO": "Kilo", "LIMA": "Lima"}
    digits = ["Zero", "One", "Two", "Three", "Four",
              "Five", "Six", "Seven", "Eight", "Nine"]

    import re

    def fix(m):
        name, num = m.group(1).upper(), m.group(2)
        base = words.get(name, name.title())
        return base + " " + " ".join(digits[int(d)] for d in num)

    return re.sub(r'\b([A-Za-z]+)-(\d+)\b', fix, text).strip()


def transmit_session(jid, cfg):
    """Put a built session on the air, one transmission at a time.

    Deliberately NOT the whole session file. Ten minutes at 1 MS/s is
    around 4.5 GiB of IQ, and most of it would be silence modulated for no
    reason. Sending clip by clip and waiting out the gaps uses a fraction
    of the memory, sounds the same on the receiving end, and is the shape
    real push-to-talk keying would need anyway.

    Also writes, alongside the existing received-audio recording:

      - a TX-side reference recording — what was actually sent, in order,
        with silence inserted for the same gaps the receiver saw, so the
        two files share one timeline and line up for comparison
      - a message-level log (JSON and a plain-text version) with each
        message's speaker, TX start/end, RX start/end, and status

    The schedule and silence timing themselves aren't new — they already
    come from the transmit plan step7_session_build.py produces. This
    only adds the recording and logging layer on top of the existing
    clip-by-clip send.
    """
    try:
        if not HAVE_PLUTO:
            raise RuntimeError("pluto_link.py is not in this folder")

        tx_uri = cfg.get("tx_uri")
        if not tx_uri:
            found = find_radios()
            if not found:
                raise RuntimeError("no radio is answering to transmit on")
            tx_uri = found[0]["uri"]
        rx_uri = cfg.get("rx_uri")
        carrier = float(cfg.get("carrier_mhz") or 2400.0) * 1e6
        tx = cfg.get("transmissions") or []

        if not tx:
            raise ValueError("nothing to transmit")

        set_job(jid, total=len(tx), step="opening the transmitter")

        sdr = pluto_link.open_tx(tx_uri, carrier=int(carrier))
        rx = pluto_link.open_rx(rx_uri, carrier=int(carrier)) if rx_uri else None

        audio_rate = pluto_link.AUDIO_RATE
        received = []       # demodulated audio actually captured, per message
        sent_ref = []        # what was actually transmitted, same timeline
        rf_levels = []
        log_entries = []
        wall0 = time.time()
        sent = 0
        skipped = 0

        def silence(seconds):
            n = max(0, int(seconds * audio_rate))
            return np.zeros(n, dtype=np.float64)

        for i, t in enumerate(tx, 1):
            if job_cancelled(jid):
                break

            speaker = t.get("cs", "") or t.get("station", "") or "?"
            path = t.get("file")
            if not path or not os.path.exists(path):
                skipped += 1
                log_entries.append({
                    "n": i, "speaker": speaker, "text": t.get("text", ""),
                    "tx_start": None, "tx_end": None,
                    "rx_start": None, "rx_end": None,
                    "status": "SKIPPED — clip file missing",
                })
                continue

            # Wait out the gap, so the timing on air matches the session —
            # and pad both recordings with the same amount of silence, so
            # they stay lined up against this same timeline rather than
            # drifting once a wait gets capped below the real gap.
            target = float(t.get("t", 0))
            behind = target - (time.time() - wall0)
            gap_padded = 0.0
            if behind > 0:
                # Cap it — nobody wants to watch four minutes of dead air
                # during a bench test.
                wait = min(behind, float(cfg.get("max_gap_s") or 4.0))
                set_job(jid, step=f"waiting {wait:.1f}s", done=i - 1)
                slept = 0.0
                while slept < wait:
                    if job_cancelled(jid):
                        break
                    time.sleep(min(0.2, wait - slept))
                    slept += 0.2
                gap_padded = slept
                if job_cancelled(jid):
                    break

            if gap_padded > 0:
                sent_ref.append(silence(gap_padded))
                if rx is not None:
                    received.append(silence(gap_padded))

            set_job(jid, step=f"transmitting {speaker}", done=i)

            audio = pluto_link.load_audio(path)
            iq = pluto_link.nbfm_modulate(audio)
            rx_start_iso = None
            rx_end_iso = None
            status = "SENT — no receiver configured"

            if rx is not None:
                got = {}
                ready = threading.Event()
                rx_thread_start = {}

                def grab():
                    try:
                        # Signals "about to start listening" rather than
                        # "definitely has samples yet" — libiio's rx() call
                        # itself is what actually blocks for data, and there
                        # is no cheaper way to know it has without reading
                        # from it. Good enough to order TX after RX has at
                        # least entered its receive call, which is the gap
                        # that mattered in practice.
                        rx_thread_start["t"] = time.time()
                        ready.set()
                        got["iq"] = pluto_link.receive(
                            rx, len(audio) / audio_rate)
                    except Exception:
                        got["iq"] = None
                        ready.set()

                th = threading.Thread(target=grab)
                th.start()
                ready.wait(timeout=1.0)   # Radio 2 ready before Radio 1 keys up
                rx_start_iso = iso_ms(rx_thread_start.get("t", time.time()))
                tx_start = time.time()
                pluto_link.transmit(sdr, iq)
                th.join()
                rx_end_iso = iso_ms(time.time())

                if got.get("iq") is not None:
                    rf = float(np.sqrt(np.mean(np.abs(got["iq"]) ** 2)))
                    rf_levels.append(rf)
                    received.append(pluto_link.nbfm_demodulate(got["iq"]))
                    status = "RECEIVED" if rf >= 20 else "SENT — weak/no signal"
                else:
                    status = "SENT — nothing captured"
            else:
                tx_start = time.time()
                pluto_link.transmit(sdr, iq)

            tx_end = time.time()
            sent_ref.append(audio)
            sent += 1

            log_entries.append({
                "n": i, "speaker": speaker, "text": t.get("text", ""),
                "tx_start": iso_ms(tx_start), "tx_end": iso_ms(tx_end),
                "rx_start": rx_start_iso, "rx_end": rx_end_iso,
                "status": status,
            })

        files = []
        stamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(OUT_DIR, exist_ok=True)

        if sent_ref:
            tx_out = os.path.join(OUT_DIR, f"net_session_tx_{stamp}.wav")
            sf.write(tx_out, np.concatenate(sent_ref), audio_rate)
            files.append(os.path.basename(tx_out))

        if received:
            rx_out = os.path.join(OUT_DIR, f"net_session_rx_{stamp}.wav")
            sf.write(rx_out, np.concatenate(received), audio_rate)
            files.append(os.path.basename(rx_out))

        if log_entries:
            log_json = os.path.join(OUT_DIR, f"net_session_log_{stamp}.json")
            with open(log_json, "w", encoding="utf-8") as f:
                json.dump({"tx_uri": tx_uri, "rx_uri": rx_uri,
                           "carrier_mhz": carrier / 1e6,
                           "messages": log_entries}, f, indent=2)
            files.append(os.path.basename(log_json))

            log_txt = os.path.join(OUT_DIR, f"net_session_log_{stamp}.txt")
            with open(log_txt, "w", encoding="utf-8") as f:
                for e in log_entries:
                    f.write(f"Message {e['n']:03d}\n")
                    f.write(f"Speaker: {e['speaker']}\n")
                    f.write(f"TX Start: {e['tx_start'] or '-'}\n")
                    f.write(f"TX End: {e['tx_end'] or '-'}\n")
                    f.write(f"RX Start: {e['rx_start'] or '-'}\n")
                    f.write(f"RX End: {e['rx_end'] or '-'}\n")
                    f.write(f"Status: {e['status']}\n\n")
            files.append(os.path.basename(log_txt))

        rx_was_open = rx is not None
        sdr = None
        rx = None

        rf_note = None
        if rx_was_open and rf_levels:
            avg_rf = sum(rf_levels) / len(rf_levels)
            if avg_rf < 20:
                rf_note = (f"Receiver was listening (avg RF level {avg_rf:.1f}) "
                           f"but that's too low to be a real signal — check the "
                           f"cable/attenuator between TX and RX, or the loopback "
                           f"wiring if using one board.")

        errs = []
        if skipped:
            errs.append(f"{skipped} clips missing")
        if rf_note:
            errs.append(rf_note)

        set_job(jid, state="done", step=f"sent {sent} transmissions",
                done=len(tx), files=files,
                error="; ".join(errs) if errs else None)
        log(f"job {jid} — transmitted {sent}, skipped {skipped}")

    except Exception as e:
        traceback.print_exc()
        set_job(jid, state="error", error=f"{type(e).__name__}: {e}")


def transmit_file(jid, cfg):
    """Send one file through one radio, in blocks.

    Whole-file sending, unlike the clip-by-clip session path. Useful when
    you want to hear a complete side of the conversation come out the other
    end exactly as it went in — the simplest possible check that the link
    carries audio faithfully.

    Processed in fixed-size time chunks (CHUNK_SECONDS) rather than as one
    modulated IQ array covering the whole file. IQ at 1 MS/s is 8 bytes a
    sample, so a five-minute file is a couple of GiB held in memory all at
    once for no real reason — the chunked version never holds more than one
    chunk's worth of raw IQ at a time, on either the transmit or receive
    side, so runtime no longer depends on how long the file is. The only
    thing that grows with file length is the demodulated audio being
    accumulated for the final WAV, which at 24 kHz mono is small even for
    a long recording.

    With cfg["repeat"] set, sends the same file over and over — a beacon
    rather than a one-shot check — until the job is cancelled (see
    cancel_job()) or the browser's own request gives up. The radio
    connections are opened once and reused across passes rather than
    reopened every loop, both because reopening is slow and because a
    continuous transmission is closer to how a real beacon behaves.
    Cancellation is checked between every chunk, not just between whole
    passes, so stopping a long file mid-transmission doesn't mean waiting
    for it to finish first.
    """
    CHUNK_SECONDS = 20   # bounds each chunk's raw IQ to ~150 MiB at 1 MS/s

    try:
        if not HAVE_PLUTO:
            raise RuntimeError("pluto_link.py is not in this folder")

        name = os.path.basename(cfg.get("file") or "")
        path = os.path.join(OUT_DIR, name)
        if not name or not os.path.exists(path):
            raise FileNotFoundError(f"no such file: {name}")

        tx_uri = cfg.get("tx_uri")
        rx_uri = tx_uri if cfg.get("loopback") else cfg.get("rx_uri")
        carrier = int(float(cfg.get("carrier_mhz") or 2400) * 1e6)
        repeat = bool(cfg.get("repeat"))
        repeat_pause = max(0.0, float(cfg.get("repeat_pause") or 1.0))

        # Transmit power. Low by default — a Pluto at its maximum carries
        # well beyond a room.
        if cfg.get("gain") is not None:
            pluto_link.TX_GAIN = float(cfg["gain"])

        # Receiver gain. Fixed manual gain rather than the SDR's own AGC, so
        # levels are repeatable between runs — but that means a link that's
        # too hot or too weak needs this adjusted by hand rather than the
        # radio compensating on its own.
        if cfg.get("rx_gain") is not None:
            pluto_link.RX_GAIN = float(cfg["rx_gain"])

        set_job(jid, step=f"reading {name}", total=1)
        audio = pluto_link.load_audio(path)
        audio_rate = pluto_link.AUDIO_RATE
        seconds = len(audio) / audio_rate

        # A generous sanity cap rather than a memory limit — chunking means
        # an hour-long file no longer runs out of RAM, but tying up the
        # transmitter for that long is still almost certainly a mistake
        # (wrong file picked) rather than intentional. Doesn't apply to a
        # deliberate repeat — that's supposed to run indefinitely.
        if not repeat and seconds > 3600:
            raise RuntimeError(
                f"{seconds/60:.0f} minutes is a lot to send as one file — "
                f"double-check this is the file you meant. If it really is, "
                f"the clip-by-clip send handles arbitrary length too.")

        set_job(jid, step="opening the transmitter")
        sdr = pluto_link.open_tx(tx_uri, carrier=carrier)
        rx = pluto_link.open_rx(rx_uri, carrier=carrier) if rx_uri else None

        chunk_samples = max(1, int(CHUNK_SECONDS * audio_rate))
        n_chunks = max(1, -(-len(audio) // chunk_samples))   # ceil division

        passes = 0
        files = []
        try:
            while True:
                passes += 1
                received_parts = []
                sq_sum = 0.0
                sq_count = 0
                cancelled = False

                for i in range(n_chunks):
                    if job_cancelled(jid):
                        cancelled = True
                        break

                    lo = i * chunk_samples
                    hi = min(len(audio), lo + chunk_samples)
                    chunk_audio = audio[lo:hi]
                    if len(chunk_audio) == 0:
                        continue
                    chunk_seconds = len(chunk_audio) / audio_rate

                    prefix = f"pass {passes} — " if repeat else ""
                    set_job(jid, step=f"{prefix}modulating {i+1}/{n_chunks}",
                            done=i, total=n_chunks)
                    iq_chunk = pluto_link.nbfm_modulate(chunk_audio)

                    if rx is not None:
                        got = {}

                        def grab():
                            try:
                                got["iq"] = pluto_link.receive(
                                    rx, chunk_seconds + 0.3)
                            except Exception as e:
                                got["err"] = e

                        th = threading.Thread(target=grab)
                        th.start()
                        # Give the receiver a moment to actually be blocked
                        # inside rx() before this chunk's transmit begins —
                        # needed on every chunk, since each one starts a
                        # fresh receive thread.
                        time.sleep(0.15)
                        set_job(jid, step=f"{prefix}transmitting {i+1}/{n_chunks}",
                                done=i, total=n_chunks)
                        pluto_link.transmit(sdr, iq_chunk)
                        th.join()

                        rx_iq = got.get("iq")
                        if rx_iq is not None:
                            sq_sum += float(np.sum(np.abs(rx_iq) ** 2))
                            sq_count += len(rx_iq)
                            # Raw demod only — no click suppression or
                            # normalization per chunk. There's no real
                            # click at an internal chunk boundary, so doing
                            # that here would fade audio that should stay
                            # full-strength. That finishing pass runs once,
                            # below, over the whole assembled recording.
                            chunk_rec = pluto_link.nbfm_demodulate_raw(rx_iq)
                            if i > 0:
                                # Not a real key-up — just where this
                                # chunk's own modulation restarted its
                                # filter state — but it produces a click of
                                # its own that needs the same treatment,
                                # just localized to this join instead of
                                # the whole recording's true start/end.
                                chunk_rec = pluto_link.declick_join(
                                    chunk_rec, audio_rate)
                            received_parts.append(chunk_rec)
                        del rx_iq   # this chunk's raw IQ can go before the next
                    else:
                        set_job(jid, step=f"{prefix}transmitting {i+1}/{n_chunks}",
                                done=i, total=n_chunks)
                        pluto_link.transmit(sdr, iq_chunk)

                    del iq_chunk

                if rx is not None and received_parts:
                    rec = pluto_link.finish_demodulated(
                        np.concatenate(received_parts), audio_rate)
                    received_parts = None   # let the concatenated array free

                    out = os.path.join(OUT_DIR, f"received_{name}")
                    sf.write(out, rec, audio_rate)
                    files = [os.path.basename(out)]

                    # Numbers rather than listening. Without these you have
                    # no way to tell a working link from a silent one.
                    # rf_level here is the RMS across every chunk's raw IQ
                    # combined — equivalent to what you'd get from the
                    # whole recording at once, just accumulated
                    # incrementally so the raw IQ never has to be kept
                    # around all at once.
                    rf_level = float(np.sqrt(sq_sum / sq_count)) if sq_count else 0.0

                    n = min(len(audio), len(rec))
                    match = 0.0
                    if n > 1000:
                        a = audio[:n] - np.mean(audio[:n])
                        b = rec[:n] - np.mean(rec[:n])
                        d = np.std(a) * np.std(b)
                        if d > 1e-12:
                            match = float(np.mean(a * b) / d)

                    measured = {
                        "rf_level": round(rf_level, 1),
                        "sent_seconds": round(seconds, 2),
                        "received_seconds": round(len(rec) / audio_rate, 2),
                        "sent_peak": round(float(np.max(np.abs(audio))), 3),
                        "received_peak": round(float(np.max(np.abs(rec))), 3),
                        "match": round(match, 3),
                        "passes": passes,
                    }

                    # A verdict, so the operator does not have to interpret it
                    if rf_level < 20:
                        measured["verdict"] = "NO SIGNAL — nothing reached the receiver"
                    elif rf_level > 2000:
                        measured["verdict"] = "OVERLOAD — reduce gain or add attenuation"
                    elif match > 0.7:
                        measured["verdict"] = "LINK GOOD — received audio matches what was sent"
                    elif match > 0.3:
                        measured["verdict"] = "MARGINAL — signal present but degraded"
                    else:
                        measured["verdict"] = "SIGNAL BUT NO MATCH — check carrier and gain"

                    set_job(jid, measured=measured)

                if cancelled or not repeat:
                    break

                set_job(jid, step=f"pass {passes} done — pausing {repeat_pause:.0f}s")
                # Sleep in small slices so a stop click lands quickly
                # instead of waiting out the whole pause.
                slept = 0.0
                while slept < repeat_pause:
                    if job_cancelled(jid):
                        cancelled = True
                        break
                    time.sleep(min(0.2, repeat_pause - slept))
                    slept += 0.2
                if cancelled:
                    break
        finally:
            del sdr
            if rx is not None:
                del rx

        j = get_job(jid)
        step = (f"stopped after {passes} pass{'es' if passes != 1 else ''}"
                if repeat else f"sent {name} ({seconds:.0f}s)")
        set_job(jid, state="done", done=n_chunks, total=n_chunks,
                step=step, files=files, measured=j.get("measured"))

    except Exception as e:
        traceback.print_exc()
        set_job(jid, state="error", error=f"{type(e).__name__}: {e}")


def render_one(jid, rec_id, profile_name):
    """Radio-process a single recording and write it as a WAV."""
    try:
        src = os.path.join(RECORDED_DIR, rec_id + ".wav")
        if not os.path.exists(src):
            raise FileNotFoundError(f"no recording {rec_id}")

        os.makedirs(OUT_DIR, exist_ok=True)
        set_job(jid, step="radio processing", total=1)

        audio = read_mono(src)
        treated = apply_radio(audio, profile_name, seed=42)

        out = os.path.join(OUT_DIR, f"{rec_id}_radio.wav")
        sf.write(out, normalise(treated), SAMPLE_RATE)

        set_job(jid, state="done", step="finished", done=1,
                files=[os.path.basename(out)])
    except Exception as e:
        traceback.print_exc()
        set_job(jid, state="error", error=f"{type(e).__name__}: {e}")


# ===========================================================================
# HTTP
# ===========================================================================

class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass   # the default logger is far too chatty

    # --- helpers ---------------------------------------------------------

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def read_json(self):
        try:
            return json.loads(self.read_body().decode("utf-8"))
        except Exception:
            return {}

    # --- GET -------------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path

        # Browsers always ask for this. Answering keeps the console log
        # clean instead of a 404 on every page load.
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # Your work in progress. Kept on disk beside the project rather
        # than in the browser, so it does not vanish if the address changes,
        # the cache is cleared, or you use a different browser.
        if path == "/api/state":
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, encoding="utf-8") as f:
                        self.send_json({"ok": True, "state": json.load(f)})
                    return
                except Exception as e:
                    self.send_json({"ok": False, "error": str(e)})
                    return
            self.send_json({"ok": True, "state": None})
            return

        # What is available to send. The console offers these as choices
        # rather than assuming which file belongs to which radio.
        if path == "/api/output-files":
            items = []
            if os.path.isdir(OUT_DIR):
                for name in sorted(os.listdir(OUT_DIR), reverse=True):
                    if not name.lower().endswith(".wav"):
                        continue
                    full = os.path.join(OUT_DIR, name)
                    try:
                        info = sf.info(full)
                        items.append({
                            "name": name,
                            "seconds": round(info.duration, 1),
                            "rate": info.samplerate,
                            "mb": round(os.path.getsize(full) / (1024 ** 2), 1),
                            # A guess at which radio it belongs to, from the
                            # filename. The person can always override it.
                            "suggests": ("A" if "_a_" in name.lower()
                                         or name.lower().startswith("radio_a")
                                         else "B" if "_b_" in name.lower()
                                         or name.lower().startswith("radio_b")
                                         else None),
                        })
                    except Exception:
                        pass
            self.send_json({"ok": True, "files": items,
                            "dir": os.path.abspath(OUT_DIR)})
            return

        if path.startswith("/api/radios"):
            force = "refresh=1" in (urlparse(self.path).query or "")
            radios = find_radios(force)
            self.send_json({"ok": True, "radios": radios,
                            "count": len(radios),
                            "problem": _radios.get("problem"),
                            "pluto_link": HAVE_PLUTO})
            return

        if path == "/api/status":
            self.send_json({
                "ok": True,
                "radio_fx": HAVE_FX,
                "ffmpeg": HAVE_FFMPEG,
                "tts": _tts is not None,
                "tts_error": _tts_error,
                "output_dir": os.path.abspath(OUT_DIR),
            })
            return

        if path.startswith("/api/job/"):
            self.send_json(get_job(path.rsplit("/", 1)[-1]))
            return

        # Audio the browser uploaded earlier, handed back after a reload.
        if path.startswith("/api/audio/"):
            parts = path.strip("/").split("/")
            if len(parts) < 4:
                self.send_json({"error": "bad path"}, 400)
                return
            folder = REFS_DIR if parts[2] == "voice" else RECORDED_DIR
            full = os.path.join(folder, os.path.basename(parts[3]) + ".wav")
            if not os.path.exists(full):
                self.send_json({"error": "not found"}, 404)
                return
            with open(full, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/api/download/"):
            name = os.path.basename(path.rsplit("/", 1)[-1])
            full = os.path.join(OUT_DIR, name)
            if not os.path.exists(full):
                self.send_json({"error": "not found"}, 404)
                return
            with open(full, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav" if name.endswith(".wav")
                             else "text/csv")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # --- static files ---
        rel = path.lstrip("/") or "index.html"
        full = os.path.join(WEBUI_DIR, rel)
        if not os.path.isfile(full):
            self.send_json({"error": "not found", "path": rel}, 404)
            return

        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # --- POST ------------------------------------------------------------

    def do_POST(self):
        path = urlparse(self.path).path

        # Upload a recording or a reference voice. The browser sends raw
        # webm; ffmpeg converts it to the WAV everything else expects.
        if path.startswith("/api/upload/"):
            parts = path.strip("/").split("/")
            if len(parts) < 4:
                self.send_json({"error": "bad path"}, 400)
                return
            kind, item_id = parts[2], os.path.basename(parts[3])
            folder = REFS_DIR if kind == "voice" else RECORDED_DIR
            os.makedirs(folder, exist_ok=True)

            raw = self.read_body()
            if not raw:
                self.send_json({"error": "no data"}, 400)
                return

            tmp = os.path.join(folder, item_id + ".webm")
            with open(tmp, "wb") as f:
                f.write(raw)

            wav = os.path.join(folder, item_id + ".wav")
            try:
                if HAVE_FFMPEG:
                    to_wav(tmp, wav)
                    os.remove(tmp)
                else:
                    shutil.move(tmp, wav)   # hope it is already playable
            except Exception as e:
                self.send_json({"error": f"convert failed: {e}"}, 500)
                return

            secs = 0.0
            try:
                info = sf.info(wav)
                secs = info.frames / info.samplerate
            except Exception:
                pass

            self.send_json({"ok": True, "id": item_id,
                            "file": wav, "seconds": round(secs, 2)})
            return

        if path == "/api/state":
            try:
                state = self.read_json()
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=1, ensure_ascii=False)
                self.send_json({"ok": True})
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)}, 500)
            return

        if path == "/api/state/clear":
            for path_ in (STATE_FILE,):
                if os.path.exists(path_):
                    os.remove(path_)
            for folder in (RECORDED_DIR, REFS_DIR):
                if os.path.isdir(folder):
                    for f in os.listdir(folder):
                        try:
                            os.remove(os.path.join(folder, f))
                        except OSError:
                            pass
            self.send_json({"ok": True})
            return

        # Radio-process one recording
        if path == "/api/render-one":
            cfg = self.read_json()
            jid = new_job()
            threading.Thread(
                target=render_one,
                args=(jid, cfg.get("id", ""), cfg.get("profile", "handheld")),
                daemon=True).start()
            self.send_json({"job": jid})
            return

        # Send one whole file through one radio.
        if path == "/api/transmit-file":
            cfg = self.read_json()
            jid = new_job()
            threading.Thread(target=transmit_file, args=(jid, cfg),
                             daemon=True).start()
            self.send_json({"job": jid})
            return

        # Ask a running job to stop after its current step. Cooperative,
        # not instant — see cancel_job()'s note.
        if path.startswith("/api/job/") and path.endswith("/cancel"):
            jid = path.split("/")[3]
            cancel_job(jid)
            self.send_json({"ok": True})
            return

        # Put a built session on the air
        if path == "/api/transmit":
            cfg = self.read_json()
            jid = new_job()
            threading.Thread(target=transmit_session, args=(jid, cfg),
                             daemon=True).start()
            self.send_json({"job": jid})
            return

        # Build the whole session
        if path == "/api/render-session":
            cfg = self.read_json()
            jid = new_job()
            threading.Thread(target=render_session, args=(jid, cfg),
                             daemon=True).start()
            self.send_json({"job": jid})
            return

        self.send_json({"error": "unknown endpoint"}, 404)


# ===========================================================================

def find_port(preferred=PORT, tries=12):
    """Take the preferred port if it is free, otherwise the next one along.

    Something else squatting on 8000 — usually a plain 'python -m http.server'
    left open — used to stop this dead. Moving on quietly is better than
    refusing to start, as long as we say which port we ended up on.
    """
    import socket
    for offset in range(tries):
        port = preferred + offset
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            sock.close()
            return port, offset
        except OSError:
            sock.close()
            continue
    return None, tries


def main():
    print()
    print("=" * 62)
    print("  NET CONTROL")
    print("=" * 62)
    print()

    if not os.path.isdir(WEBUI_DIR):
        print(f"  Cannot find the {WEBUI_DIR}\\ folder.")
        print()
        print("  This file must sit next to it:")
        print()
        print("    your project folder\\")
        print("      server.py            <- this file")
        print("      step5_radio_fx.py")
        print(f"      {WEBUI_DIR}\\")
        print("        index.html")
        print("        style.css")
        print("        app.js")
        print()
        sys.exit(1)

    log(f"radio DSP     {'ready' if HAVE_FX else 'MISSING — no radio processing'}")
    log(f"ffmpeg        {'ready' if HAVE_FFMPEG else 'MISSING — recordings cannot be converted'}")
    log("voice cloning loaded when first needed")
    log(f"SDR link      {'ready' if HAVE_PLUTO else 'pluto_link.py not found'}")

    for d in (REFS_DIR, RECORDED_DIR, CLIPS_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)
    log(f"output        {os.path.abspath(OUT_DIR)}")

    if not HAVE_FX:
        print()
        print("  step5_radio_fx.py was not found in this folder, so nothing")
        print("  can be radio-processed. Everything else still works.")

    if not HAVE_FFMPEG:
        print()
        print("  ffmpeg is not on PATH. Install it with:")
        print("    winget install ffmpeg")
        print("  then close and reopen this window.")

    port, moved = find_port()
    if port is None:
        print()
        print(f"  Could not find a free port between {PORT} and {PORT + 11}.")
        print("  Close some other servers and try again.")
        sys.exit(1)

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"\n  Could not start: {e}")
        sys.exit(1)

    url = f"http://localhost:{port}"

    print()
    print("  " + "-" * 58)
    print(f"   Ready at  {url}")
    print("  " + "-" * 58)

    if moved:
        print()
        print(f"  Port {PORT} was busy, so this is on {port} instead.")
        print("  Use the address above — the old one is something else.")

    print()
    print("  Opening your browser now. Leave this window open while you")
    print("  work; closing it stops the server.")
    print()
    print("  Ctrl+C to stop")
    print()

    # Open the browser once the server is actually accepting connections.
    def open_later():
        time.sleep(1.0)
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_later, daemon=True).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
