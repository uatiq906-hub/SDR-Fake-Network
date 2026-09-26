"""
bench.py — end-to-end test of the two-radio pipeline.

One question, answered stage by stage:

    text -> Chatterbox -> DSP -> Radio A TX -> Radio B RX -> playback

Nothing else. No scheduling, no collisions, no protocol, no scenarios.
Just: does a message get from one radio to the other with the right voice
and the right audio.

    python bench.py

Then open http://localhost:8010

---------------------------------------------------------------------------
WHAT IT DOES NOT DO

If a stage cannot run it is reported as FAILED with the reason, not quietly
substituted. A bench that silently swaps in a placeholder when Chatterbox is
missing will happily tell you the pipeline works when it does not.

The one exception is the radio hop, which has an explicit LOOPBACK mode for
when no hardware is attached. It is labelled as simulated everywhere it
appears, so it can never be mistaken for a real transmission.
---------------------------------------------------------------------------

Files produced:

    session/
      generated/A_MSG_001_raw.wav     exactly what Chatterbox produced,
      generated/B_MSG_002_raw.wav     untouched
      processed/A_MSG_001_tx.wav      after the radio DSP
      received/A_MSG_001_rx.wav       what came back off the receiver
      radio_A.wav                     everything A transmitted
      radio_B.wav                     everything B transmitted
      combined_mix.wav                both, on one timeline
      session_log.json
"""

import json
import mimetypes
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("soundfile is needed:  pip install soundfile")
    sys.exit(1)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

PORT = 8010
SESSION_DIR = "session"
RAW_DIR = os.path.join(SESSION_DIR, "generated")
TX_DIR = os.path.join(SESSION_DIR, "processed")
RX_DIR = os.path.join(SESSION_DIR, "received")
REFS_DIR = "refs"

SAMPLE_RATE = 24000

RADIOS = {
    "A": {"name": "Radio A", "voice": "Voice A", "ref": "voice_a.wav",
          "profile": "base",     "uri": "ip:192.168.2.1"},
    "B": {"name": "Radio B", "voice": "Voice B", "ref": "voice_b.wav",
          "profile": "handheld", "uri": "ip:192.168.3.1"},
}


# ===========================================================================
# WHAT IS AVAILABLE
# ===========================================================================

try:
    from step5_radio_fx import PROFILES, process as radio_process
    HAVE_FX = True
except ImportError:
    HAVE_FX = False
    PROFILES, radio_process = {}, None

try:
    import pluto_link
    HAVE_PLUTO_MODULE = True
except Exception:
    pluto_link = None
    HAVE_PLUTO_MODULE = False

_tts = None
_tts_error = None
_tts_tried = False


def load_tts():
    """Chatterbox. Loaded once, on demand — it pulls in torch."""
    global _tts, _tts_error, _tts_tried
    if _tts_tried:
        return _tts
    _tts_tried = True

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  loading Chatterbox on {device} — the first time is slow")
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            _tts = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        except Exception:
            from chatterbox.tts import ChatterboxTTS
            _tts = ChatterboxTTS.from_pretrained(device=device)
        print("  Chatterbox ready")
    except Exception as e:
        _tts_error = f"{type(e).__name__}: {e}"
        print(f"  Chatterbox unavailable — {_tts_error}")

    return _tts


def find_radios():
    """Which Plutos are answering."""
    out = []
    for key, cfg in RADIOS.items():
        entry = {"radio": key, "uri": cfg["uri"], "online": False, "detail": ""}
        if HAVE_PLUTO_MODULE:
            try:
                import adi
                dev = adi.Pluto(cfg["uri"])
                entry["online"] = True
                entry["detail"] = f"{dev.sample_rate/1e6:.1f} MS/s"
                del dev
            except ImportError:
                entry["detail"] = "pyadi-iio not installed"
            except Exception as e:
                entry["detail"] = type(e).__name__
        else:
            entry["detail"] = "pluto_link.py not found"
        out.append(entry)
    return out


# ===========================================================================
# SESSION STATE
# ===========================================================================

session = {
    "t0": None,          # wall clock when the session started
    "messages": [],
    "counter": 0,
}
lock = threading.Lock()


def clock():
    """Seconds since the session began. One master clock, so both radio
    files stay on the same timeline."""
    if session["t0"] is None:
        session["t0"] = time.time()
    return time.time() - session["t0"]


def next_id():
    with lock:
        session["counter"] += 1
        return f"MSG-{session['counter']:03d}"


def ensure_dirs():
    for d in (SESSION_DIR, RAW_DIR, TX_DIR, RX_DIR, REFS_DIR):
        os.makedirs(d, exist_ok=True)


# ===========================================================================
# THE PIPELINE
# Each stage returns (ok, detail). Nothing is glossed over.
# ===========================================================================

def stage_generate(text, radio, msg_id):
    """Chatterbox produces the speech. Saved untouched."""
    model = load_tts()
    if model is None:
        return None, False, (_tts_error or "Chatterbox is not installed") + \
            "  —  pip install chatterbox-tts"

    ref = os.path.join(REFS_DIR, RADIOS[radio]["ref"])
    kwargs = {"exaggeration": 0.35, "cfg_weight": 0.5}
    used_ref = os.path.exists(ref)
    if used_ref:
        kwargs["audio_prompt_path"] = ref

    try:
        wav = model.generate(text, **kwargs)
        arr = wav.squeeze().cpu().numpy().astype(np.float64)
        sr = model.sr
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"

    # Saved before anything touches it, so you can hear exactly what
    # Chatterbox produced.
    raw_path = os.path.join(RAW_DIR, f"{radio}_{msg_id}_raw.wav")
    sf.write(raw_path, arr, sr)

    if sr != SAMPLE_RATE:
        n = int(len(arr) * SAMPLE_RATE / sr)
        arr = np.interp(np.linspace(0, len(arr) - 1, n),
                        np.arange(len(arr)), arr)

    detail = (f"{len(arr)/SAMPLE_RATE:.2f}s at {sr} Hz"
              + ("" if used_ref else "  —  no reference voice, built-in used"))
    return {"audio": arr, "path": raw_path, "sr": sr}, True, detail


def stage_dsp(audio, radio, msg_id):
    """The radio treatment — bandpass, compression, noise, squelch."""
    if not HAVE_FX:
        return None, False, "step5_radio_fx.py not found in this folder"

    try:
        prof = PROFILES.get(RADIOS[radio]["profile"]) or PROFILES.get("handheld")
        out = radio_process(audio, SAMPLE_RATE, prof, seed=42)
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"

    path = os.path.join(TX_DIR, f"{radio}_{msg_id}_tx.wav")
    sf.write(path, out, SAMPLE_RATE)
    return {"audio": out, "path": path}, True, \
        f"{RADIOS[radio]['profile']} profile, {len(out)/SAMPLE_RATE:.2f}s"


def stage_radio(audio, tx_radio, rx_radio, msg_id, mode, carrier_mhz):
    """The hop between radios.

    In LOOPBACK there is no hardware — the audio is handed straight across.
    Everything else in the chain is genuinely exercised, but the radio hop
    is simulated, and it says so everywhere it appears.
    """
    rx_path = os.path.join(RX_DIR, f"{tx_radio}_{msg_id}_rx.wav")

    if mode == "loopback":
        sf.write(rx_path, audio, SAMPLE_RATE)
        return {"audio": audio, "path": rx_path, "simulated": True}, True, \
            "LOOPBACK — no hardware, audio passed straight across"

    if not HAVE_PLUTO_MODULE:
        return None, False, "pluto_link.py not found"

    tx_uri = RADIOS[tx_radio]["uri"]
    rx_uri = RADIOS[rx_radio]["uri"]

    try:
        carrier = int(float(carrier_mhz) * 1e6)
        sdr_tx = pluto_link.open_tx(tx_uri, carrier=carrier)
        sdr_rx = pluto_link.open_rx(rx_uri, carrier=carrier)
    except Exception as e:
        return None, False, f"could not open the radios: {type(e).__name__}: {e}"

    try:
        iq = pluto_link.nbfm_modulate(audio)
        seconds = len(audio) / SAMPLE_RATE

        got = {}

        def grab():
            try:
                got["iq"] = pluto_link.receive(sdr_rx, seconds + 0.3)
            except Exception as e:
                got["err"] = e

        th = threading.Thread(target=grab)
        th.start()
        time.sleep(0.25)
        pluto_link.transmit(sdr_tx, iq)
        th.join()

        if got.get("iq") is None:
            return None, False, f"nothing received: {got.get('err','no samples')}"

        rx_iq = got["iq"]
        level = float(np.sqrt(np.mean(np.abs(rx_iq) ** 2)))
        recovered = pluto_link.nbfm_demodulate(rx_iq)

        sf.write(rx_path, recovered, SAMPLE_RATE)

        note = ""
        if level < 20:
            note = "  —  very weak, check the cable and attenuator"
        elif level > 2000:
            note = "  —  very strong, likely clipping, add attenuation"

        return {"audio": recovered, "path": rx_path, "simulated": False,
                "level": level}, True, \
            f"{tx_uri} -> {rx_uri} at {carrier_mhz} MHz, rx level {level:.0f}{note}"

    except Exception as e:
        traceback.print_exc()
        return None, False, f"{type(e).__name__}: {e}"
    finally:
        try:
            del sdr_tx, sdr_rx
        except Exception:
            pass


def correlate(a, b):
    """How closely two signals match. 1.0 is identical."""
    n = min(len(a), len(b))
    if n < 100:
        return 0.0
    x = a[:n] - np.mean(a[:n])
    y = b[:n] - np.mean(b[:n])
    denom = np.std(x) * np.std(y)
    return float(np.mean(x * y) / denom) if denom > 1e-12 else 0.0


def compare(raw, transmitted, received):
    """Did the audio survive.

    The comparison that matters is TRANSMITTED against RECEIVED, because
    that is the only part the radio hop is responsible for.

    Comparing the raw Chatterbox output against what came back would be
    misleading: the DSP is supposed to change the audio. It band-limits it
    to 300-3400 Hz, compresses it hard, adds a noise floor and appends a
    squelch tail. A low correlation there is the processing working, not a
    fault, and reporting it as a failure would send you hunting for a
    problem that does not exist.
    """
    if transmitted is None or received is None:
        return {}

    t_dur = len(transmitted) / SAMPLE_RATE
    r_dur = len(received) / SAMPLE_RATE

    out = {
        "raw_duration": round(len(raw) / SAMPLE_RATE, 3) if raw is not None else None,
        "tx_duration": round(t_dur, 3),
        "rx_duration": round(r_dur, 3),
        "duration_drift": round(r_dur - t_dur, 3),
        "sample_rate": SAMPLE_RATE,
        "tx_peak": round(float(np.max(np.abs(transmitted))), 3),
        "rx_peak": round(float(np.max(np.abs(received))), 3),
        # The one that judges the radio hop
        "link_match": round(correlate(transmitted, received), 3),
    }

    # Informational only — the DSP is meant to change things.
    if raw is not None:
        out["dsp_change"] = round(correlate(raw, transmitted), 3)

    return out


def send_message(text, tx_radio, mode, carrier_mhz):
    """One message, all the way through. Returns a record of every stage."""
    ensure_dirs()
    rx_radio = "B" if tx_radio == "A" else "A"
    msg_id = next_id()
    start = clock()

    record = {
        "id": msg_id,
        "sender": RADIOS[tx_radio]["name"],
        "receiver": RADIOS[rx_radio]["name"],
        "sender_key": tx_radio,
        "receiver_key": rx_radio,
        "voice": RADIOS[tx_radio]["voice"],
        "text": text,
        "start_time": round(start, 3),
        "mode": mode,
        "stages": [],
        "status": "RUNNING",
    }

    def note(name, ok, detail, path=None):
        record["stages"].append({"stage": name, "ok": ok,
                                 "detail": detail, "file": path})

    note("TEXT", True, f"{len(text.split())} words")

    gen, ok, detail = stage_generate(text, tx_radio, msg_id)
    note("CHATTERBOX", ok, detail, gen["path"] if gen else None)
    if not ok:
        record["status"] = "FAILED"
        record["failed_at"] = "CHATTERBOX"
        return record

    note("RAW AUDIO", True,
         f"saved untouched as {os.path.basename(gen['path'])}", gen["path"])

    dsp, ok, detail = stage_dsp(gen["audio"], tx_radio, msg_id)
    note("RADIO DSP", ok, detail, dsp["path"] if dsp else None)
    if not ok:
        record["status"] = "FAILED"
        record["failed_at"] = "RADIO DSP"
        return record

    note(f"RADIO {tx_radio} TX", True,
         f"{len(dsp['audio'])/SAMPLE_RATE:.2f}s ready to send")

    rx, ok, detail = stage_radio(dsp["audio"], tx_radio, rx_radio,
                                 msg_id, mode, carrier_mhz)
    note(f"RADIO {rx_radio} RX", ok, detail, rx["path"] if rx else None)
    if not ok:
        record["status"] = "FAILED"
        record["failed_at"] = f"RADIO {rx_radio} RX"
        return record

    note("RECEIVED", True,
         f"{len(rx['audio'])/SAMPLE_RATE:.2f}s written", rx["path"])

    # The message occupies the timeline for as long as the AUDIO lasts, not
    # for however long the machine took to produce it. Using wall-clock
    # elapsed here made a three-second transmission occupy a fraction of a
    # second, and the audio then overran the session buffer.
    audio_len = len(dsp["audio"]) / SAMPLE_RATE
    end = start + audio_len

    record.update({
        "end_time": round(end, 3),
        "duration": round(audio_len, 3),
        "generated_in": round(clock() - start, 3),
        "raw_file": gen["path"].replace("\\", "/"),
        "tx_file": dsp["path"].replace("\\", "/"),
        "rx_file": rx["path"].replace("\\", "/"),
        "simulated_radio": rx.get("simulated", False),
        "comparison": compare(gen["audio"], dsp["audio"], rx["audio"]),
        "status": "COMPLETE",
    })

    with lock:
        session["messages"].append(record)
        # Push the clock past this transmission, plus a short gap, so the
        # next message lands after it rather than on top of it.
        session["t0"] -= max(0.0, end - clock()) + 1.2

    return record


# ===========================================================================
# BUILDING THE SESSION FILES
# ===========================================================================

def build_session_files():
    """Lay every transmission on one timeline and write the three WAVs.

    Both radio files share the master clock, so a transmission at 00:02.4
    appears at 00:02.4 in whichever file it belongs to and nowhere else.
    """
    with lock:
        msgs = [m for m in session["messages"] if m.get("status") == "COMPLETE"]

    if not msgs:
        return None, "no completed messages yet"

    total = max(m["end_time"] for m in msgs) + 1.0
    n = int(total * SAMPLE_RATE)
    buffers = {"A": np.zeros(n), "B": np.zeros(n)}

    for m in msgs:
        path = m["tx_file"].replace("/", os.sep)
        if not os.path.exists(path):
            continue
        audio, _ = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        start = int(m["start_time"] * SAMPLE_RATE)
        end = min(start + len(audio), n)
        if start < n:
            buffers[m["sender_key"]][start:end] += audio[:end - start]

    def norm(x, peak=0.9):
        mx = np.max(np.abs(x))
        return x * (peak / mx) if mx > 1e-9 else x

    files = []
    for key, buf in buffers.items():
        path = os.path.join(SESSION_DIR, f"radio_{key}.wav")
        sf.write(path, norm(buf), SAMPLE_RATE)
        files.append(path)

    mixed = norm(buffers["A"] + buffers["B"])
    mix_path = os.path.join(SESSION_DIR, "combined_mix.wav")
    sf.write(mix_path, mixed, SAMPLE_RATE)
    files.append(mix_path)

    log_path = os.path.join(SESSION_DIR, "session_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"sample_rate": SAMPLE_RATE, "duration_s": round(total, 3),
                   "messages": msgs}, f, indent=2)
    files.append(log_path)

    return files, f"{len(msgs)} messages over {total:.1f}s"


# ===========================================================================
# HTTP
# ===========================================================================

jobs = {}


class Handler(BaseHTTPRequestHandler):

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # --- GET ---

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path

        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path == "/api/status":
            self.send_json({
                "ok": True,
                "chatterbox": _tts is not None,
                "chatterbox_tried": _tts_tried,
                "chatterbox_error": _tts_error,
                "radio_fx": HAVE_FX,
                "pluto_link": HAVE_PLUTO_MODULE,
                "radios": find_radios(),
                "refs": {k: os.path.exists(os.path.join(REFS_DIR, v["ref"]))
                         for k, v in RADIOS.items()},
                "messages": len(session["messages"]),
                "session_dir": os.path.abspath(SESSION_DIR),
            })
            return

        if path == "/api/messages":
            with lock:
                self.send_json({"ok": True, "messages": session["messages"]})
            return

        if path.startswith("/api/job/"):
            self.send_json(jobs.get(path.rsplit("/", 1)[-1],
                                    {"state": "unknown"}))
            return

        if path.startswith("/audio/"):
            rel = path[len("/audio/"):]
            full = os.path.normpath(os.path.join(SESSION_DIR, rel))
            if not full.startswith(os.path.abspath(SESSION_DIR)) and \
               not os.path.abspath(full).startswith(os.path.abspath(SESSION_DIR)):
                self.send_json({"error": "no"}, 403)
                return
            if not os.path.exists(full):
                self.send_json({"error": "not found", "path": rel}, 404)
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

        # static
        name = path.lstrip("/") or "bench.html"
        if not os.path.isfile(name):
            self.send_json({"error": "not found", "path": name}, 404)
            return
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        with open(name, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # --- POST ---

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/send":
            cfg = self.read_json()
            text = (cfg.get("text") or "").strip()
            radio = (cfg.get("radio") or "A").upper()
            mode = cfg.get("mode") or "loopback"
            carrier = cfg.get("carrier_mhz") or 2400

            if not text:
                self.send_json({"ok": False, "error": "no text"}, 400)
                return
            if radio not in RADIOS:
                self.send_json({"ok": False, "error": "unknown radio"}, 400)
                return

            jid = uuid.uuid4().hex[:10]
            jobs[jid] = {"state": "running", "step": "starting"}

            def work():
                try:
                    rec = send_message(text, radio, mode, carrier)
                    jobs[jid] = {"state": "done", "record": rec}
                except Exception as e:
                    traceback.print_exc()
                    jobs[jid] = {"state": "error",
                                 "error": f"{type(e).__name__}: {e}"}

            threading.Thread(target=work, daemon=True).start()
            self.send_json({"ok": True, "job": jid})
            return

        if path == "/api/build":
            files, detail = build_session_files()
            if not files:
                self.send_json({"ok": False, "error": detail})
                return
            self.send_json({"ok": True, "detail": detail,
                            "files": [os.path.basename(f) for f in files],
                            "dir": os.path.abspath(SESSION_DIR)})
            return

        if path == "/api/reset":
            with lock:
                session["messages"] = []
                session["counter"] = 0
                session["t0"] = None
            jobs.clear()
            self.send_json({"ok": True})
            return

        if path == "/api/warmup":
            jid = uuid.uuid4().hex[:10]
            jobs[jid] = {"state": "running", "step": "loading Chatterbox"}

            def work():
                m = load_tts()
                jobs[jid] = {"state": "done" if m else "error",
                             "error": _tts_error,
                             "step": "ready" if m else "failed"}

            threading.Thread(target=work, daemon=True).start()
            self.send_json({"ok": True, "job": jid})
            return

        self.send_json({"error": "unknown endpoint"}, 404)


# ===========================================================================

def main():
    print()
    print("=" * 62)
    print("  TWO-RADIO PIPELINE BENCH")
    print("=" * 62)
    print()
    print("  text -> Chatterbox -> DSP -> Radio A TX -> Radio B RX -> playback")
    print()

    ensure_dirs()

    print(f"  radio DSP     {'ready' if HAVE_FX else 'MISSING (step5_radio_fx.py)'}")
    print(f"  SDR link      {'ready' if HAVE_PLUTO_MODULE else 'MISSING (pluto_link.py)'}")
    print("  Chatterbox    loaded when first needed")

    for k, cfg in RADIOS.items():
        ref = os.path.join(REFS_DIR, cfg["ref"])
        state = "found" if os.path.exists(ref) else "MISSING"
        print(f"  {cfg['name']} voice  {cfg['ref']:16s} {state}")

    print(f"  output        {os.path.abspath(SESSION_DIR)}")
    print()

    if not os.path.exists("bench.html"):
        print("  bench.html is not in this folder — the console will not load.")
        print()

    import socket
    port = PORT
    for offset in range(10):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", PORT + offset))
            s.close()
            port = PORT + offset
            break
        except OSError:
            s.close()

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"

    print("  " + "-" * 58)
    print(f"   Ready at  {url}")
    print("  " + "-" * 58)
    print()
    print("  Leave this window open. Ctrl+C to stop.")
    print()

    def open_browser():
        time.sleep(1.0)
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
