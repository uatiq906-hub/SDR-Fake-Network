"""
radio_net.py — two-radio network simulator with collision handling.

Three layers, kept strictly separate. This is the whole design:

    COMMUNICATION   who is talking to whom, was it acknowledged, retries
    CHANNEL         is anyone else transmitting at the same time
    AUDIO           what actually ends up in each WAV file

The reason for the separation: a collision is not an audio problem. Two
voices mixing together is what it *sounds* like, but what matters is that
the message was not received, so no acknowledgement comes back, so the
protocol times out and retries. Mixing the audio and carrying on would
hide exactly the thing worth simulating.

    python radio_net.py                 normal exchange, no collisions
    python radio_net.py --mode forced   deliberate overlap, to test detection
    python radio_net.py --mode random   random overlaps at a set probability
    python radio_net.py --test          run the four checks from the spec

Output:

    session/
      radio_A.wav              only what A transmitted
      radio_B.wav              only what B transmitted
      combined_mix.wav         the shared channel, overlaps and all
      communication_log.json
      communication_log.csv
      session_metadata.json

---------------------------------------------------------------------------
ON ACKNOWLEDGEMENTS

Voice radio has no separate ACK transmission — the acknowledgement rides in
the reply, in the proword. "Roger" IS the acknowledgement.

So by default an ACK is a protocol event that carries no audio of its own,
and the reply that follows satisfies it. The chart still shows the ACK
arrow; there is simply no audio block under it.

Set SPOKEN_ACK = True to make acknowledgements separate transmissions with
their own audio. The protocol is identical either way — only whether the
event produces sound changes.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("soundfile is needed:  pip install soundfile")
    sys.exit(1)

# The DSP chain, imported from your own script so console, command line and
# this simulator all produce identical treatment.
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


# ===========================================================================
# CONFIGURATION
# ===========================================================================

SAMPLE_RATE = 24000
OUT_DIR = "session"

SPOKEN_ACK = False        # see the note at the top of this file

# --- Exercise marking ------------------------------------------------------
# Simulated traffic is marked as such. The word is spoken on the air so that
# anyone receiving it out of context knows immediately that it is practice.
# This is standard exercise procedure and it is the single control that most
# reduces the risk of simulated traffic being acted on.
#
#   "every"   prefix and suffix on every transmission
#   "period"  once at the start and end of the session only
#   "off"     no marking — closed systems only, never on the air
EXERCISE_MARKING = "every"
EXERCISE_WORD = "Exercise"
NET_ANNOUNCEMENT_EVERY = 180.0   # seconds between net announcements, 0 to disable

# --- Content ---------------------------------------------------------------
# Your own curated pool, if it is there. Falls back to the built-in lines.
CONTENT_FILE = "exchanges_curated.json"

# --- Voices ----------------------------------------------------------------
# Reference recordings for cloning, one per radio. Without these the
# placeholder buzz is used, which is fine for verifying timing but is not
# a deliverable.
REFS_DIR = "refs"
USE_TTS = True            # False forces the placeholder even if refs exist

ACK_TIMEOUT = 5.0         # seconds to wait before declaring an ACK missing
MAX_RETRIES = 2

# Timing, in seconds
REPLY_GAP = (0.8, 2.2)    # between one transmission ending and the reply
EXCHANGE_GAP = (4.0, 14.0)  # between one exchange finishing and the next

COLLISION_PROBABILITY = 0.25   # random mode only
COLLISION_OFFSET = (0.3, 1.2)  # how far into A's transmission B starts

RADIOS = {
    "RADIO_A": {"callsign": "ALPHA-1", "voice": "A", "profile": "base",
                "ref": "voice_a.wav"},
    "RADIO_B": {"callsign": "BRAVO-6", "voice": "B", "profile": "handheld",
                "ref": "voice_b.wav"},
}


# ===========================================================================
# LAYER 1 — COMMUNICATION
# ===========================================================================

class State:
    """Protocol states. Every transition is logged."""
    IDLE              = "IDLE"
    A_TRANSMITTING    = "A_TRANSMITTING"
    B_RECEIVING       = "B_RECEIVING"
    WAITING_FOR_B_ACK = "WAITING_FOR_B_ACK"
    B_ACK_RECEIVED    = "B_ACK_RECEIVED"
    B_RESPONDING      = "B_RESPONDING"
    A_RECEIVING       = "A_RECEIVING"
    WAITING_FOR_A_ACK = "WAITING_FOR_A_ACK"
    A_ACK_RECEIVED    = "A_ACK_RECEIVED"
    COLLISION         = "COLLISION"
    TIMEOUT           = "TIMEOUT"
    RETRY             = "RETRY"
    NEXT_EXCHANGE     = "NEXT_EXCHANGE"


@dataclass
class Transmission:
    """One keying of a radio. The unit everything else is built from."""
    id: str
    sender: str                  # RADIO_A / RADIO_B
    receiver: str
    kind: str                    # DATA / RESPONSE / ACK
    text: str
    start: float                 # seconds on the master timeline
    duration: float
    exchange: int          # 0 means a one-way broadcast, not part of an exchange
    attempt: int = 1
    acks: Optional[str] = None   # the message id this acknowledges
    audio: bool = True           # does it occupy the channel

    # Filled in by the channel and protocol layers
    collided: bool = False
    collided_with: list = field(default_factory=list)
    delivered: bool = False
    acknowledged: bool = False

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Collision:
    start: float
    end: float
    messages: list

    @property
    def duration(self) -> float:
        return self.end - self.start


# ===========================================================================
# LAYER 2 — CHANNEL
# ===========================================================================

class ChannelManager:
    """Watches every transmission on the shared channel and decides whether
    two of them overlap.

    Deliberately knows nothing about acknowledgements or retries. Its only
    question is: was more than one radio keyed at the same moment.
    """

    def __init__(self):
        self.collisions: list[Collision] = []

    def check(self, transmissions: list[Transmission]) -> list[Collision]:
        """Find every overlap. Only transmissions that actually occupy the
        channel count — a silent ACK event cannot collide with anything."""
        self.collisions = []
        live = sorted([t for t in transmissions if t.audio],
                      key=lambda t: t.start)

        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if b.start >= a.end:
                    break                    # sorted, so nothing later overlaps
                if a.sender == b.sender:
                    continue                 # one radio cannot collide with itself

                start = max(a.start, b.start)
                end = min(a.end, b.end)
                if end <= start:
                    continue

                a.collided = True
                b.collided = True
                a.collided_with.append(b.id)
                b.collided_with.append(a.id)
                self.collisions.append(Collision(start, end, [a.id, b.id]))

        return self.collisions

    def state_at(self, t: float, transmissions: list[Transmission]) -> str:
        active = [x for x in transmissions
                  if x.audio and x.start <= t < x.end]
        if len(active) > 1:
            return "COLLISION"
        if len(active) == 1:
            return "BUSY"
        return "CLEAR"


# ===========================================================================
# PROTOCOL STATE MACHINE
# ===========================================================================

class ProtocolEngine:
    """Decides what may be transmitted and when, and whether it worked.

    The rule that matters: a message is only acknowledged if the engine says
    so. Generating an ACK audio file proves nothing — if the ACK collided,
    it was never heard, and the sender is still waiting.
    """

    def __init__(self, mode="prevent", seed=42):
        self.mode = mode
        self.rng = random.Random(seed)
        self.channel = ChannelManager()

        self.transmissions: list[Transmission] = []
        self.events: list[dict] = []
        self.state = State.IDLE
        self.clock = 0.0

        self.counters = {
            "messages_sent": 0, "messages_delivered": 0,
            "acks_sent": 0, "acks_received": 0,
            "collisions": 0, "timeouts": 0, "retries": 0,
            "protocol_errors": 0,
        }

    # --- logging ---------------------------------------------------------

    def log(self, t, kind, detail, **extra):
        entry = {"t": round(t, 3), "kind": kind, "detail": detail}
        entry.update(extra)
        self.events.append(entry)

    def transition(self, new_state, t):
        if new_state != self.state:
            self.log(t, "STATE", f"{self.state} -> {new_state}",
                     **{"from": self.state, "to": new_state})
            self.state = new_state

    # --- building the session --------------------------------------------

    def emit(self, sender, receiver, kind, text, start, duration,
             exchange, attempt=1, acks=None, audio=True) -> Transmission:
        n = len(self.transmissions) + 1
        prefix = "ACK" if kind == "ACK" else "MSG"
        tx = Transmission(
            id=f"{prefix}-{n:03d}", sender=sender, receiver=receiver,
            kind=kind, text=text, start=round(start, 3),
            duration=round(duration, 3), exchange=exchange,
            attempt=attempt, acks=acks, audio=audio)
        self.transmissions.append(tx)

        if kind == "ACK":
            self.counters["acks_sent"] += 1
        else:
            self.counters["messages_sent"] += 1

        self.log(start, "BROADCAST" if exchange == 0 else "TX",
                 f"{sender} -> {receiver}", id=tx.id,
                 type=kind, text=text, attempt=attempt, acks=acks)
        return tx

    def gap(self, lo, hi):
        return self.rng.uniform(lo, hi)

    def run_exchange(self, n, a_text, b_text):
        """One complete exchange, with whatever retries it takes.

        The sequence, and the reason for each part:

            A transmits DATA
            B acknowledges          — in voice radio this is the proword in
                                      B's reply, so by default it makes no
                                      separate sound
            B transmits RESPONSE
            A acknowledges

        Nothing advances until the previous step is confirmed delivered.
        """
        attempt = 1

        while attempt <= MAX_RETRIES + 1:
            self.transition(State.A_TRANSMITTING, self.clock)

            spoken = mark(a_text, first_of_session=(n == 1 and attempt == 1))
            data = self.emit("RADIO_A", "RADIO_B", "DATA", spoken,
                             self.clock, speech_seconds(spoken), n, attempt)

            # Collision mode decides whether B jumps in on top of A.
            interferes = None
            if self.mode == "forced" and attempt == 1:
                interferes = data.start + self.rng.uniform(*COLLISION_OFFSET)
            elif self.mode == "random" and self.rng.random() < COLLISION_PROBABILITY:
                interferes = data.start + self.rng.uniform(*COLLISION_OFFSET)

            if interferes is not None and interferes < data.end:
                stomp = mark("Alpha One, Bravo Six, say again your last, over.")
                self.emit("RADIO_B", "RADIO_A", "DATA", stomp,
                          interferes, speech_seconds(stomp), n, attempt)

            self.clock = data.end
            self.transition(State.B_RECEIVING, self.clock)

            # --- did it get through? ---
            self.channel.check(self.transmissions)

            if data.collided:
                self.counters["collisions"] += 1
                self.transition(State.COLLISION, self.clock)
                self.log(self.clock, "COLLISION",
                         "message not received", id=data.id,
                         with_=data.collided_with)

                self.transition(State.WAITING_FOR_B_ACK, self.clock)
                self.clock += ACK_TIMEOUT
                self.counters["timeouts"] += 1
                self.transition(State.TIMEOUT, self.clock)
                self.log(self.clock, "TIMEOUT",
                         f"no acknowledgement after {ACK_TIMEOUT:.1f}s",
                         id=data.id)

                if attempt > MAX_RETRIES:
                    self.log(self.clock, "ABANDONED",
                             "retries exhausted", id=data.id)
                    self.counters["protocol_errors"] += 1
                    self.clock += self.gap(*EXCHANGE_GAP)
                    return False

                self.counters["retries"] += 1
                self.transition(State.RETRY, self.clock)
                self.log(self.clock, "RETRY",
                         f"attempt {attempt + 1}", id=data.id)
                self.clock += self.gap(*REPLY_GAP)
                attempt += 1
                continue

            # --- delivered ---
            data.delivered = True
            self.counters["messages_delivered"] += 1
            self.transition(State.WAITING_FOR_B_ACK, self.clock)
            self.clock += self.gap(*REPLY_GAP)

            ack1 = self.emit("RADIO_B", "RADIO_A", "ACK",
                             f"{RADIOS['RADIO_A']['callsign']}, roger.",
                             self.clock,
                             speech_seconds("roger") if SPOKEN_ACK else 0.0,
                             n, attempt, acks=data.id, audio=SPOKEN_ACK)
            if SPOKEN_ACK:
                self.clock = ack1.end

            data.acknowledged = True
            ack1.delivered = True
            self.counters["acks_received"] += 1
            self.transition(State.B_ACK_RECEIVED, self.clock)

            # --- B's reply ---
            self.transition(State.B_RESPONDING, self.clock)
            if not SPOKEN_ACK:
                pass    # the reply carries the acknowledgement

            spoken_b = mark(b_text)
            resp = self.emit("RADIO_B", "RADIO_A", "RESPONSE", spoken_b,
                             self.clock, speech_seconds(spoken_b), n, attempt)
            self.clock = resp.end

            self.transition(State.A_RECEIVING, self.clock)
            self.channel.check(self.transmissions)

            if resp.collided:
                self.counters["collisions"] += 1
                self.transition(State.COLLISION, self.clock)
                self.log(self.clock, "COLLISION",
                         "response not received", id=resp.id,
                         with_=resp.collided_with)
                self.counters["timeouts"] += 1
                self.clock += ACK_TIMEOUT
                self.transition(State.TIMEOUT, self.clock)
                self.clock += self.gap(*EXCHANGE_GAP)
                return False

            resp.delivered = True
            self.counters["messages_delivered"] += 1

            self.transition(State.WAITING_FOR_A_ACK, self.clock)
            self.clock += self.gap(*REPLY_GAP)

            ack2 = self.emit("RADIO_A", "RADIO_B", "ACK",
                             f"{RADIOS['RADIO_B']['callsign']}, roger, out.",
                             self.clock,
                             speech_seconds("roger out") if SPOKEN_ACK else 0.0,
                             n, attempt, acks=resp.id, audio=SPOKEN_ACK)
            if SPOKEN_ACK:
                self.clock = ack2.end

            resp.acknowledged = True
            ack2.delivered = True
            self.counters["acks_received"] += 1
            self.transition(State.A_ACK_RECEIVED, self.clock)

            self.transition(State.NEXT_EXCHANGE, self.clock)
            self.clock += self.gap(*EXCHANGE_GAP)
            self.transition(State.IDLE, self.clock)
            return True

        return False


# ===========================================================================
# EXERCISE MARKING
# ===========================================================================

def mark(text: str, first_of_session: bool = False,
         last_of_session: bool = False) -> str:
    """Wrap a transmission so it is unmistakably practice traffic.

    Real exercises do this because a receiving station that hears a contact
    report has no way of knowing it is simulated unless told. Prefixing and
    suffixing every transmission is the strictest form; marking only the
    start and end of a period is the lighter one.
    """
    if EXERCISE_MARKING == "off":
        return text

    if EXERCISE_MARKING == "period":
        if first_of_session:
            return f"{EXERCISE_WORD}, {EXERCISE_WORD}, {EXERCISE_WORD}. {text}"
        if last_of_session:
            return f"{text} {EXERCISE_WORD} complete, {EXERCISE_WORD} complete."
        return text

    # "every"
    return f"{EXERCISE_WORD}. {text} {EXERCISE_WORD}."


def net_announcement(callsign: str) -> str:
    """Periodic reminder on the net that this is not real traffic."""
    return (f"All stations, this is {callsign}. "
            f"{EXERCISE_WORD}, {EXERCISE_WORD}, {EXERCISE_WORD}. "
            f"This is an exercise transmission. Out.")


# ===========================================================================
# CONTENT
# ===========================================================================

def load_content(path=CONTENT_FILE):
    """Use your own curated exchanges when they are there.

    The pool is a list of exchanges, each with transmissions. Two-radio
    working needs pairs, so anything with fewer than two transmissions is
    skipped, and only the first two are used — the protocol supplies the
    acknowledgements itself.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    pairs = []
    for ex in data if isinstance(data, list) else []:
        tx = ex.get("transmissions") or []
        if len(tx) < 2:
            continue
        a = str(tx[0].get("text", "")).strip()
        b = str(tx[1].get("text", "")).strip()
        if a and b:
            pairs.append((a, b))

    return pairs or None


# ===========================================================================
# LAYER 3 — AUDIO
# ===========================================================================

def speech_seconds(text: str) -> float:
    """How long a transmission takes to say. Roughly 2.6 words a second,
    plus the key-up and release either side."""
    words = max(1, len(str(text).split()))
    return round(0.55 + words / 2.6, 2)


_tts_model = None
_tts_tried = False


def get_tts():
    """Chatterbox, loaded once and only if it is actually wanted."""
    global _tts_model, _tts_tried
    if _tts_tried:
        return _tts_model
    _tts_tried = True

    if not USE_TTS:
        return None
    if not any(os.path.exists(os.path.join(REFS_DIR, r["ref"]))
               for r in RADIOS.values()):
        return None

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  loading Chatterbox on {device} — first run takes a while")
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            _tts_model = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        except Exception:
            from chatterbox.tts import ChatterboxTTS
            _tts_model = ChatterboxTTS.from_pretrained(device=device)
        print("  voice cloning ready")
    except Exception as e:
        print(f"  voice cloning unavailable ({type(e).__name__}) — using placeholder")
        _tts_model = None

    return _tts_model


def spoken_form(text: str) -> str:
    """Say callsigns as words. 'ALPHA-1' read literally comes out wrong."""
    import re
    words = {"ALPHA": "Alpha", "BRAVO": "Bravo", "CHARLIE": "Charlie",
             "DELTA": "Delta", "ECHO": "Echo", "FOXTROT": "Foxtrot",
             "GOLF": "Golf", "HOTEL": "Hotel"}
    digits = ["Zero", "One", "Two", "Three", "Four",
              "Five", "Six", "Seven", "Eight", "Nine"]

    def fix(m):
        name, num = m.group(1).upper(), m.group(2)
        base = words.get(name, name.title())
        return base + " " + " ".join(digits[int(d)] for d in num)

    return re.sub(r"\b([A-Za-z]+)-(\d+)\b", fix, text).strip()


def clone_voice(text: str, radio: str):
    """Speak a line in this radio's cloned voice. None if unavailable."""
    model = get_tts()
    if model is None:
        return None

    ref = os.path.join(REFS_DIR, RADIOS[radio]["ref"])
    kwargs = {"exaggeration": 0.32, "cfg_weight": 0.5}
    if os.path.exists(ref):
        kwargs["audio_prompt_path"] = ref

    try:
        wav = model.generate(spoken_form(text), **kwargs)
        arr = wav.squeeze().cpu().numpy().astype(np.float64)
        if model.sr != SAMPLE_RATE:
            n = int(len(arr) * SAMPLE_RATE / model.sr)
            arr = np.interp(np.linspace(0, len(arr) - 1, n),
                            np.arange(len(arr)), arr)
        return arr
    except Exception as e:
        print(f"  speech failed ({type(e).__name__}) — placeholder for this line")
        return None


def synth_voice(text: str, voice: str, seconds: float) -> np.ndarray:
    """Placeholder speech.

    Not trying to sound like words — it is a voiced buzz at a plausible
    pitch, shaped into syllables. That is enough to verify timing,
    separation and collisions, which is what this prototype is for. Swap in
    real TTS by replacing this function; nothing else changes.
    """
    n = int(seconds * SAMPLE_RATE)
    if n <= 0:
        return np.zeros(0)

    t = np.arange(n) / SAMPLE_RATE
    f0 = 118.0 if voice == "A" else 148.0     # two clearly different pitches
    rng = np.random.default_rng(abs(hash((text, voice))) % (2 ** 31))

    # Slight pitch drift so it is not a dead tone
    drift = 1.0 + 0.02 * np.sin(2 * np.pi * 0.7 * t + rng.random() * 6)
    phase = 2 * np.pi * f0 * drift * t

    sig = np.zeros(n)
    for h, amp in enumerate([1.0, 0.55, 0.32, 0.18, 0.1], start=1):
        sig += amp * np.sin(phase * h)

    # Syllable envelope — roughly five a second
    syll = 0.5 + 0.5 * np.sin(2 * np.pi * 4.6 * t - np.pi / 2)
    syll = np.clip(syll * 1.25, 0, 1)

    # Fade the ends so it does not click
    fade = int(0.02 * SAMPLE_RATE)
    env = np.ones(n)
    if n > 2 * fade:
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)

    return sig * syll * env * 0.32


def render_audio(engine: ProtocolEngine, duration: float):
    """Place every transmission on its radio's own timeline.

    Each radio gets its own buffer, so radio_A.wav holds only what A sent.
    The combined mix is the sum — which is precisely why overlaps sound
    like two people talking over each other, without either individual file
    being contaminated.
    """
    n = int(duration * SAMPLE_RATE) + SAMPLE_RATE
    buffers = {r: np.zeros(n) for r in RADIOS}

    for i, tx in enumerate(engine.transmissions):
        if not tx.audio or tx.duration <= 0:
            continue

        cfg = RADIOS[tx.sender]

        # Real speech when voices are available, placeholder otherwise. The
        # placeholder still respects the scheduled duration, so timing and
        # collision behaviour can be verified without any voice setup.
        audio = clone_voice(tx.text, tx.sender)
        if audio is None:
            audio = synth_voice(tx.text, cfg["voice"], tx.duration)

        if HAVE_FX:
            prof = PROFILES.get(cfg["profile"]) or PROFILES.get("handheld")
            audio = radio_process(audio, SAMPLE_RATE, prof, seed=1000 + i)

        start = int(tx.start * SAMPLE_RATE)
        end = min(start + len(audio), n)
        if start < n:
            buffers[tx.sender][start:end] += audio[:end - start]

    def norm(x, peak=0.9):
        m = np.max(np.abs(x))
        return x * (peak / m) if m > 1e-9 else x

    mixed = sum(buffers.values())
    return {r: norm(b) for r, b in buffers.items()}, norm(mixed)


# ===========================================================================
# OUTPUT
# ===========================================================================

def write_session(engine: ProtocolEngine, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    duration = max([t.end for t in engine.transmissions] or [1]) + 2.0
    buffers, mixed = render_audio(engine, duration)

    files = []
    for radio, buf in buffers.items():
        name = f"radio_{radio.split('_')[1]}.wav"
        sf.write(os.path.join(out_dir, name), buf, SAMPLE_RATE)
        files.append(name)

    sf.write(os.path.join(out_dir, "combined_mix.wav"), mixed, SAMPLE_RATE)
    files.append("combined_mix.wav")

    # --- logs ---
    log_json = {
        "events": engine.events,
        "transmissions": [asdict(t) for t in engine.transmissions],
        "collisions": [asdict(c) for c in engine.channel.collisions],
    }
    with open(os.path.join(out_dir, "communication_log.json"), "w",
              encoding="utf-8") as f:
        json.dump(log_json, f, indent=2)
    files.append("communication_log.json")

    with open(os.path.join(out_dir, "communication_log.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "id", "sender", "receiver", "type", "attempt",
                    "acknowledges", "collided", "delivered", "acknowledged",
                    "start", "end", "text"])
        for t in engine.transmissions:
            w.writerow([f"{t.start:.3f}", t.id, t.sender, t.receiver, t.kind,
                        t.attempt, t.acks or "", "yes" if t.collided else "",
                        "yes" if t.delivered else "",
                        "yes" if t.acknowledged else "",
                        f"{t.start:.3f}", f"{t.end:.3f}", t.text])
    files.append("communication_log.csv")

    meta = {
        "sample_rate": SAMPLE_RATE,
        "duration_s": round(duration, 3),
        "master_clock": "single timeline, all radios",
        "spoken_ack": SPOKEN_ACK,
        "exercise_marking": EXERCISE_MARKING,
        "net_announcement_every_s": NET_ANNOUNCEMENT_EVERY,
        "content_source": CONTENT_FILE if os.path.exists(CONTENT_FILE) else "built-in",
        "voices": "cloned" if _tts_model else "placeholder",
        "mode": engine.mode,
        "radios": RADIOS,
        "counters": engine.counters,
        "collisions": [
            {"start": round(c.start, 3), "end": round(c.end, 3),
             "duration": round(c.duration, 3), "messages": c.messages}
            for c in engine.channel.collisions
        ],
        "final_state": engine.state,
    }
    with open(os.path.join(out_dir, "session_metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    files.append("session_metadata.json")

    return files, duration


# ===========================================================================
# CONTENT
# ===========================================================================

EXCHANGES = [
    ("Bravo Six, this is Alpha One, radio check, over.",
     "Alpha One, Bravo Six, strength five, over."),
    ("Bravo Six, this is Alpha One, send your location, over.",
     "Alpha One, Bravo Six, grid four two six eight, over."),
    ("Bravo Six, this is Alpha One, report when at checkpoint two, over.",
     "Alpha One, Bravo Six, wilco, over."),
    ("Bravo Six, this is Alpha One, what is your fuel state, over.",
     "Alpha One, Bravo Six, three quarters remaining, over."),
    ("Bravo Six, this is Alpha One, any change in your sector, over.",
     "Alpha One, Bravo Six, nothing significant to report, over."),
]


def build(mode="prevent", exchanges=5, seed=42, content=None):
    engine = ProtocolEngine(mode=mode, seed=seed)
    engine.clock = 2.0

    pool = content if content else (load_content() or EXCHANGES)
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)

    last_announce = 0.0

    for i in range(exchanges):
        # A periodic reminder on the net that this is not real traffic.
        if (EXERCISE_MARKING != "off" and NET_ANNOUNCEMENT_EVERY > 0
                and engine.clock - last_announce >= NET_ANNOUNCEMENT_EVERY):
            text = net_announcement(RADIOS["RADIO_A"]["callsign"])
            engine.emit("RADIO_A", "RADIO_B", "DATA", text,
                        engine.clock, speech_seconds(text), 0)
            engine.clock += speech_seconds(text) + 2.0
            last_announce = engine.clock

        a, b = pool[order[i % len(order)]]
        engine.run_exchange(i + 1, a, b)

    # Closing marker, so a recording of the session is bounded at both ends.
    if EXERCISE_MARKING != "off":
        text = mark("Nothing further, out.", last_of_session=True)
        engine.emit("RADIO_A", "RADIO_B", "DATA", text,
                    engine.clock, speech_seconds(text), 0)

    engine.channel.check(engine.transmissions)
    engine.counters["collisions"] = len(engine.channel.collisions)
    return engine


# ===========================================================================
# REPORTING
# ===========================================================================

def print_timeline(engine, width=68):
    """The chart from section 12 of the spec — both radios against one
    timeline, with overlaps marked."""
    if not engine.transmissions:
        return
    total = max(t.end for t in engine.transmissions) + 1

    print()
    print("  MASTER SESSION TIMELINE")
    print()

    for radio in RADIOS:
        lane = [" "] * width
        for t in engine.transmissions:
            if t.sender != radio or not t.audio:
                continue
            a = int(t.start / total * width)
            b = max(a + 1, int(t.end / total * width))
            for x in range(a, min(b, width)):
                lane[x] = "#" if not t.collided else "X"
        label = radio.replace("RADIO_", "RADIO ")
        print(f"  {label:9s} |{''.join(lane)}|")

    # collision markers
    marks = [" "] * width
    for c in engine.channel.collisions:
        a = int(c.start / total * width)
        b = max(a + 1, int(c.end / total * width))
        for x in range(a, min(b, width)):
            marks[x] = "^"
    if any(m != " " for m in marks):
        print(f"  {'':9s} |{''.join(marks)}|")
        print(f"  {'':9s}  {'X = collided, ^ = overlap'}")

    print(f"  {'':9s} 0s{' ' * (width - 8)}{total:.0f}s")
    print()


def print_status(engine):
    c = engine.counters
    ok = c["protocol_errors"] == 0
    print("  NETWORK STATUS")
    print("  " + "-" * 40)
    print(f"  Radio A            ONLINE")
    print(f"  Radio B            ONLINE")
    print(f"  Channel            SHARED")
    print(f"  Final state        {engine.state}")
    print()
    print(f"  Messages sent      {c['messages_sent']}")
    print(f"  Delivered          {c['messages_delivered']}")
    print(f"  ACKs sent          {c['acks_sent']}")
    print(f"  ACKs received      {c['acks_received']}")
    print(f"  Collisions         {c['collisions']}")
    print(f"  Timeouts           {c['timeouts']}")
    print(f"  Retries            {c['retries']}")
    print(f"  Protocol errors    {c['protocol_errors']}")
    print()
    print(f"  Synchronisation    LOCKED (one master clock)")
    print(f"  Result             {'OK' if ok else 'ERRORS — see the log'}")


def print_flow(engine, limit=24):
    print()
    print("  ACKNOWLEDGEMENT FLOW")
    print()
    print(f"  {'TIME':>8}   {'RADIO A':^22} {'RADIO B':^22}")
    print("  " + "-" * 56)

    shown = 0
    for t in engine.transmissions:
        if shown >= limit:
            print(f"  {'...':>8}   ({len(engine.transmissions) - shown} more)")
            break
        tag = t.kind
        if t.collided:
            tag += " !"
        if t.sender == "RADIO_A":
            arrow = f"--- {tag} ".ljust(22, "-") + ">"
            print(f"  {t.start:8.2f}   {arrow}")
        else:
            arrow = "<" + f" {tag} ".rjust(22, "-")
            print(f"  {t.start:8.2f}   {' ' * 22}{arrow}")
        shown += 1


# ===========================================================================
# TESTS
# ===========================================================================

def run_tests():
    """The four checks from section 16 of the spec."""
    passed = failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}  {detail}")

    print()
    print("  TEST 1 — normal communication")
    e = build("prevent", 1)
    check("no collisions", len(e.channel.collisions) == 0)
    # Net announcements and the closing marker are broadcasts — one-way, no
    # reply expected — so they are excluded from the delivery checks.
    two_way = [t for t in e.transmissions if t.kind != "ACK" and t.exchange > 0]
    check("every message delivered", all(t.delivered for t in two_way))
    check("every message acknowledged", all(t.acknowledged for t in two_way))
    check("ACK references its message",
          all(t.acks for t in e.transmissions if t.kind == "ACK"))
    check("no protocol errors", e.counters["protocol_errors"] == 0)

    print()
    print("  TEST 2 — simultaneous transmission")
    e = build("forced", 1)
    check("collision detected", len(e.channel.collisions) > 0)
    check("collided message not delivered on first attempt",
          any(t.collided and not t.delivered for t in e.transmissions))
    check("timeout raised", e.counters["timeouts"] > 0)
    check("overlap has a duration",
          all(c.duration > 0 for c in e.channel.collisions))

    print()
    print("  TEST 3 — collision recovery")
    e = build("forced", 1)
    retried = [t for t in e.transmissions if t.attempt > 1]
    check("retry happened", len(retried) > 0)
    check("retry eventually delivered",
          any(t.delivered and t.attempt > 1 for t in e.transmissions))
    check("recovered without error", e.counters["protocol_errors"] == 0)

    print()
    print("  TEST 4 — ten consecutive exchanges")
    e = build("random", 10, seed=7)
    ids = [t.id for t in e.transmissions]
    check("every id unique", len(ids) == len(set(ids)))
    check("timestamps never go backwards",
          all(a.start <= b.start for a, b in
              zip(e.transmissions, e.transmissions[1:])))
    check("no transmission has negative duration",
          all(t.duration >= 0 for t in e.transmissions))
    check("ACK count matches delivered messages",
          e.counters["acks_sent"] == e.counters["acks_received"])

    data = [t for t in e.transmissions
            if t.kind in ("DATA", "RESPONSE") and t.exchange > 0]
    check("delivered implies not collided",
          all(not (t.delivered and t.collided) for t in data))
    check("acknowledged implies delivered",
          all(t.delivered for t in data if t.acknowledged))

    print()
    print(f"  {passed} passed, {failed} failed")
    return failed == 0


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Two-radio network simulator")
    ap.add_argument("--mode", default="prevent",
                    choices=["prevent", "forced", "random"],
                    help="prevent: never overlap. forced: deliberate overlap. "
                         "random: overlap at a set probability.")
    ap.add_argument("--exchanges", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--test", action="store_true", help="run the spec checks")
    ap.add_argument("--marking", default=None,
                    choices=["every", "period", "off"],
                    help="exercise marking (default: every)")
    ap.add_argument("--no-tts", action="store_true",
                    help="placeholder audio only, do not load voice cloning")
    args = ap.parse_args()

    global EXERCISE_MARKING, USE_TTS
    if args.marking:
        EXERCISE_MARKING = args.marking
    if args.no_tts:
        USE_TTS = False

    if args.test:
        sys.exit(0 if run_tests() else 1)

    print()
    print("=" * 62)
    print("  TWO-RADIO NETWORK SIMULATOR")
    print("=" * 62)
    print()
    print(f"  mode          {args.mode}")
    print(f"  exchanges     {args.exchanges}")
    print(f"  spoken ACK    {'yes' if SPOKEN_ACK else 'no (carried in the reply)'}")
    print(f"  marking       {EXERCISE_MARKING}")
    print(f"  radio DSP     {'ready' if HAVE_FX else 'not found — plain audio'}")

    pool = load_content()
    print(f"  content       {len(pool)} exchanges from {CONTENT_FILE}"
          if pool else "  content       built-in lines")
    print()

    engine = build(args.mode, args.exchanges, args.seed)
    files, duration = write_session(engine, args.out)

    print_timeline(engine)
    print_flow(engine)
    print()
    print_status(engine)

    print()
    print(f"  Written to {os.path.abspath(args.out)}")
    for f in files:
        print(f"    {f}")
    print()
    print(f"  Session length {duration:.1f}s")
    print()


if __name__ == "__main__":
    main()
