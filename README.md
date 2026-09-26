# Net Control — AI-Generated Radio Net Simulator

An offline pipeline that generates, synthesises, and plays back realistic
military-style radio traffic for training, exercise support, and radio-system
integration testing. It also includes a from-scratch two-radio network
simulator with collision detection and retry logic, and an optional
software-defined-radio (SDR) hardware link for testing the full audio chain
over real RF hardware in a closed, conducted (non-radiating) loop.

Everything runs locally. No audio, text, or session data leaves the machine
unless the operator deliberately puts it on the air through an authorised
exercise setup (see [On transmitting](#on-transmitting)).

> **Status:** personal / academic project, built solo. Some components
> (marked below) have been exercised more thoroughly than others — see
> [Reproducibility & current limitations](#reproducibility--current-limitations).

---

## Table of contents

- [Why this project exists](#why-this-project-exists)
- [Key features](#key-features)
- [System architecture](#system-architecture)
- [Technologies used](#technologies-used)
- [Project structure](#project-structure)
- [Installation](#installation)
- [How to run it](#how-to-run-it)
- [Example output](#example-output)
- [The two-radio simulator, in detail](#the-two-radio-simulator-in-detail)
- [PlutoSDR hardware link](#plutosdr-hardware-link)
- [On transmitting](#on-transmitting)
- [Reproducibility & current limitations](#reproducibility--current-limitations)
- [Future improvements](#future-improvements)
- [Author](#author)

---

## Why this project exists

Realistic radio traffic is useful for training radio operators, testing how
comms and command-post software behaves under real net conditions (collisions,
retries, variable signal quality), and rehearsing exercises — but producing
enough of it by hand (writing exchanges, voicing every callsign, timing a
believable net) doesn't scale.

This project builds that traffic end-to-end: an LLM writes plausible
exchanges for a set of scenarios, each line is spoken in a distinct cloned
voice, the audio is run through a DSP chain that makes it sound like it came
off an actual handheld radio, and the result is scheduled into a timed,
multi-channel session — all clearly and repeatedly marked as an exercise, and
with a separate simulator that models how two radios actually behave on a
shared channel (collisions, ACKs, retries) rather than just concatenating
audio clips.

## Key features

- **LLM-driven scenario generation** — a local model (Llama 3.1 8B via Ollama)
  writes radio exchanges from a bank of situations, unit pairs, and event
  threads, with balanced sampling and live deduplication so output doesn't
  repeat itself. A variety-scoring step (`step2_curate.py stats`) flags when
  more generation won't help and new scenarios are needed instead.
- **Voice cloning per callsign** — each station gets its own cloned voice
  (Chatterbox TTS) from a short reference recording, instead of every
  transmission sounding like the same speaker.
- **A radio DSP chain** (`step5_radio_fx.py`) that band-limits, saturates,
  compresses, adds fading and a scaled noise floor, and adds key-click/squelch
  artefacts — the difference between "clean TTS" and "sounds like a radio."
- **A from-scratch two-radio network simulator** (`radio_net.py`) that models
  communication, channel contention, and audio as three separate layers, so a
  collision is a *protocol* event (no ACK → timeout → retry), not just two
  waveforms mixed together. Includes forced- and random-collision test modes
  and an 18-check automated test suite (`--test`).
- **A local web console** (`server.py` + `webui/`) for recording reference
  voices, composing or generating messages, assigning callsigns/voices, and
  building a session — without touching the command line.
- **A live network visualiser** (`network_view.html`) showing the collision
  timeline, acknowledgement flow, and event log for a simulator run.
- **Exercise-marking safeguards built into the traffic generator and the
  transmit path** — every transmission is marked "Exercise" by default, and
  `net_transmit.py` refuses to key a radio on a session whose text isn't
  properly marked unless explicitly overridden.
- **Conducted-loop SDR testing** (`pluto_link.py`, `net_transmit.py`) — puts
  the generated audio through two PlutoSDR units over a cable and attenuator
  (real modulation, real receive path, nothing radiating), for integration
  testing without needing spectrum authority.

## System architecture

```text
                    ┌────────────────────┐
                    │   step1_generate.py │  LLM (Ollama / Llama 3.1)
                    │  scenario → text    │  writes radio exchanges
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │   step2_curate.py   │  dedupe, variety check,
                    │  review / keep      │  human curation
                    └─────────┬──────────┘
                              ▼
        ┌─────────────────────┴─────────────────────┐
        ▼                                             ▼
┌───────────────────┐                     ┌────────────────────────┐
│ step3_prep.py      │                     │ step4_render_voices.py │
│ step3b_import_...  │                     │ Chatterbox TTS +       │
│ clean up recordings │                     │ cloned reference voice │
└──────────┬─────────┘                     └───────────┬────────────┘
           │                                            │
           └───────────────────┬────────────────────────┘
                                ▼
                    ┌────────────────────┐
                    │ step6_process_clips │  step5_radio_fx.py DSP chain
                    │  radio character    │  applied to every clip
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ step7_session_build │  timing, silence ratio,
                    │  → timed session    │  net discipline, doubling
                    └─────────┬──────────┘
                              ▼
              ┌───────────────┴────────────────┐
              ▼                                 ▼
   output/session_*.wav + logs         net_transmit.py / pluto_link.py
   (finished session, for playback)     (conducted-loop SDR test, optional)

──────────────────────────── separate track ────────────────────────────

           radio_net.py  (two-radio protocol simulator)
                 │
     COMMUNICATION layer   —  who's talking, ACKs, retries
     CHANNEL layer         —  is more than one radio keyed at once
     AUDIO layer           —  what ends up in each WAV
                 │
                 ▼
      session/radio_A.wav, radio_B.wav, combined_mix.wav, logs
                 │
                 ▼
        network_view.html  (collision / ACK / event timeline)
```

`server.py` and `webui/` provide a browser console over the same pipeline
(steps 3–7) so the work can be done without the command line; the two are
functionally equivalent for that part of the pipeline.

## Technologies used

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| LLM scenario generation | Ollama (local), Llama 3.1 8B |
| Text-to-speech / voice cloning | Chatterbox TTS, PyTorch, torchaudio |
| Signal processing | NumPy, SciPy, librosa, soundfile |
| Local web app | Plain HTML/CSS/JS front end, Python `http.server`-based backend (no framework) |
| SDR hardware | Analog Devices PlutoSDR, `pyadi-iio` |
| Audio conversion | ffmpeg |

## Project structure

```text
net-control/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── server.py                    # local web console backend — RUN THIS to use the console
├── radio_net.py                 # two-radio protocol simulator
├── net_transmit.py              # walks a built session live across two SDRs
├── pluto_link.py                # low-level PlutoSDR TX/RX (conducted loop)
├── bench.py                     # end-to-end pipeline smoke test + its own mini console
├── bench.html                   # console for bench.py (must stay next to bench.py)
├── step1_generate.py            # Phase 1 — LLM scenario/exchange generation
├── step2_curate.py              # Phase 1 — dedupe, variety check, human curation
├── step3_prep.py                # Phase 2 — clean up recorded reference/message audio
├── step3b_import_recorded.py    # Phase 2 — bring console recordings into the pipeline
├── step4_render_voices.py       # Phase 2 — TTS + voice cloning
├── step5_radio_fx.py            # DSP chain — imported by step6, server.py and bench.py
├── step6_process_clips.py       # Phase 3 — apply radio DSP to every clip
├── step7_session_build.py       # Phase 4 — schedule clips into a timed session
├── webui/                       # front end for server.py (must stay next to server.py)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── docs/
│   └── network_view.html        # standalone viewer for radio_net.py's logs (open directly in a browser)
├── setup/
│   ├── localhost.bat            # Windows: run server.py from anywhere (copy into repo root to use)
│   └── setup-localhost-command.ps1  # adds a `localhost` PowerShell command (copy into repo root to use)
└── samples/
    ├── sample_transmit_plan.json    # example step7 output — what a built session's plan looks like
    └── sample_session_log.csv       # example step7 output — timestamped transmission log
```

**Not included in this repository** (see [Reproducibility](#reproducibility--current-limitations)
for why, and how to regenerate them):

| Path | What it holds | Why it's excluded |
|---|---|---|
| `refs/` | Reference voice recordings used for cloning | Personal voice recordings — not something to publish |
| `recorded/` | Console-recorded message audio | User-generated at run time, often personal recordings |
| `clips/` | Every rendered/processed speech clip | Fully regenerable from the pipeline; hundreds of MB |
| `output/`, `session/` | Finished session WAVs, logs, transmit plans | Regenerable; the full set is close to 1 GB of audio |
| `exchanges.json`, `exchanges_curated.json`, `manifest*.json` | Pipeline intermediate state | Regenerable by re-running the relevant step |
| `__pycache__/` | Compiled bytecode | Not source |

## Installation

```bash
git clone https://github.com/<your-username>/net-control.git
cd net-control

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

External dependencies not covered by `pip`:

- **ffmpeg** — required to convert browser-recorded audio in the console.
  `winget install ffmpeg` (Windows) or your OS package manager, then make
  sure it's on `PATH`.
- **Ollama**, with the `llama3.1:8b` model pulled — required only for
  `step1_generate.py`. Install from <https://ollama.com>, then:
  ```bash
  ollama pull llama3.1:8b
  ```
- **PlutoSDR + `pyadi-iio`** — required only for `pluto_link.py` /
  `net_transmit.py`. Two PlutoSDR units, a coax cable, and a 30–40 dB
  in-line attenuator are needed for the conducted-loop hardware test; see
  [PlutoSDR hardware link](#plutosdr-hardware-link).

Without Chatterbox TTS installed, the console and pipeline still run —
recorded (as opposed to LLM-written) messages work in full, and the two-radio
simulator falls back to a placeholder tone that still respects scheduled
timing, so protocol and collision behaviour can be verified with no voice
setup at all.

## How to run it

### Option A — the console (recommended starting point)

```bash
python server.py
```

This serves the console in your browser and runs the real pipeline (DSP,
voice cloning) on your machine — nothing is uploaded anywhere. If port 8000
is in use it moves to 8001 automatically. From the console you can record
reference voices, write or generate messages, assign callsigns/voices, and
build a session; output lands in `output/`.

### Option B — the pipeline, from the command line

```bash
python step1_generate.py 500          # write ~500 candidate exchanges
python step2_curate.py stats          # check variety before reading them
python step2_curate.py review         # keep the good ones
python step4_render_voices.py         # speak them (needs Chatterbox + optional refs/)
python step6_process_clips.py         # apply radio DSP
python step7_session_build.py         # schedule into a timed session
```

Curation (`step2_curate.py`) is a manual, human-in-the-loop step by design —
`stats` reports a variety percentage; below 60% means the model has started
repeating itself and more generation won't help, so add scenarios instead.

### Option C — the two-radio protocol simulator (independent of the pipeline above)

```bash
python radio_net.py                    # clean run, no collisions
python radio_net.py --mode forced      # deliberate overlap, to test detection
python radio_net.py --mode random      # random overlaps at a set probability
python radio_net.py --test             # 18 automated checks
```

Then open `docs/network_view.html` directly in a browser for the collision
timeline and event log.

## Example output

A built session produces a transmit plan and a timestamped log — samples are
included in [`samples/`](samples/):

```json
{
  "t": 7.31,
  "cs": "ALPHA-1",
  "ch": "COMMAND",
  "text": "All stations, this is ALPHA-1. Exercise, Exercise, Exercise. This is an exercise transmission. Out.",
  "radio": "A",
  "file": "clips/tx_0001_ALPHA-1.wav",
  "dur": 0.0
}
```

```csv
time_s,mm:ss,channel,station,source,message
7.31,00:07,COMMAND,ALPHA-1,generated,"All stations, this is ALPHA-1. Exercise, Exercise, Exercise. This is an exercise transmission. Out."
10.31,00:10,COMMAND,BRAVO-6,generated,"Exercise. Alpha One, this is Bravo Six, message follows, over. Exercise."
```

Note every line above is marked `Exercise` — this is on by default and is
the main control against simulated traffic being mistaken for real traffic;
see [On transmitting](#on-transmitting).

Actual audio output isn't included in the repository (see
[Project structure](#project-structure)), but running `radio_net.py --test`
or the console end-to-end will regenerate equivalent WAVs and logs locally.

## The two-radio simulator, in detail

`radio_net.py` is a self-contained protocol simulator, separate from the
generation pipeline. It exists to prove — and let you inspect — how two
radio stations behave on a shared channel, not just to produce audio:

- **Three layers, kept apart**: `COMMUNICATION` (who's talking, was it
  acknowledged, retries), `CHANNEL` (is more than one radio keyed at once),
  and `AUDIO` (what actually ends up in each WAV). A collision is treated as
  a protocol event — no ACK comes back, so the sender times out and retries —
  rather than an audio-mixing artefact. `delivered` is set by the protocol
  engine, never inferred from the presence of audio.
- **Acknowledgements are implicit by default**: voice radio doesn't have a
  separate ACK transmission — "Roger" *is* the acknowledgement, carried in
  the reply's own audio. Setting `SPOKEN_ACK = True` makes acknowledgements
  separate spoken transmissions; the protocol logic is identical either way.
- **Both output WAVs share one master clock**, so `radio_A.wav` and
  `radio_B.wav` stay synchronised and can be analysed independently or
  together (`combined_mix.wav`).
- **18 automated checks** (`--test`) cover the collision/ACK/retry logic
  without needing to listen to any audio.

## PlutoSDR hardware link

`pluto_link.py` and `net_transmit.py` put the generated audio through two
PlutoSDR units over a cable — real NBFM modulation and a real receive path,
with **nothing radiating**:

```text
Pluto A  TX ──[ 30–40 dB attenuator ]── Pluto B  RX
```

The attenuator is not optional: Pluto transmits at roughly 0 to +7 dBm and
its receiver is rated to about −10 dBm, so a direct TX→RX connection damages
the front end. Unused ports are terminated with a 50 Ω load. The default
carrier (2.4 GHz) is only a carrier for the cable — it isn't intended to
radiate, and the tone test is run with the cable connected, then again with
it disconnected, to confirm the level collapses (i.e. nothing is escaping
some other way).

```bash
python pluto_link.py --list                # find attached Plutos
python pluto_link.py --tone --loopback     # one board, TX to its own RX
python pluto_link.py --tone                # both boards, through the cable
python pluto_link.py --send session/radio_A.wav --receive rx.wav
```

`net_transmit.py` walks a built session's timeline and keys whichever radio
is due to talk (rather than sending one station's entire file, then the
other's), so a third receiver on the same frequency hears an actual
back-and-forth. It has a `--dry-run` mode that prints timing/ordering with no
hardware involved, and it refuses to run on a session whose text isn't
properly exercise-marked unless overridden with `--unmarked`.

## On transmitting

Everything in this repository produces **audio files**. Whether that audio
is ever put on an actual antenna is an operational decision, not a software
one, and depends on the operator having:

- a frequency assignment held by the operating organisation for exercise use,
- exercise control — someone able to stop the exercise immediately and aware
  of what's on adjacent channels, and
- a defined exercise area with a known receiving population.

Without those in place, output is used as files, over headsets, or through a
conducted loop (transmitter → attenuator → receiver, all coax, no antenna) —
which exercises every part of the system except the final radiating hop, and
is standard practice for integration-testing radio systems. Content generated
by this project resembles realistic contact reports and casualty-evacuation
requests — exactly the category of traffic a receiving station would act on —
which is why exercise marking is on by default and why the control measures
above matter more than the software itself.

## Reproducibility & current limitations

- **Requires local compute for TTS.** Voice cloning via Chatterbox is CPU/GPU
  intensive; the DSP and simulator paths (`radio_net.py`, `step5_radio_fx.py`)
  do not require it.
- **Requires Ollama for scenario generation.** `step1_generate.py` calls a
  local Ollama server; without it, the curation/render/DSP/session stages
  still work on hand-written or previously generated exchanges.
- **PlutoSDR steps require the actual hardware** (two PlutoSDR units, cable,
  attenuator) and cannot be verified without it — `pluto_link.py --tone
  --loopback` and `net_transmit.py --dry-run` are the parts that can still be
  checked without hardware.
- **Reference voices are not included**, so a fresh clone will render every
  station in Chatterbox's default voice until you add your own `refs/`
  recordings — this is a deliberate exclusion (see table above), not a bug.
- _[Author to confirm]_: which parts of this pipeline have been run
  end-to-end most recently, and on what OS/Python version, so a reviewer
  knows what "known good" reproduction looks like.

## Future improvements

- Package the pipeline steps as an installable CLI (e.g. `net-control
  generate`, `net-control build`) instead of separate scripts.
- Add unit tests for the DSP chain (`step5_radio_fx.py`) alongside the
  existing `radio_net.py --test` protocol suite.
- Containerise the Ollama + Chatterbox dependencies so setup doesn't require
  installing a local LLM runtime by hand.
- Extend the two-radio simulator to N stations on a shared channel.

## Author

**[Your Name]**
_[Add: your program/field of study or role, and how this project relates to it —
e.g. "built while exploring signal processing, applied ML, and RF systems
ahead of applying to [program]."]_

- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com
