"""
step5_radio_fx.py — turn clean TTS output into believable radio-net audio.

This is the piece that actually sells the illusion. Clean synthesized speech
never sounds like radio; speech pushed through a narrow band, hard-compressed,
saturated, and buried in a noise floor does — even if the underlying TTS is
mediocre.

Signal chain, in order:
    1. Band-limit         (300–3400 Hz, the classic comms passband)
    2. Saturate           (transmitter/receiver nonlinearity)
    3. Compress           (heavy limiting — radio has almost no dynamic range)
    4. Fading / wobble    (slow amplitude drift, signal strength variation)
    5. Noise floor        (band-limited hiss, scaled to a target SNR)
    6. Key-up click + squelch tail

Dependencies:
    pip install numpy scipy soundfile

Usage:
    python step5_radio_fx.py in.wav out.wav --profile handheld
    python step5_radio_fx.py --net clips/ net_output.wav
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal


# --------------------------------------------------------------------------
# Profiles — different radios sound different. Assign one per callsign so
# ALPHA-1 (base station, strong signal) and BRAVO-6 (handheld, 3 km out,
# behind a ridge) are instantly distinguishable by ear.
# --------------------------------------------------------------------------

@dataclass
class RadioProfile:
    name: str = "generic"
    low_hz: float = 300.0        # passband low edge
    high_hz: float = 3400.0      # passband high edge
    drive: float = 3.0           # saturation amount, 1.0 = clean
    threshold_db: float = -22.0  # compressor threshold
    ratio: float = 8.0           # compressor ratio
    snr_db: float = 22.0         # signal-to-noise; lower = more hiss
    fade_depth: float = 0.10     # 0.0 = rock steady, 0.4 = badly fading
    fade_rate_hz: float = 0.7    # how fast the signal strength drifts
    click_level: float = 0.30    # key-up pop amplitude
    tail_ms: float = 180.0       # squelch tail length after transmission
    dropout_prob: float = 0.0    # chance per transmission of a brief cutout


PROFILES = {
    # Base station / vehicle mount — strong, clean-ish, authoritative
    "base": RadioProfile(
        name="base", low_hz=280, high_hz=3600, drive=2.0,
        threshold_db=-20, ratio=6.0, snr_db=30.0,
        fade_depth=0.04, tail_ms=140, dropout_prob=0.0,
    ),
    # Standard handheld at moderate range — the default patrol sound
    "handheld": RadioProfile(
        name="handheld", low_hz=350, high_hz=3200, drive=3.5,
        threshold_db=-24, ratio=9.0, snr_db=20.0,
        fade_depth=0.12, tail_ms=200, dropout_prob=0.08,
    ),
    # Distant / obstructed — weak, noisy, drifting. Use sparingly; one unit
    # sounding rough makes the others sound correspondingly solid.
    "distant": RadioProfile(
        name="distant", low_hz=420, high_hz=2800, drive=5.0,
        threshold_db=-28, ratio=12.0, snr_db=11.0,
        fade_depth=0.30, fade_rate_hz=1.1, tail_ms=260, dropout_prob=0.25,
    ),
    # Aircraft — brighter, cleaner, but with prop/rotor rumble underneath
    "air": RadioProfile(
        name="air", low_hz=300, high_hz=3800, drive=2.5,
        threshold_db=-22, ratio=7.0, snr_db=17.0,
        fade_depth=0.06, tail_ms=120, dropout_prob=0.02,
    ),
}


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def bandpass(x: np.ndarray, sr: int, low: float, high: float,
             order: int = 4) -> np.ndarray:
    """Restrict to the comms passband. This single step does more for
    realism than anything else in the chain."""
    nyq = sr / 2.0
    high = min(high, nyq * 0.98)
    sos = signal.butter(order, [low / nyq, high / nyq], btype="band",
                        output="sos")
    return signal.sosfilt(sos, x)


def saturate(x: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """Soft clipping. Adds the slightly harsh, 'pushed' quality of a
    transmitter running into its limits."""
    if drive <= 1.0:
        return x
    return np.tanh(x * drive) / np.tanh(drive)


def compress(x: np.ndarray, sr: int, threshold_db: float = -22.0,
             ratio: float = 8.0, attack_ms: float = 2.0,
             release_ms: float = 60.0) -> np.ndarray:
    """Envelope-following compressor. Radio traffic has almost no dynamic
    range — quiet syllables come through at nearly the same level as loud
    ones. Heavy ratios (6–12) are correct here, not excessive.

    Note: this uses a per-sample loop for the asymmetric attack/release.
    Fine for the few-second clips this project produces; if you ever push
    minutes of audio through it, vectorize or drop in numba.
    """
    eps = 1e-9
    env = np.abs(x)
    a_att = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    a_rel = np.exp(-1.0 / (sr * release_ms / 1000.0))

    follower = np.empty_like(env)
    prev = 0.0
    for i in range(env.size):
        v = env[i]
        coef = a_att if v > prev else a_rel
        prev = coef * prev + (1.0 - coef) * v
        follower[i] = prev

    env_db = 20.0 * np.log10(follower + eps)
    over_db = np.maximum(env_db - threshold_db, 0.0)
    gain_db = -over_db * (1.0 - 1.0 / ratio)

    # Makeup gain so the result sits at a consistent level
    makeup_db = -threshold_db * (1.0 - 1.0 / ratio) * 0.6
    return x * (10.0 ** ((gain_db + makeup_db) / 20.0))


def fading(x: np.ndarray, sr: int, depth: float = 0.12,
           rate_hz: float = 0.7, rng: np.random.Generator | None = None
           ) -> np.ndarray:
    """Slow amplitude drift — the signal strength wandering as an antenna
    moves or the path changes. Subtle, but its absence is what makes clean
    audio read as 'recording' rather than 'transmission'."""
    if depth <= 0.0:
        return x
    rng = rng or np.random.default_rng()
    n = x.size
    # Low-frequency noise as the modulator, so drift is irregular not sinusoidal
    raw = rng.standard_normal(n)
    sos = signal.butter(2, max(rate_hz, 0.05) / (sr / 2.0), btype="low",
                        output="sos")
    mod = signal.sosfilt(sos, raw)
    mod = mod / (np.max(np.abs(mod)) + 1e-9)
    return x * (1.0 - depth + depth * (0.5 + 0.5 * mod))


def dropout(x: np.ndarray, sr: int, prob: float,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Occasional brief cutout mid-transmission. Use at low probability —
    one dropout per several transmissions reads as realistic, more reads
    as broken."""
    if prob <= 0.0:
        return x
    rng = rng or np.random.default_rng()
    if rng.random() > prob:
        return x
    x = x.copy()
    dur = int(sr * rng.uniform(0.04, 0.14))
    if dur >= x.size:
        return x
    start = rng.integers(0, x.size - dur)
    ramp = int(sr * 0.005)
    x[start:start + dur] *= 0.05
    # soften the edges so it sounds like signal loss, not an edit
    if ramp > 0 and start > ramp:
        x[start - ramp:start] *= np.linspace(1.0, 0.05, ramp)
        end = start + dur
        if end + ramp < x.size:
            x[end:end + ramp] *= np.linspace(0.05, 1.0, ramp)
    return x


def band_noise(n: int, sr: int, low: float, high: float,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Hiss shaped by the same passband as the speech — receiver noise is
    filtered by the receiver, so unfiltered white noise sounds wrong."""
    rng = rng or np.random.default_rng()
    return bandpass(rng.standard_normal(n), sr, low, high)


def key_click(sr: int, level: float = 0.3,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """The pop of a PTT switch being pressed. Very short, quite loud."""
    rng = rng or np.random.default_rng()
    n = int(sr * 0.012)
    env = np.exp(-np.linspace(0, 9, n))
    return rng.standard_normal(n) * env * level


def squelch_tail(sr: int, ms: float = 180.0, level: float = 0.25,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """The burst of noise after the carrier drops and before squelch
    closes. Every listener recognizes this sound even if they can't name
    it — it's the strongest single cue that something is a radio."""
    rng = rng or np.random.default_rng()
    n = int(sr * ms / 1000.0)
    env = np.concatenate([
        np.ones(int(n * 0.6)),
        np.linspace(1.0, 0.0, n - int(n * 0.6)),
    ])
    noise = bandpass(rng.standard_normal(n), sr, 400, 3200)
    noise = noise / (np.max(np.abs(noise)) + 1e-9)
    return noise * env * level


def normalize(x: np.ndarray, peak: float = 0.92) -> np.ndarray:
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 1e-9 else x


# --------------------------------------------------------------------------
# Full chain
# --------------------------------------------------------------------------

def process(x: np.ndarray, sr: int, profile: RadioProfile,
            seed: int | None = None) -> np.ndarray:
    """Run one transmission through the whole chain."""
    rng = np.random.default_rng(seed)
    x = _to_mono(np.asarray(x, dtype=np.float64))
    x = normalize(x, 0.9)

    x = bandpass(x, sr, profile.low_hz, profile.high_hz)
    x = saturate(x, profile.drive)
    x = compress(x, sr, profile.threshold_db, profile.ratio)
    x = fading(x, sr, profile.fade_depth, profile.fade_rate_hz, rng)
    x = dropout(x, sr, profile.dropout_prob, rng)
    x = normalize(x, 0.85)

    click = key_click(sr, profile.click_level, rng)
    tail = squelch_tail(sr, profile.tail_ms, rng=rng)
    gap = np.zeros(int(sr * 0.05))
    body = np.concatenate([click, gap, x, gap, tail])

    # Noise floor across the whole transmission, at the target SNR
    sig_rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    noise_rms = sig_rms / (10.0 ** (profile.snr_db / 20.0))
    noise = band_noise(body.size, sr, profile.low_hz, profile.high_hz, rng)
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-9) * noise_rms

    return normalize(body + noise, 0.92)


def build_net(clips: list[tuple[np.ndarray, RadioProfile]], sr: int,
              gap_range: tuple[float, float] = (0.5, 1.6),
              seed: int | None = None) -> np.ndarray:
    """Stitch processed transmissions into a continuous net recording with
    realistic dead air between them. Variable gaps matter — evenly spaced
    transmissions sound scripted."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = [np.zeros(int(sr * 0.4))]
    for audio, profile in clips:
        out.append(process(audio, sr, profile))
        out.append(np.zeros(int(sr * rng.uniform(*gap_range))))
    return normalize(np.concatenate(out), 0.92)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Apply radio effects to TTS audio.")
    ap.add_argument("input", help="input WAV file, or directory when --net is set")
    ap.add_argument("output", help="output WAV file")
    ap.add_argument("--profile", default="handheld", choices=sorted(PROFILES),
                    help="radio profile to apply (default: handheld)")
    ap.add_argument("--net", action="store_true",
                    help="treat input as a directory of clips and build a "
                         "continuous net recording, alternating profiles")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.net:
        paths = sorted(glob.glob(os.path.join(args.input, "*.wav")))
        if not paths:
            raise SystemExit(f"no .wav files found in {args.input}")
        clips, sr = [], None
        # Alternate base <-> handheld so the exchange sounds like two ends
        order = ["base", "handheld"]
        for i, p in enumerate(paths):
            audio, file_sr = sf.read(p)
            sr = sr or file_sr
            if file_sr != sr:
                raise SystemExit("all clips must share a sample rate")
            clips.append((audio, PROFILES[order[i % len(order)]]))
        out = build_net(clips, sr, seed=args.seed)
    else:
        audio, sr = sf.read(args.input)
        out = process(audio, sr, PROFILES[args.profile], seed=args.seed)

    sf.write(args.output, out, sr)
    print(f"wrote {args.output}  ({out.size / sr:.1f}s @ {sr} Hz)")


if __name__ == "__main__":
    main()