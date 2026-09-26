"""
pluto_link.py — put your generated audio through PlutoSDR hardware.

WHAT THIS IS

A conducted link. Transmitter into an attenuator into the receiver, all
coax, no antennas. Every part of the chain is real — modulation, timing,
the receive path — except the final radiating hop.

That is deliberate, and it is not a lesser version of the test. Conducted
work is repeatable, cannot interfere with anything, needs no coordination,
and can be demonstrated anywhere. It is how multi-node radio systems get
integration-tested in the first place.

---------------------------------------------------------------------------
WIRING — read this before plugging anything in

    Pluto A  TX  ──[ 30-40 dB attenuator ]──  Pluto B  RX

Pluto transmits at roughly 0 to +7 dBm. Its receiver is happy up to about
-10 dBm. Connecting TX straight to RX will damage the front end.

The attenuator is not optional. Start with more than you think you need —
you can always take some out, but you cannot un-damage a receiver.

Terminate any unused port with a 50 ohm load.

---------------------------------------------------------------------------
FREQUENCY

Nothing here radiates, so the frequency is only a carrier for the cable.
2.4 GHz is used by default because it sits comfortably inside Pluto's
specified 325 MHz - 3.8 GHz range and away from anything you might care
about.

If you later have an assignment and authority to radiate, that is a
different conversation and a different setup. This file stays on the cable.

---------------------------------------------------------------------------
USAGE

    python pluto_link.py --list                 find your Plutos
    python pluto_link.py --loopback             one Pluto, TX to its own RX
    python pluto_link.py --tone                 carrier test, no audio
    python pluto_link.py --send session/radio_A.wav
    python pluto_link.py --receive out.wav --seconds 30

    python pluto_link.py --send session/radio_A.wav --tx ip:192.168.2.1 \\
                         --receive rx.wav --rx ip:192.168.3.1

Needs:
    pip install pyadi-iio numpy scipy soundfile
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("soundfile is needed:  pip install soundfile")
    sys.exit(1)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

DEFAULT_TX_URI = "ip:192.168.2.1"
DEFAULT_RX_URI = "ip:192.168.3.1"   # second Pluto, once you have renumbered it

CARRIER_HZ   = 2_400_000_000        # cable only — see the note above
SAMPLE_RATE  = 1_000_000            # 1 MS/s, comfortable for Pluto over USB
AUDIO_RATE   = 24_000               # what the pipeline produces
DEVIATION_HZ = 2_500                # NBFM, matching 12.5 kHz channel spacing

TX_GAIN = -20     # dB of attenuation. Less negative = more power.
RX_GAIN = 30      # dB. Manual gain so levels are repeatable between runs.


# ===========================================================================
# FINDING THE HARDWARE
# ===========================================================================

def need_adi():
    try:
        import adi
        return adi
    except ImportError:
        print()
        print("  pyadi-iio is not installed.")
        print()
        print("    pip install pyadi-iio")
        print()
        print("  You also need the Analog Devices USB drivers. On Windows")
        print("  install the PlutoSDR driver package from analog.com first,")
        print("  then reconnect the board.")
        sys.exit(1)


def list_devices():
    """Find every Pluto that answers."""
    adi = need_adi()
    print()
    print("  Looking for PlutoSDRs...")
    print()

    found = []
    seen = {}
    for uri in ("ip:192.168.2.1", "ip:192.168.3.1", "ip:pluto.local"):
        try:
            dev = adi.Pluto(uri)
            sr = dev.sample_rate
            lo = dev.rx_lo
            key = (sr, lo)
            if key in seen:
                print(f"  --     {uri}  (same board as {seen[key]})")
            else:
                seen[key] = uri
                print(f"  FOUND  {uri}")
                print(f"           sample rate {sr:,} Hz")
                print(f"           rx tuned to  {lo/1e6:.1f} MHz")
                found.append(uri)
            del dev
        except Exception:
            print(f"  --     {uri}")

    print()
    if len(found) == 1:
        print(f"  One board responding, on {found[0]}.")
        print()
        print("  For a two-radio link the second must be renumbered onto its")
        print("  own subnet. Plug it in alone, edit config.txt on the drive:")
        print()
        print("      [NETWORK]")
        print("      ipaddr = 192.168.2.1")
        print("      ipaddr_host = 192.168.2.10")
        print("      netmask = 255.255.255.0")
        print()
        print("  Eject the drive properly before unplugging, or the change is")
        print("  lost. Single-board loopback works now without this.")
        print()
    if not found:
        print("  Nothing answered.")
        print()
        print("  Check: the board is plugged in, the green Ready light is on,")
        print("  and it appears as a network device. Default address is")
        print("  192.168.2.1 — try opening http://192.168.2.1 in a browser.")
        print()
        print("  Two Plutos on one machine will both claim 192.168.2.1. You")
        print("  must renumber the second one — see the notes at the bottom")
        print("  of this file.")
    else:
        print(f"  {len(found)} device(s) responding.")
    return found


# ===========================================================================
# MODULATION
# ===========================================================================

def load_audio(path, target_rate=AUDIO_RATE):
    """Read a WAV as mono floats at the working rate."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_rate:
        n = int(len(audio) * target_rate / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio / peak * 0.9
    return audio.astype(np.float64)


def preemphasis(x, coeff=0.95):
    """Lift the high frequencies before transmitting.

    FM noise rises with frequency, so voice radio boosts the top end going
    in and cuts it coming out. The result is the same audio with less hiss.
    Every real NBFM set does this; leaving it out is audible.
    """
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]
    peak = np.max(np.abs(y))
    return y / peak * 0.9 if peak > 1e-9 else y


def deemphasis(x, coeff=0.95):
    """Undo the lift on the way out. Must match the transmit side.

    Deliberately does not normalize its own output — nbfm_demodulate()
    does one normalization pass, after this, against a level that ignores
    outlier spikes. Normalizing here too would just be a second, less
    robust vote on the same decision.
    """
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = x[i] + coeff * acc
        y[i] = acc
    return y


def nbfm_modulate(audio, audio_rate=AUDIO_RATE, sample_rate=SAMPLE_RATE,
                  deviation=DEVIATION_HZ, emphasis=True):
    """Narrowband FM, which is what voice radio uses.

    The audio is already band-limited to roughly 300-3400 Hz by the DSP
    chain, so it needs no further filtering — it is the right shape for a
    12.5 kHz channel as it stands.
    """
    if emphasis:
        audio = preemphasis(audio)
    # Up to the radio's sample rate
    factor = sample_rate / audio_rate
    n = int(len(audio) * factor)
    up = np.interp(np.linspace(0, len(audio) - 1, n),
                   np.arange(len(audio)), audio)

    # FM is phase proportional to the integral of the message
    k = 2 * np.pi * deviation / sample_rate
    phase = np.cumsum(up) * k

    iq = np.exp(1j * phase)
    return (iq * (2 ** 14)).astype(np.complex64)   # Pluto expects this scale


def estimate_size(seconds, sample_rate=SAMPLE_RATE):
    """Complex64 is 8 bytes a sample, and a megasample a second adds up
    fast. Worth knowing before you try to modulate an hour of audio."""
    return seconds * sample_rate * 8 / (1024 ** 3)   # GiB


def channel_filter(iq, sample_rate=SAMPLE_RATE, bandwidth=12_500,
                   block=1 << 18):
    """Keep only the channel, throw the rest of the band away.

    Filtered in overlapping blocks rather than in one go. A minute of
    audio at 1 MS/s is sixty million complex samples; taking an FFT of
    that at once needs the best part of a gigabyte and will simply be
    killed. Blocks with an overlap-and-crossfade give the same result at
    a fixed memory cost.
    """
    n = len(iq)
    if n <= block:
        return _filter_block(iq, sample_rate, bandwidth)

    overlap = block // 8
    out = np.zeros(n, dtype=np.complex64)
    pos = 0

    while pos < n:
        end = min(pos + block, n)
        lo = max(0, pos - overlap)
        hi = min(n, end + overlap)

        chunk = _filter_block(iq[lo:hi], sample_rate, bandwidth)
        out[pos:end] = chunk[pos - lo:pos - lo + (end - pos)]
        pos = end

    return out


def _filter_block(iq, sample_rate, bandwidth):
    n = len(iq)
    spec = np.fft.fft(iq)
    freqs = np.fft.fftfreq(n, 1.0 / sample_rate)

    keep = np.abs(freqs) <= bandwidth / 2.0
    # Soften the edges so the filter does not ring
    edge = np.abs(freqs) - bandwidth / 2.0
    taper = np.clip(1.0 - edge / (bandwidth * 0.25), 0.0, 1.0)
    mask = np.where(keep, 1.0, taper)

    return np.fft.ifft(spec * mask).astype(np.complex64)


def declick_join(audio, audio_rate=AUDIO_RATE, ms=3):
    """Fade in the first few milliseconds of a chunk that continues a
    transmission already in progress.

    Each independent nbfm_modulate() call resets the preemphasis filter
    and restarts the FM phase integration from zero, so splicing two
    separately modulated chunks back-to-back — as a long transmission
    processed in bounded pieces has to — produces a brief click right at
    the join, similar in kind to (though smaller than) the keying
    transient at a real start/end. A few milliseconds is far shorter than
    any actual syllable, so fading it is inaudible as a dip but removes
    a click that would otherwise be the loudest thing in the chunk and
    throw off finish_demodulated()'s normalization. Not the same thing as
    the true start/end edge handling in finish_demodulated() — this is
    for joins that aren't a real key-up/key-down at all.
    """
    audio = np.asarray(audio, dtype=np.float64).copy()
    n = min(int(ms / 1000 * audio_rate), len(audio) // 4)
    if n > 1:
        audio[:n] *= np.linspace(0, 1, n)
    return audio


def nbfm_demodulate_raw(iq, sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE,
                        emphasis=True):
    """The demodulation math only — channel filter, angle difference,
    downsample, deemphasis. No click suppression, no normalization.

    Deliberately separate from nbfm_demodulate() so a long transmission can
    be received and demodulated one bounded chunk of raw IQ at a time —
    each chunk's audio-rate output is tiny even when the raw IQ behind it
    isn't — with finish_demodulated() applied once at the end over the
    whole assembled recording. Calling this directly on an internal chunk
    boundary is fine: unlike a real key-up/key-down, there's no genuine
    click to suppress there, so skipping that step is correct, not a
    shortcut.
    """
    if len(iq) < 2:
        return np.zeros(0)

    # Filter to the channel before demodulating.
    #
    # This matters more than it looks. The radio hands us the full sample
    # rate — 1 MHz here — but the signal only occupies about 12.5 kHz. Every
    # Hz of that extra bandwidth is noise going into the demodulator for no
    # benefit. A real receiver has a channel filter for exactly this reason,
    # and without it the link falls apart at signal levels that should be
    # perfectly usable.
    iq = channel_filter(iq, sample_rate)

    demod = np.angle(iq[1:] * np.conj(iq[:-1]))

    # Down to audio rate
    n = int(len(demod) * audio_rate / sample_rate)
    if n < 2:
        return np.zeros(0)
    audio = np.interp(np.linspace(0, len(demod) - 1, n),
                      np.arange(len(demod)), demod)

    if emphasis:
        audio = deemphasis(audio)
    return audio


def finish_demodulated(audio, audio_rate=AUDIO_RATE):
    """The click suppression and normalization that used to live inside
    nbfm_demodulate() directly. Call this once, on the complete recording
    — whether it came from a single nbfm_demodulate_raw() call or several
    concatenated ones — never per internal chunk.
    """
    audio = np.asarray(audio, dtype=np.float64).copy()
    if len(audio) == 0:
        return audio

    # The carrier keying on and off produces a brief, sharp click at each
    # end — a real discontinuity, not noise, and much louder than any
    # speech. Left in, it becomes the loudest thing in the recording and
    # throws off the normalization below, making genuine speech land well
    # under full scale even on a link that's working properly. A short
    # fade removes the click itself; trimming a few extra milliseconds off
    # each end is cheap insurance against whatever's left of the edge
    # transient bleeding past the fade.
    edge = min(int(0.015 * audio_rate), len(audio) // 4)   # ~15 ms
    if edge > 1:
        ramp = np.linspace(0, 1, edge)
        audio[:edge] *= ramp
        audio[-edge:] *= ramp[::-1]

    # Normalized against a high percentile rather than the true peak, so a
    # single leftover spike (a residual click, a brief fade, a moment of
    # channel noise) can't single-handedly set the reference level and
    # make every actual word in the recording sound quiet by comparison.
    ref = np.percentile(np.abs(audio), 99)
    if ref > 1e-9:
        audio = np.clip(audio / ref * 0.9, -1.0, 1.0)
    return audio


def nbfm_demodulate(iq, sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE,
                    emphasis=True):
    """Recover the audio in one call: the demodulation math, then the
    click suppression and normalization, over the whole thing.

    This is nbfm_demodulate_raw() followed by finish_demodulated() — kept
    as a single function for every existing caller that already has the
    whole recording's IQ in memory at once (a single clip, a loopback
    test). For a transmission too long to hold as one IQ array, use the
    two pieces directly instead — see transmit_file() in server.py.
    """
    return finish_demodulated(
        nbfm_demodulate_raw(iq, sample_rate, audio_rate, emphasis),
        audio_rate)


def make_tone(seconds=5.0, hz=1000.0):
    """A plain tone, for proving the link before involving your audio."""
    t = np.arange(int(seconds * AUDIO_RATE)) / AUDIO_RATE
    return 0.7 * np.sin(2 * np.pi * hz * t)


# ===========================================================================
# TRANSMIT AND RECEIVE
# ===========================================================================

def open_tx(uri, carrier=CARRIER_HZ):
    adi = need_adi()
    sdr = adi.Pluto(uri)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_lo = int(carrier)
    sdr.tx_hardwaregain_chan0 = int(TX_GAIN)
    sdr.tx_cyclic_buffer = False

    # Pluto sets its transmit buffer from the first block it is given and
    # cannot change it afterwards, so the block size is fixed here.
    try:
        sdr.tx_buffer_size = 1 << 16
    except Exception:
        pass          # older pyadi-iio versions set this implicitly

    return sdr


def open_rx(uri, carrier=CARRIER_HZ):
    adi = need_adi()
    sdr = adi.Pluto(uri)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_lo = int(carrier)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = int(RX_GAIN)
    sdr.rx_buffer_size = 1 << 16
    return sdr


def transmit(sdr, iq, chunk=1 << 16, label=""):
    """Send the samples in blocks.

    Pluto fixes its transmit buffer on the first tx() call and cannot resize
    it afterwards. Every block must therefore be exactly the same length, so
    the tail is zero-padded up to a full block rather than sent short.

    Padding with zeros is correct here: an FM transmitter sending zero
    amplitude is simply an unmodulated carrier for those few milliseconds,
    which is what happens at the end of a real transmission anyway.
    """
    total = len(iq)
    if total == 0:
        return

    # One partial block at the end, padded to full length
    remainder = total % chunk
    if remainder:
        pad = np.zeros(chunk - remainder, dtype=np.complex64)
        iq = np.concatenate([iq, pad])

    sent = 0
    started = time.time()

    print(f"  transmitting {total/SAMPLE_RATE:.1f}s {label}")
    while sent < len(iq):
        sdr.tx(iq[sent:sent + chunk])
        sent += chunk
        print(f"\r    {min(sent, total) / total * 100:5.1f}%", end="", flush=True)

    print(f"\r    done in {time.time() - started:.1f}s      ")


def receive(sdr, seconds):
    """Collect samples for a while and return them as one array."""
    need = int(seconds * SAMPLE_RATE)
    parts = []
    got = 0
    started = time.time()

    print(f"  receiving {seconds:.0f}s")
    while got < need:
        block = sdr.rx()
        parts.append(block)
        got += len(block)
        print(f"\r    {got/need*100:5.1f}%", end="", flush=True)

    print(f"\r    done in {time.time() - started:.1f}s")
    return np.concatenate(parts)[:need]


def signal_report(iq):
    """Is anything actually arriving, and at a sensible level."""
    mag = np.abs(iq)
    rms = float(np.sqrt(np.mean(mag ** 2)))
    peak = float(np.max(mag))

    print()
    print(f"  received level   rms {rms:8.1f}   peak {peak:8.1f}")

    if rms < 20:
        print("  Very weak. Check the cable, and that the transmitter is")
        print("  actually running. Try less attenuation, or raise RX_GAIN.")
    elif peak > 2000:
        print("  Very strong — likely clipping. Add more attenuation, or")
        print("  lower RX_GAIN. Do not leave it like this.")
    else:
        print("  Level looks reasonable.")
    return rms


# ===========================================================================
# THE TESTS, IN THE ORDER WORTH DOING THEM
# ===========================================================================

def test_tone(tx_uri, rx_uri, seconds=5.0):
    """Step one. Prove the hardware, the cable and the attenuator before
    anything else is involved. If this does not work, nothing else will,
    and you will not know why."""
    print()
    print("  CARRIER TEST — a plain tone, no audio")
    print()

    tone = make_tone(seconds)
    iq = nbfm_modulate(tone)

    tx = open_tx(tx_uri)
    rx = open_rx(rx_uri)

    print(f"  tx {tx_uri}  ->  rx {rx_uri}")
    print(f"  carrier {CARRIER_HZ/1e6:.0f} MHz   tx gain {TX_GAIN} dB   "
          f"rx gain {RX_GAIN} dB")
    print()

    # Prime the receiver, then transmit while collecting
    import threading
    got = {}

    def grab():
        # Listen a little longer than the transmission, not shorter — the
        # receiver finishing first is what truncates the recording.
        got["iq"] = receive(rx, seconds + 0.5)

    t = threading.Thread(target=grab)
    t.start()
    time.sleep(0.3)
    transmit(tx, iq, label="tone")
    t.join()

    rms = signal_report(got.get("iq", np.zeros(0)))

    audio = nbfm_demodulate(got.get("iq", np.zeros(0)))
    if len(audio):
        sf.write("tone_received.wav", audio, AUDIO_RATE)
        print()
        print("  wrote tone_received.wav — it should be a steady 1 kHz tone")

    print()
    print("  NOW UNPLUG THE CABLE and run this again.")
    print("  The level should collapse. If it does not, energy is getting")
    print("  out some other way and you need to find it before going on.")

    del tx, rx
    return rms


def send_file(path, tx_uri, rx_uri=None, out_path=None):
    """Put a WAV through the link."""
    print()
    print(f"  SENDING {path}")
    print()

    audio = load_audio(path)
    dur = len(audio) / AUDIO_RATE
    print(f"  {dur:.1f}s of audio")

    gib = estimate_size(dur)
    if gib > 1.0:
        print()
        print(f"  That is {gib:.1f} GiB of IQ at {SAMPLE_RATE/1e6:.1f} MS/s.")
        print("  Lower SAMPLE_RATE, or send a shorter file. Most of a session")
        print("  is silence anyway — the individual clips are a better test.")
        print()
        return

    iq = nbfm_modulate(audio)
    print(f"  {len(iq):,} samples at {SAMPLE_RATE:,} Hz")

    tx = open_tx(tx_uri)

    if rx_uri:
        rx = open_rx(rx_uri)
        import threading
        got = {}

        def grab():
            got["iq"] = receive(rx, dur + 0.5)

        t = threading.Thread(target=grab)
        t.start()
        time.sleep(0.3)
        transmit(tx, iq, label=path)
        t.join()

        signal_report(got.get("iq", np.zeros(0)))

        rec = nbfm_demodulate(got.get("iq", np.zeros(0)))
        out = out_path or "received.wav"
        sf.write(out, rec, AUDIO_RATE)
        print()
        print(f"  wrote {out}")
        print("  Play it against the original. It should sound the same,")
        print("  with a little added noise from the link.")
        del rx
    else:
        transmit(tx, iq, label=path)

    del tx


def receive_only(rx_uri, seconds, out_path):
    print()
    print(f"  RECEIVING for {seconds:.0f}s")
    print()
    rx = open_rx(rx_uri)
    iq = receive(rx, seconds)
    signal_report(iq)

    audio = nbfm_demodulate(iq)
    sf.write(out_path, audio, AUDIO_RATE)
    print()
    print(f"  wrote {out_path}")
    del rx


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Put generated audio through two PlutoSDRs, over cable.")
    ap.add_argument("--list", action="store_true", help="find connected Plutos")
    ap.add_argument("--tone", action="store_true",
                    help="carrier test — do this first")
    ap.add_argument("--loopback", action="store_true",
                    help="one Pluto, its own TX to its own RX")
    ap.add_argument("--send", metavar="WAV", help="transmit a WAV file")
    ap.add_argument("--receive", metavar="WAV", help="write what is received here")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="how long to receive for")
    ap.add_argument("--tx", default=DEFAULT_TX_URI, help="transmitter address")
    ap.add_argument("--rx", default=DEFAULT_RX_URI, help="receiver address")
    ap.add_argument("--carrier", type=float, default=None,
                    help="carrier in MHz")
    ap.add_argument("--gain", type=float, default=None,
                    help="transmit attenuation in dB, default -20. More "
                         "negative is quieter. Use -70 for radiated testing.")
    ap.add_argument("--rx-gain", type=float, default=None,
                    help="receive gain in dB, default 30")
    args = ap.parse_args()

    global CARRIER_HZ, TX_GAIN, RX_GAIN
    if args.carrier:
        CARRIER_HZ = int(args.carrier * 1e6)
    if args.gain is not None:
        TX_GAIN = args.gain
    if args.rx_gain is not None:
        RX_GAIN = args.rx_gain

    print()
    print("=" * 62)
    print("  PLUTO LINK")
    print("=" * 62)
    print()
    print(f"  carrier {CARRIER_HZ/1e6:.1f} MHz   tx {TX_GAIN} dB   rx {RX_GAIN} dB")
    print()
    print("  Conducted: TX --[ 30-40 dB attenuator ]-- RX, no antennas.")
    print("  Radiated:  antennas about a metre apart, tx gain -70 or lower.")
    print()

    if args.list:
        list_devices()
        return

    tx_uri = args.tx
    rx_uri = args.tx if args.loopback else args.rx

    if args.tone:
        test_tone(tx_uri, rx_uri)
        return

    if args.send:
        send_file(args.send, tx_uri, rx_uri if args.receive or args.loopback
                  else None, args.receive)
        return

    if args.receive:
        receive_only(rx_uri, args.seconds, args.receive)
        return

    print("  Nothing to do. Try:")
    print()
    print("    python pluto_link.py --list")
    print("    python pluto_link.py --tone --loopback")
    print("    python pluto_link.py --send session/radio_A.wav --receive rx.wav")
    print()


if __name__ == "__main__":
    main()


# ===========================================================================
# TWO PLUTOS ON ONE MACHINE
# ---------------------------------------------------------------------------
# Both boards ship claiming 192.168.2.1, so the second one has to be
# renumbered before the machine can see them at once.
#
# Plug in ONE board. It appears as a small USB drive with a file called
# config.txt on it. Open that file and change:
#
#     [USB_ETHERNET]
#     ipaddr = 192.168.3.1
#     ipaddr_host = 192.168.3.10
#
# Save it, eject the drive properly, then unplug and replug the board. It
# now answers on 192.168.3.1 and both can be connected together.
#
# Check with:  python pluto_link.py --list
# ===========================================================================
