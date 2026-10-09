"""Play an audio file and stream its stored features over OSC, in time with playback.

Usage:
    uv run python stream_features.py audio/track1.wav features.npz
    uv run python stream_features.py audio/track1.wav features.npz --classes Music Drum Singing
    uv run python stream_features.py audio/track1.wav features.npz --no-audio   # features only

OSC addresses (become channel names in TouchDesigner's OSC In CHOP):
    /rms /centroid /flatness /onset   handcrafted, normalised to 0..1 (5th-95th percentile)
    /<ClassName>                      learned class score (0..1), spaces become underscores
    /time                             playback time in seconds
"""
import argparse
import time

import librosa
import numpy as np
from pythonosc.udp_client import SimpleUDPClient


def normalise(x, lo=5, hi=95):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio")
    p.add_argument("features")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7000)
    p.add_argument("--rate", type=float, default=60.0, help="OSC messages per second")
    p.add_argument("--classes", nargs="*", default=None, help="class names to send")
    p.add_argument("--no-audio", action="store_true", help="do not play audio, only send features")
    p.add_argument("--loop", action="store_true")
    a = p.parse_args()

    d = np.load(a.features, allow_pickle=True)
    times = d["times"]
    hand = {k: normalise(d[k]) for k in ("rms", "centroid", "flatness", "onset")}

    l_times = d["learned_times"]
    scores = d["learned_scores"]
    names = [str(n) for n in d["class_names"]]
    chosen = {}
    if len(l_times) and len(names):
        wanted = a.classes if a.classes else [n for n in ("Music", "Speech", "Drum", "Singing", "Silence") if n in names]
        for n in wanted:
            if n in names:
                chosen[n] = scores[:, names.index(n)]
            else:
                print(f"[warn] class not found: {n!r}")
    elif a.classes:
        print("[warn] this features file has no learned scores; only handcrafted features will be sent")

    client = SimpleUDPClient(a.host, a.port)
    duration = times[-1]
    y = None
    if not a.no_audio:
        import sounddevice as sd
        y, sr = librosa.load(a.audio, sr=None, mono=True)

    print(f"Streaming to {a.host}:{a.port}. Handcrafted: {list(hand)}. Learned: {list(chosen)}. Ctrl+C to stop.")
    try:
        while True:
            if y is not None:
                sd.play(y, sr)
            t0 = time.perf_counter()
            dt = 1.0 / a.rate
            k = 0
            while True:
                t = time.perf_counter() - t0
                if t > duration:
                    break
                for name, v in hand.items():
                    client.send_message(f"/{name}", float(np.interp(t, times, v)))
                for name, v in chosen.items():
                    client.send_message("/" + name.replace(" ", "_"), float(np.interp(t, l_times, v)))
                client.send_message("/time", float(t))
                k += 1
                time.sleep(max(0.0, t0 + k * dt - time.perf_counter()))
            if not a.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if y is not None:
            sd.stop()
    print("Done.")


if __name__ == "__main__":
    main()
