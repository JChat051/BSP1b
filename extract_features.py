"""Extract handcrafted and learned audio features and save them to a .npz file.

Usage:
    uv run python extract_features.py audio/track1.wav
    uv run python extract_features.py audio/track1.wav --model models/yamnet.tflite
    uv run python extract_features.py audio/track1.wav --no-learned

Saved keys:
    times, rms, centroid, flatness, onset   handcrafted, one value per frame
    learned_times (M,), learned_scores (M, C), class_names (C,)
    sr, hop
"""
import argparse
import time
from pathlib import Path

import librosa
import numpy as np


def handcrafted(y, sr, hop):
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = min(len(rms), len(centroid), len(flatness), len(onset))
    times = librosa.times_like(rms[:n], sr=sr, hop_length=hop)
    return times, rms[:n], centroid[:n], flatness[:n], onset[:n]


def learned(audio_path, model_path):
    """Run a pre-trained MediaPipe audio classifier (YAMNet-style) over the file.

    Returns (learned_times, learned_scores, class_names).
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import audio as mp_audio
    from mediapipe.tasks.python.components import containers

    model_path = str(model_path)
    y16, _ = librosa.load(audio_path, sr=16000, mono=True)  # YAMNet expects 16 kHz

    options = mp_audio.AudioClassifierOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_audio.RunningMode.AUDIO_CLIPS,
        max_results=-1,        # return every class
        score_threshold=0.0,
    )
    with mp_audio.AudioClassifier.create_from_options(options) as clf:
        clip = containers.AudioData.create_from_array(y16.astype(np.float32), 16000)
        results = clf.classify(clip)

    names = None
    times, rows = [], []
    for r in results:
        cats = r.classifications[0].categories
        if names is None:
            order = sorted(cats, key=lambda c: c.index)
            names = [c.category_name for c in order]
        row = np.zeros(len(names), dtype=np.float32)
        for c in cats:
            row[c.index] = c.score
        # r.timestamp_ms is the window start; the window is ~0.975 s long
        times.append(r.timestamp_ms / 1000.0 + 0.4875)
        rows.append(row)
    return np.array(times), np.vstack(rows), np.array(names)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio")
    p.add_argument("--out", default="features.npz")
    p.add_argument("--model", default="models/yamnet.tflite")
    p.add_argument("--no-learned", action="store_true", help="skip the neural classifier")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--hop", type=int, default=512)
    a = p.parse_args()

    t0 = time.perf_counter()
    y, sr = librosa.load(a.audio, sr=a.sr, mono=True)
    duration = len(y) / sr

    times, rms, centroid, flatness, onset = handcrafted(y, sr, a.hop)
    t_hand = time.perf_counter() - t0

    l_times, l_scores, names = np.zeros(0), np.zeros((0, 0), dtype=np.float32), np.array([], dtype="<U1")
    t_learn = 0.0
    if not a.no_learned:
        if not Path(a.model).exists():
            print(f"[warn] model file not found: {a.model}. Skipping learned features.")
        else:
            try:
                t1 = time.perf_counter()
                l_times, l_scores, names = learned(a.audio, a.model)
                t_learn = time.perf_counter() - t1
            except Exception as e:  # keep the workshop moving
                print(f"[warn] learned features failed ({type(e).__name__}: {e}). Saving handcrafted only.")

    np.savez(a.out, times=times, rms=rms, centroid=centroid, flatness=flatness, onset=onset,
             learned_times=l_times, learned_scores=l_scores, class_names=names,
             sr=sr, hop=a.hop)

    total = time.perf_counter() - t0
    print(f"Audio duration:        {duration:.1f} s")
    print(f"Handcrafted frames:    {len(times)}  ({len(times) / duration:.1f} per second)")
    print(f"Learned frames:        {len(l_times)}  ({len(l_times) / duration:.1f} per second), {len(names)} classes")
    print(f"Handcrafted time:      {t_hand:.2f} s")
    print(f"Learned time:          {t_learn:.2f} s")
    print(f"Real-time factor:      {duration / total:.1f}x  (above 1 = faster than real time)")
    print(f"Saved to {a.out}")


if __name__ == "__main__":
    main()
