import librosa
import matplotlib.pyplot as plt

y, sr = librosa.load("audio/track1.mp3", sr=22050, mono=True)
hop = 512  # frames per second = sr / hop

rms = librosa.feature.rms(y=y, hop_length=hop)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
t = librosa.times_like(rms, sr=sr, hop_length=hop)

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 7))
for a, (name, v) in zip(ax, [("RMS", rms), ("Centroid", centroid),
                             ("Flatness", flatness), ("Onset", onset)]):
    a.plot(t, v)
    a.set_ylabel(name)
ax[-1].set_xlabel("time (s)")
plt.tight_layout()
plt.show()
