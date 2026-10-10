# Workshop 2: What Does the Machine Hear?

**CLASS-2a Models, Methods and Practice in Computer Science**
Handcrafted versus learned audio features

*Duration: 3 hours | Python + TouchDesigner | Work in pairs*

## Contents

- [Introduction](#introduction-10-min)
- [Part 1: Set up the pipeline](#part-1-set-up-the-pipeline-25-min)
- [Part 2: Handcrafted features](#part-2-handcrafted-features-35-min)
- [Part 3: Learned features](#part-3-learned-features-35-min)
- [Part 4: Head to head](#part-4-head-to-head-30-min)
- [Part 5: Mapping and your audio source](#part-5-mapping-and-your-audio-source-20-min)
- [Show and tell, and the bridge to your Python project](#show-and-tell-and-the-bridge-to-your-python-project-15-min)
- [After the workshop](#after-the-workshop)
- [Glossary](#glossary)
- [Quick fixes](#quick-fixes)
- [Mapping sheet](#mapping-sheet)

---

## Introduction (10 min)

In Workshop 1 you turned sound into numbers with **handcrafted** features: loudness and band energies that a person designed. A feature like RMS tells you *how much* signal there is, but not *what* it is. A drum, a voice and a vacuum cleaner can all be equally loud.

Today you compare those handcrafted features with **learned** features: outputs of a neural network trained on a very large collection of labelled sounds. Instead of "how loud", a learned model can say "this sounds like music, with a drum kit, and some singing". That sounds powerful, and sometimes it is. It can also be wrong, slow, hard to interpret, or tuned to a world of sounds that is not yours.

**The question for today is not which is better. It is: what does each kind of feature tell you, what does it miss, and which one should drive each part of your piece?**

### By the end of today you will have

- run an analysis pipeline that produces handcrafted *and* learned features from one audio file;
- plotted both and checked them against your own ears;
- built and compared two detectors for the same musical event, one handcrafted and one learned, with simple evidence;
- sent both kinds of feature live into TouchDesigner to drive visuals;
- written a first case for your chosen audio source and a mapping sheet that records how each feature can fail.

### How the manual is numbered

The manual is split into **Parts 1 to 5**. Each Part contains numbered **Tasks** whose first digit is the Part number. Part 3 contains Tasks 3.1 to 3.5, and so on.

### What you need

- the workshop folder from your tutor (scripts, three audio tracks, and pre-computed fallback files);
- a Python environment set up with `uv sync --extra learned`;
- TouchDesigner (as in Workshop 1) and headphones;
- a text editor or notebook for your decision log.

### Timetable at a glance

| Time | Part |
|------|------|
| 0:00–0:10 | Introduction and listening demo |
| 0:10–0:35 | Part 1: Set up the pipeline |
| 0:35–1:10 | Part 2: Handcrafted features |
| 1:10–1:20 | Break |
| 1:20–1:55 | Part 3: Learned features |
| 1:55–2:25 | Part 4: Head to head |
| 2:25–2:45 | Part 5: Mapping and your audio source |
| 2:45–3:00 | Show and tell, and the bridge to your Python project |

> **Two working habits for today**
>
> **Predict before you look.** Writing down what you expect, then checking, is how you learn what a feature really does.
> **Keep a decision log.** After every Part, write one or two lines: what you tried, what you found, what you would do next. This is raw material for your supporting documentation.

---

## Part 1: Set up the pipeline (25 min)

### The architecture

Your final piece will analyse audio in Python and render in Python. Today the pipeline has one extra stage so that you can explore mappings quickly in TouchDesigner.

```text
Audio file → Extract (librosa + model) → features file → Stream (OSC) → TouchDesigner visuals
```

> **Why this matters**
>
> The slow, heavy work (running the neural network) happens **once, offline**, and the results are stored. The live part only plays the stored numbers back in time with the audio. This "precompute, then play back" pattern is exactly what you can use in your final project when your audio source is a file. It keeps the render loop fast and the piece reliable.

### What the two scripts do

You will run two Python scripts today. You do not need to write them, but you should know what each one does, because your final project will contain the same stages.

**`extract_features.py`**

1. Loads the audio file as a single mono channel.
2. Splits it into short overlapping frames (about 43 per second) and measures four handcrafted features on each: RMS, spectral centroid, spectral flatness and onset strength, using librosa.
3. Runs a pre-trained classifier over the audio. For each window of about one second (about two per second) it produces a score for every sound class it knows.
4. Prints the audio length, how long the analysis took and the real-time factor.
5. Saves everything to one file, `features.npz`.

**`stream_features.py`**

1. Loads `features.npz` and scales the handcrafted features to the range 0 to 1 (using the percentile method from Task 2.3).
2. Picks the classes you asked for with `--classes`.
3. Starts playing the audio and, about 60 times a second, looks up the feature values for the current playback time. For the slow learned scores it uses interpolation, as in Task 3.4.
4. Sends each value over the network as an OSC message, one address per feature. TouchDesigner's **OSC In CHOP** turns each address into a channel.

> **The idea to take away**
>
> `extract_features.py` is the **slow, heavy** stage. It runs once. `stream_features.py` is the **fast, light** stage: it only looks up stored numbers in time with the sound. Your final project will follow the same split, with the lookup happening inside your own render loop instead of over OSC.

### Task 1.1: Run the extraction

> **Do this**
>
> 1. Open a terminal in the workshop folder and run `uv sync --extra learned`. (If this fails, run `uv sync` alone and tell your tutor: you can still do everything with the fallback files.)
> 2. Run the extraction on the first track:
>
>    ```bash
>    uv run python extract_features.py audio/track1.wav
>    ```
>
>    It writes a file called `features.npz`.
> 3. Note how long the extraction took, and how long the track is. Divide track length by processing time: this is the **real-time factor**. Above 1 means faster than real time.

> **If you are stuck**
>
> **Install fails or the model will not load:** do not spend more than five minutes on it. Copy the pre-computed `features.npz` for your track from the `fallback/` folder and carry on.
> **File not found:** check you are in the workshop folder and the audio path is right.

### Task 1.2: Inspect the data

> **Do this**
>
> Open a Python session or notebook and run:
>
> ```python
> import numpy as np
> d = np.load("features.npz", allow_pickle=True)
> for k in d.files:
>     print(k, d[k].shape)
> ```
>
> Find these entries and answer the questions below.
>
> | Entry | Meaning |
> |-------|---------|
> | `times` | Timestamps (seconds) of each handcrafted frame |
> | `rms`, `centroid`, `flatness`, `onset` | Handcrafted features, one value per frame |
> | `learned_times` | Timestamps of each learned frame |
> | `learned_scores` | Classifier output: one row per learned frame, one column per sound class |
> | `class_names` | The name of each class (column) |
>
> 1. How many handcrafted frames per second are there? How many learned frames per second?
> 2. How many classes can the model recognise? Print ten of the class names.
> 3. Which kind of feature updates faster? What does that mean for animation?

> **You should see**
>
> Handcrafted features update about 40 times per second. Learned features update about twice a second. You can state in one sentence why that mismatch will matter when you drive a 60 frames-per-second visual.

### Task 1.3: Connect TouchDesigner

> **Do this**
>
> 1. In TouchDesigner, create an **OSC In CHOP**. Set `Network Port` to `7000`.
> 2. In your terminal, run:
>
>    ```bash
>    uv run python stream_features.py audio/track1.wav features.npz
>    ```
>
>    The script plays the audio **from Python** and sends the stored features in time with it. (You do not need an Audio File In CHOP today.)
> 3. Open the **OSC In CHOP** viewer. You should see channels named after the features, moving with the music.
> 4. Add a **Null CHOP** named **osc_out** after it, and a **Trail CHOP** so you can see the signals.

> **You should see**
>
> Several channels (for example `rms`, `centroid`, `onset`) move in time with the music you hear. If you restart the Python script, they restart too.

> **If you are stuck**
>
> **No channels appear:** check the port is `7000` in both places, that the script is running, and that no firewall is blocking it.
> **Sound but no channels:** start the TouchDesigner node first, then the script.
> **Channels but no sound:** check your output device in the script options with your tutor.

> **Evidence to capture for 02_Technical_Workshop_Tasks**
>
> - [ ] Screenshot of the terminal output from Task 1.1 and your real-time factor.
> - [ ] Your answers to the three questions in Task 1.2.
> - [ ] Screenshot of the OSC In CHOP with channels moving.
> - [ ] One decision-log entry for Part 1.

---

## Part 2: Handcrafted features (35 min)

**Goal:** understand what four standard features measure, and use them to drive a visual.

| Feature | What it measures |
|---------|------------------|
| **RMS** | Overall energy of the signal; loudness |
| **Spectral centroid** | The "centre of mass" of the spectrum, in Hz; high means bright, low means dark |
| **Spectral flatness** | How noise-like (close to 1) or tone-like (close to 0) the sound is |
| **Onset strength** | How strongly new sound events begin at each moment |

### Task 2.1: Compute and plot

> **Do this**
>
> 1. Compute the features yourself from the audio, so you see how they are made:
>
>    ```python
>    import librosa, numpy as np
>    import matplotlib.pyplot as plt
>
>    y, sr = librosa.load("audio/track1.wav", sr=22050, mono=True)
>    hop = 512                                   # frames per second = sr / hop
>
>    rms      = librosa.feature.rms(y=y, hop_length=hop)[0]
>    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
>    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
>    onset    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
>    t        = librosa.times_like(rms, sr=sr, hop_length=hop)
>
>    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 7))
>    for a, (name, v) in zip(ax, [("RMS", rms), ("Centroid", centroid),
>                                  ("Flatness", flatness), ("Onset", onset)]):
>        a.plot(t, v); a.set_ylabel(name)
>    ax[-1].set_xlabel("time (s)")
>    plt.tight_layout(); plt.show()
>    ```
>
> 2. Check that your values look similar to the ones in `features.npz`. Small differences are fine.

### Task 2.2: Predict, then check

> **Do this**
>
> 1. **Before** looking closely at the plots, write down your predictions:
>    - Which feature jumps at a drum hit?
>    - Which feature changes most when a bright instrument or voice enters?
>    - Which feature separates a pitched sound from a noisy one?
> 2. Play the track and note **three timestamps** with a clear event (for example a drum entry, a quiet section, a vocal).
> 3. Look at the four plots at those times. Were your predictions right? Write one sentence on the biggest surprise.

### Task 2.3: Make features usable

Raw features have awkward ranges: RMS is tiny, and the centroid is measured in thousands of Hz. A visual needs values between 0 and 1.

> **Do this**
>
> 1. Write a normalise function that uses *percentiles*, not the minimum and maximum, so that a single spike does not squash everything else:
>
>    ```python
>    def normalise(x, lo=5, hi=95):
>        a, b = np.percentile(x, [lo, hi])
>        return np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
>    ```
>
> 2. Apply it to each feature and plot the before and after for the centroid.
> 3. Explain in one sentence why percentiles are safer than min–max for audio.

### Task 2.4: Drive a visual in TouchDesigner

> **Do this**
>
> 1. With the stream script running, use **Select CHOP**, **Lag CHOP** and **Math CHOP** nodes (as in Workshop 1) to turn `rms` and `centroid` into clean 0 to 1 values, ending in a **Null CHOP** for each.
> 2. Drive a **Circle TOP** `Radius` from `rms`, and the `Brightness 1` of a **Level TOP** from `centroid`. Remember: no wire from CHOP to TOP, use an expression such as `op('rms_out')[0]`.
> 3. Trigger a short flash from `onset`: use a **Logic CHOP** or **Trigger CHOP**, then reference it in a **Level TOP**.

> **You should see**
>
> A circle that pulses with loudness, gets brighter on bright sounds, and flashes on new events. You can say which feature does which job.

> **If you are stuck**
>
> **A value never leaves zero:** the feature may be tiny. Check the channel in the viewer and rescale with the **Math CHOP** range.
> **Flicker:** increase the Lag.
> **Flash never fires:** lower the threshold of the Logic or Trigger CHOP.

> **Evidence to capture for 02_Technical_Workshop_Tasks**
>
> - [ ] Your four-panel plot with the three event timestamps annotated.
> - [ ] Your predictions, and a sentence on the biggest surprise.
> - [ ] The before and after normalisation plot.
> - [ ] Screenshot of the TouchDesigner network and output.
> - [ ] One decision-log entry for Part 2.

**Break (10 min).** Save your work. Walk away from the screen.

---

## Part 3: Learned features (35 min)

**Goal:** look at what a pre-trained classifier "hears", test it against your own ears, and choose which of its outputs are worth using.

> **What is a learned feature?**
>
> The model was trained on a very large collection of short, labelled audio clips. Given a new sound, it outputs a **score for every class it knows** (for example *Music*, *Speech*, *Drum*, *Singing*). These scores behave like probabilities that rise and fall as the sound changes. The model is a **pre-trained** one: you did not train it, and it only knows the categories it was given. The list of categories comes from the people who built the training data, and may not match how you think about your music.

### Task 3.1: Meet the classifier output

> **Do this**
>
> 1. Plot the ten classes with the highest average score as a heatmap:
>
>    ```python
>    d = np.load("features.npz", allow_pickle=True)
>    scores, names = d["learned_scores"], list(d["class_names"])
>    lt = d["learned_times"]
>
>    top = np.argsort(scores.mean(axis=0))[::-1][:10]
>    plt.figure(figsize=(10, 4))
>    plt.imshow(scores[:, top].T, aspect="auto", origin="lower",
>               extent=[lt[0], lt[-1], 0, len(top)], cmap="magma")
>    plt.yticks(np.arange(len(top)) + 0.5, [names[i] for i in top])
>    plt.xlabel("time (s)"); plt.colorbar(label="score"); plt.show()
>    ```
>
> 2. Describe in two sentences what the heatmap tells you about your track.

### Task 3.2: The listening check

> **Do this**
>
> 1. Choose **five timestamps** in your track. At each one, listen, then print the top three classes from the model.
> 2. Fill in a table with one row per timestamp: **what I hear / what the model says / agree, partly, or no**.
> 3. Find at least one clear **failure**: a place where the model is wrong, uncertain or misses something obvious. Write a sentence on why you think it happened.

> **Be critical**
>
> A model can be *confidently wrong*. It was trained on certain kinds of audio, so it may be less reliable for music styles, instruments or recording conditions that were rare in its training data. Your documentation should record where your chosen feature works and where it does not. Admitting a limitation is good analysis, not a weakness in your project.

### Task 3.3: Choose interpretable features

There are hundreds of classes. You do not need them all, and using them all would make your mapping arbitrary.

> **Do this**
>
> 1. Pick **three to five classes** that are meaningful for *your* track (for example *Music*, *Drum*, *Singing*, *Speech*, *Silence*).
> 2. Plot them over time. For each one write: why it is useful, and how noisy it is.
> 3. Reject at least one class you tried, and say why.

### Task 3.4: Align and smooth

The learned features update only about twice a second, but your visual runs at 60 frames per second. You must **resample** the slow signal onto the fast timeline, then smooth it.

> **Do this**
>
> 1. Resample one chosen class to the handcrafted timeline:
>
>    ```python
>    idx = names.index("Music")                  # use one of your chosen classes
>    music = np.interp(t, lt, scores[:, idx])    # t = handcrafted timestamps
>
>    def smooth(x, alpha=0.1):                   # exponential smoothing (a Lag)
>        out = np.zeros_like(x); v = 0.0
>        for i, target in enumerate(x):
>            v += (target - v) * alpha
>            out[i] = v
>        return out
>    music_s = smooth(music)
>    ```
>
> 2. Plot the raw learned signal, the interpolated one and the smoothed one on one figure.
> 3. Answer: how does the learned signal behave at the moment a new sound starts? Is it early, on time, or late compared with the onset strength? What does that mean for a visual that must react "instantly"?

### Task 3.5: Bring it into TouchDesigner

> **Do this**
>
> 1. Restart the stream script with your chosen classes, for example:
>
>    ```bash
>    uv run python stream_features.py audio/track1.wav features.npz --classes Music Drum Singing
>    ```
>
>    (Use the exact class names from your list.)
> 2. In TouchDesigner, make a **second visual layer** driven by a learned class, for example a noise texture that fades in when *Singing* is high, or a colour shift when *Drum* is high.
> 3. Composite it over your Part 2 circle. Watch for a moment where learned and handcrafted features tell different stories.

> **If you are stuck**
>
> **Class name not found:** class names are case-sensitive and must match exactly; print `names` to check.
> **Learned channels look stepped or jumpy:** that is the low update rate; increase the Lag or smooth in Python.
> **Everything is near zero:** the class may not be present in your track; choose another.

> **Evidence to capture for 02_Technical_Workshop_Tasks**
>
> - [ ] The heatmap, with two sentences of interpretation.
> - [ ] Your five-row listening-check table, including one clear failure.
> - [ ] Plot of your chosen classes, with a note on which you rejected and why.
> - [ ] The raw, interpolated and smoothed comparison plot, and your answer about timing.
> - [ ] Screenshot of the two-layer visual in TouchDesigner.
> - [ ] One decision-log entry for Part 3.

---

## Part 4: Head to head (30 min)

**Goal:** give both kinds of feature the same job, and use evidence to decide which does it better.

### Task 4.1: Choose a question and mark the ground truth

> **Do this**
>
> 1. Choose **one** question that matters for your piece:
>    - When does a drum hit or beat start?
>    - When do vocals enter or leave?
>    - When does the music change section?
>    - When is the sound speech and not music?
> 2. Listen through the track and write down the **ground truth**: the timestamps (to the nearest second) where your event really occurs. Aim for at least eight events.

### Task 4.2: Build two detectors

> **Do this**
>
> 1. **Handcrafted detector.** Pick a feature (such as onset strength, RMS or centroid) and use a threshold or peak detection to find events:
>
>    ```python
>    from scipy.signal import find_peaks
>    peaks, _ = find_peaks(normalise(onset), height=0.5, distance=int(0.2 * sr / hop))
>    events_hand = t[peaks]
>    ```
>
> 2. **Learned detector.** Use a class score and detect when it crosses a threshold:
>
>    ```python
>    x = music_s                                  # or Singing, Drum, ...
>    crossings = np.where((x[1:] > 0.5) & (x[:-1] <= 0.5))[0] + 1
>    events_learned = t[crossings]
>    ```
>
> 3. Compare each detector with your ground truth using a **tolerance** of 0.5 seconds. Count **hits** (real events found), **misses** (real events not found) and **false alarms** (detections with no real event).
> 4. Record the results in a small table, one row per detector.

> **Why a table of counts?**
>
> It is easy to say "it works well". A table of hits, misses and false alarms is *evidence*. It also shows trade-offs: lowering a threshold finds more events but adds false alarms. You will use this style of reasoning in your supporting documentation.

### Task 4.3: Test on a second track

> **Do this**
>
> 1. Run extraction on a contrasting track (for example a sparse ambient piece against a beat-driven one).
> 2. Re-run both detectors **without changing the thresholds**.
> 3. Which detector held up better? Which one needed retuning?

### Task 4.4: Decide

> **Do this**
>
> In three to five sentences, answer: **for this job, which feature would you use in your piece, or would you combine them, and why?** Mention accuracy, speed, simplicity and how well you can explain what the feature means.

> **You should see**
>
> You have a results table for two detectors on two tracks and a written decision that refers to numbers from the table.

> **Evidence to capture for 02_Technical_Workshop_Tasks**
>
> - [ ] Your question and ground-truth timestamps.
> - [ ] Hits, misses and false alarms for both detectors on both tracks.
> - [ ] Your decision paragraph.
> - [ ] One decision-log entry for Part 4.

---

## Part 5: Mapping and your audio source (20 min)

### Task 5.1: Update your mapping sheet

> **Do this**
>
> Use the [mapping sheet](#mapping-sheet) at the end of this manual. Complete **at least four rows**, with **at least two handcrafted and two learned features**. For each row record:
>
> 1. the feature and its type (H for handcrafted, L for learned);
> 2. the visual parameter it drives;
> 3. the reason, using the idea of a perceptual match (for example brightness of sound to brightness of image);
> 4. the **failure mode**: when does this feature mislead, and what does the piece do then?

### Task 5.2: Write the case for your audio source

Your supporting documentation needs a section on your audio source.

> **Do this**
>
> Write **five sentences** covering:
>
> 1. what the audio source is and why you chose it;
> 2. what its features look like (refer to a specific plot from today);
> 3. what makes it an interesting data source (for example variation, structure, contrast);
> 4. one feature that works well on it, and one that does not.

> **Evidence to capture for 02_Technical_Workshop_Tasks**
>
> - [ ] The completed mapping sheet.
> - [ ] Your five-sentence audio source case.
> - [ ] One decision-log entry for Part 5.

---

## Show and tell, and the bridge to your Python project (15 min)

### Show and tell (10 min)

Some pairs show their two-layer visual for one minute. The audience guesses which layer uses a handcrafted feature and which uses a learned one, and then the pair explains. Listen for the *failure modes* people found.

### What today means for your final piece (5 min)

| Today | In your Python project |
|-------|------------------------|
| `extract_features.py` runs offline | Precompute features for a file and save them in the repository |
| Interpolate slow features to a fast timeline | `np.interp` onto the render-loop frame times |
| Normalise with percentiles | A `normalise()` helper applied before mapping |
| Exponential smoothing | A `smooth()` step in the render loop |
| Thresholds for events | Event detection with hit/miss evidence from testing |
| OSC to TouchDesigner | Direct function calls in your own render loop |
| Real-time factor of the model | Decide offline (file) or live (microphone); live needs a latency budget |

**Important for your project:** the brief requires an audio analysis pipeline built with Python, NumPy and librosa. Treat learned features as an *extension* of that foundation, and make sure your repository still runs clean from `uv sync` and `uv run python main.py`. A model that needs a huge download or a special machine is a risk for your submission.

---

## After the workshop

### Hand-in checklist for 02_Technical_Workshop_Tasks

- [ ] All plots and tables from the evidence boxes, clearly labelled by Part and Task
- [ ] Your TouchDesigner project, saved as `STUDENTID_w2.toe`
- [ ] Your handcrafted-feature code and comparison code
- [ ] Completed mapping sheet and audio source case
- [ ] Your decision log, one entry per Part

### Before the next workshop

Next time we look at **decomposing sound**: separating a track into drums, bass, vocals and other parts, and giving each its own visual role.

1. Choose a track with **clearly different layers** (for example drums plus bass plus voice).
2. Decide which **two layers** you would most like to treat separately in a visual, and write down why.
3. Run today's extraction on your chosen track and check that the output looks sensible.

### Short reflection (5 minutes, for your Reflective Learning Summary)

- What did the learned features give you that handcrafted ones could not? What did they cost?
- Where did you trust the model, and where did you stop trusting it? What changed your mind?
- How will this change the way you describe your audio source in the documentation?

---

## Glossary

| Term | Meaning |
|------|---------|
| **Handcrafted feature** | A measurement designed by a person, such as RMS or spectral centroid |
| **Learned feature** | A value produced by a model trained on data, such as a class score |
| **Pre-trained model** | A model already trained by someone else, used here without further training |
| **Classifier** | A model that assigns scores to a fixed list of categories |
| **Class score** | How strongly the model believes a category is present, between 0 and 1 |
| **Inference** | Running a trained model on new data |
| **Real-time factor** | Audio length divided by processing time; above 1 is faster than real time |
| **Hop length** | Number of audio samples between successive analysis frames |
| **Spectral centroid** | The "centre of mass" of the spectrum; a measure of brightness |
| **Spectral flatness** | How noise-like or tone-like a sound is |
| **Onset strength** | A curve that peaks when new sound events begin |
| **Interpolation** | Estimating values at new time points between known ones |
| **Ground truth** | The correct answer, here marked by listening |
| **False alarm** | A detection where no real event happened |
| **OSC** | Open Sound Control, a simple network protocol for sending live values |

---

## Quick fixes

| Problem | Try this |
|---------|----------|
| Model will not install or load | Use the pre-computed file from `fallback/`; tell your tutor |
| No OSC channels in TouchDesigner | Same port both sides; start TouchDesigner first; check firewall |
| No sound from the stream script | Check output device; close other audio apps |
| Class name not found | Names are case-sensitive; print the list |
| Learned signal looks stepped | Expected: low update rate. Interpolate and smooth |
| Detector finds too many events | Raise the threshold or minimum spacing |
| Detector misses events | Lower the threshold; check the feature is normalised |
| Behind the group | Use the fallback files and carry on with [Part 4](#part-4-head-to-head-30-min) |

---

## Mapping sheet

**Names:** ______________________  **Audio source:** ______________________

| H or L | Feature | Visual parameter | Reason for this mapping | Failure mode |
|--------|---------|------------------|-------------------------|--------------|
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

**Most trustworthy feature, and why:**

______________________________________________________________________

______________________________________________________________________

**Least trustworthy feature, and what the piece should do when it fails:**

______________________________________________________________________

______________________________________________________________________
