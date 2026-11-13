# The Math Behind the Mix
### Building a mini auto-DJ from first principles

> Final exam project — Math Concepts for Developers (SoftUni)
> Category: *Math applications in music* + *Algorithm implementation and analysis*

---

## 1. Problem statement

A DJ set looks like an artistic act, but underneath it is a chain of
mathematical problems. Before two tracks can be mixed, a DJ (or DJ software)
must answer:

1. **What is the tempo of each track?** (beats per minute, BPM)
2. **What is the musical key of each track?**
3. **Which tracks are compatible?** Tracks that are close in tempo and in
   related keys can be blended without sounding off.
4. **In what order should the tracks be played?** A good set minimises
   abrupt jumps in tempo and harmony.
5. **How do we actually blend two tracks?** Their beats must be aligned and
   the volume handed over without a dip or spike in loudness.

This project asks: **can these questions be answered from first principles,
using only signal processing, linear algebra, group theory and graph
algorithms?** Each step is implemented from scratch, explained mathematically,
and evaluated against annotated data and against an established library
(`librosa`) as a baseline.

Before tackling DJing, the project answers a more basic question that
everything else depends on: **why do some combinations of frequencies (chords)
sound consonant, and how does Western tuning turn this into 12 notes?**

## 2. Objectives

| # | Objective | Main mathematics |
|---|-----------|------------------|
| 1 | Represent sound and its frequency content | Fourier series, DFT/FFT, STFT |
| 2 | Explain chord formation and consonance | Frequency ratios, equal temperament (2^(1/12)), dissonance curves |
| 3 | Extract pitch-class content from audio | Chromagrams, projection onto pitch classes |
| 4 | Detect the key of a track | Correlation with key profiles (Krumhansl–Schmuckler) |
| 5 | Detect the tempo of a track | Onset detection (spectral flux), autocorrelation |
| 6 | Measure harmonic compatibility | Circle of fifths / Camelot wheel as a cyclic group |
| 7 | Blend two tracks smoothly | Equal-power crossfades, time-stretching |
| 8 | Order tracks into a set | Shortest Hamiltonian path: greedy vs. Held–Karp DP vs. heuristics |

## 3. Scope

**In scope**
- Electronic / dance music with a steady tempo (the main DJ use case).
- Global (single) tempo and key per track.
- Small sets (roughly 5–15 tracks) for exact ordering algorithms.

**Out of scope**
- Tempo changes within a track, key modulations, live performance effects.
- Real-time processing.
- Full beat-grid tracking across an entire track (only local alignment for a
  transition).

## 4. Assumptions

- Audio is mono or downmixed to mono, resampled to a common sample rate.
- Tracks use 12-tone equal temperament tuned to A4 = 440 Hz (approximately).
- Tempo lies in a typical dance range (about 70–180 BPM); estimates outside
  it are folded by factors of 2.
- Key is one of 24 major/minor keys.

## 5. Existing solutions

Commercial and open tools already solve parts of this problem:

- **Mixed In Key** — commercial key and energy detection, popularised the
  Camelot notation.
- **rekordbox / Serato / Traktor** — DJ software with BPM, key and beat-grid
  analysis.
- **librosa / Essentia** — open-source music analysis libraries.

These tools are accurate but act as black boxes for the user. This project
does not aim to beat them; it aims to **open the box**: derive each method,
implement it, and measure where and why it fails.

## 6. Repository structure

```
math-behind-the-mix/
├── README.md              # this file
├── main.ipynb             # the full write-up: text, math, code, results
├── src/djmath/            # reusable implementation
│   ├── signals.py         # synthesis, DFT/FFT, STFT
│   ├── harmony.py         # tuning, consonance, chroma
│   ├── key.py             # key detection, Camelot distance
│   ├── tempo.py           # onset detection, tempo estimation
│   ├── mixing.py          # crossfades, time-stretching
│   └── setlist.py         # track graph and ordering algorithms
├── tests/                 # unit tests (pytest)
├── scripts/
│   └── download_data.py   # fetches datasets (audio is never committed)
├── data/                  # local data, git-ignored
└── requirements.txt
```

## 7. Setup

```bash
git clone <this-repo-url>
cd math-behind-the-mix
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # makes `djmath` importable
pytest                           # run the tests
jupyter notebook main.ipynb
```

## 8. Data

No audio is stored in this repository. Audio files are downloaded locally into
`data/` (which is git-ignored) by `scripts/download_data.py`.

Planned sources:
- **Synthetic signals** generated in code, with exactly known tempo and key —
  used as ground truth for validation.
- **GiantSteps Key** and **GiantSteps Tempo** datasets — annotated electronic
  music, used for evaluation.
- **Free Music Archive (FMA)** — Creative Commons tracks, used for rendering
  example transitions.

All datasets are credited in the References section, and their licences are
respected.

## 9. Roadmap

- [x] Project scaffold and problem formulation
- [x] Sound as math: synthesis, DFT vs. FFT
- [x] Chords and tuning: ratios, equal temperament, dissonance curves
- [x] Chromagram and chord recognition
- [x] Key detection and Camelot distance
- [ ] Key detection on real recordings
- [ ] Onset detection and tempo estimation
- [ ] Evaluation against annotated datasets and librosa
- [ ] Transitions: crossfades and time-stretching
- [ ] Set ordering: greedy vs. Held–Karp vs. heuristic
- [ ] Results, conclusion, limitations, self-assessment

## 10. References

*(Expanded as the project progresses.)*

1. M. Müller, *Fundamentals of Music Processing*, 2nd ed., Springer, 2021.
   Companion notebooks: FMP Notebooks, International Audio Laboratories Erlangen.
2. D. Benson, *Music: A Mathematical Offering*, Cambridge University Press, 2006.
3. W. Sethares, *Tuning, Timbre, Spectrum, Scale*, 2nd ed., Springer, 2005.
4. J. O. Smith III, *Mathematics of the Discrete Fourier Transform (DFT)*, CCRMA, Stanford.
5. C. L. Krumhansl, *Cognitive Foundations of Musical Pitch*, Oxford University Press, 1990.
6. E. D. Scheirer, "Tempo and beat analysis of acoustic musical signals,"
   *Journal of the Acoustical Society of America*, 103(1), 1998.
7. D. P. W. Ellis, "Beat tracking by dynamic programming,"
   *Journal of New Music Research*, 36(1), 2007.
8. J. Laroche and M. Dolson, "Improved phase vocoder time-scale modification
   of audio," *IEEE Transactions on Speech and Audio Processing*, 7(3), 1999.
9. R. Plomp and W. J. M. Levelt, "Tonal consonance and critical bandwidth,"
   *Journal of the Acoustical Society of America*, 38(4), 1965.
10. J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of
    complex Fourier series," *Mathematics of Computation*, 19(90), 1965.
11. W. A. Sethares, "Local consonance and the relationship between timbre and
    scale," *Journal of the Acoustical Society of America*, 94(3), 1993.
12. C. R. Harris et al., "Array programming with NumPy," *Nature*, 585, 2020.
13. J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in
    Science & Engineering*, 9(3), 2007.
14. B. McFee et al., "librosa: Audio and music signal analysis in Python,"
    *Proceedings of the 14th Python in Science Conference*, 2015.

## License

Code is released under the MIT License (see `LICENSE`). Audio datasets keep
their original licences and are not redistributed here.
