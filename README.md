# The Math Behind the Mix
### First steps towards an auto-DJ

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

This project asks: **can the mathematics behind each step be derived and
implemented from scratch, and how do the results compare with existing
libraries and tools?** Each method is explained mathematically, verified
with unit tests on synthetic signals with known answers, and evaluated on
annotated real music against simple baselines and published results.

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
| 7 | *(Future work)* Blend two tracks smoothly | Equal-power crossfades, time-stretching |
| 8 | *(Future work)* Order tracks into a set | Shortest Hamiltonian path: greedy vs. Held–Karp DP |

## 3. Scope

**In scope**
- Electronic / dance music with a steady tempo (the main DJ use case).
- Global (single) tempo and key per track.

**Out of scope**
- Tempo changes within a track, key modulations, live performance effects.
- Real-time processing.
- Full beat-grid tracking across an entire track (only local alignment for a
  transition).
- Mixing tracks and ordering a set (planned as future work, see the roadmap).

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
│   ├── datasets.py        # reading the downloaded datasets
│   ├── mixing.py          # planned: crossfades, time-stretching
│   └── setlist.py         # planned: track graph and ordering
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

**Done**
- [x] Project scaffold and problem formulation
- [x] Sound as math: synthesis, DFT vs. FFT
- [x] Chords and tuning: ratios, equal temperament, dissonance curves
- [x] Chromagram
- [x] Chord recognition
- [x] Key detection (Krumhansl–Schmuckler)
- [x] Harmonic distance (Camelot wheel)
- [x] Key detection on real recordings
- [x] Onset detection and tempo estimation
- [x] Conclusion and future work

**Future work**
- [ ] Tempo estimation on real recordings (GiantSteps Tempo)
- [ ] Key profiles for electronic dance music
- [ ] Transitions: beat alignment and equal-power crossfades
- [ ] Set ordering: greedy vs. Held–Karp

## 10. References

1. M. Müller, *Fundamentals of Music Processing*, 2nd ed., Springer, 2021.
   Companion notebooks: FMP Notebooks, International Audio Laboratories Erlangen.
2. D. Benson, *Music: A Mathematical Offering*, Cambridge University Press, 2006.
3. W. A. Sethares, *Tuning, Timbre, Spectrum, Scale*, 2nd ed., Springer, 2005.
4. J. O. Smith III, *Mathematics of the Discrete Fourier Transform (DFT)*, CCRMA, Stanford.
5. C. L. Krumhansl, *Cognitive Foundations of Musical Pitch*, Oxford University Press, 1990.
6. E. D. Scheirer, "Tempo and beat analysis of acoustic musical signals,"
   *Journal of the Acoustical Society of America*, 103(1), 1998.
7. D. P. W. Ellis, "Beat tracking by dynamic programming,"
   *Journal of New Music Research*, 36(1), 2007.
8. J. Laroche and M. Dolson, "Improved phase vocoder time-scale modification of audio,"
   *IEEE Transactions on Speech and Audio Processing*, 7(3), 1999.
9. R. Plomp and W. J. M. Levelt, "Tonal consonance and critical bandwidth,"
   *Journal of the Acoustical Society of America*, 38(4), 1965.
10. J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of
    complex Fourier series," *Mathematics of Computation*, 19(90), 1965.
11. W. A. Sethares, "Local consonance and the relationship between timbre and scale,"
    *Journal of the Acoustical Society of America*, 94(3), 1993.
12. C. L. Krumhansl and E. J. Kessler, "Tracing the dynamic changes in perceived tonal
    organization in a spatial representation of musical keys,"
    *Psychological Review*, 89(4), 1982.
13. P. Knees, Á. Faraldo, P. Herrera, R. Vogl, S. Böck, F. Hörschläger and M. Le Goff,
    "Two data sets for tempo estimation and key detection in electronic dance music
    annotated from user corrections," *Proceedings of the 16th International Society
    for Music Information Retrieval Conference (ISMIR)*, 2015.
14. Á. Faraldo, E. Gómez, S. Jordà and P. Herrera, "Key estimation in electronic dance
    music," *Proceedings of the European Conference on Information Retrieval (ECIR)*, 2016.

### Software

15. C. R. Harris et al., "Array programming with NumPy," *Nature*, 585, 2020.
16. P. Virtanen et al., "SciPy 1.0: Fundamental algorithms for scientific computing
    in Python," *Nature Methods*, 17, 2020.
17. J. D. Hunter, "Matplotlib: A 2D graphics environment,"
    *Computing in Science & Engineering*, 9(3), 2007.
18. B. McFee et al., "librosa: Audio and music signal analysis in Python,"

## Self-assessment

**What went well:** I am proud to have found out that DJing, which I had always seen as a purely artistic skill, can be explained through mathematics. It made the project very interesting to work on.

**What was difficult:** The project turned out to be more difficult than I expected. Some of the mathematics took time to understand, and the technical setup was a challenge at times, for example fixing a certificate error that blocked the dataset download.

**What I learned:** How mathematics I knew from school, like logarithms and modular arithmetic, solves real problems in music, and how to build a project step by step with tests, real data and regular commits.

**What I would do differently:** Because the project grew larger than planned, I had to cut its scope and leave mixing and set ordering for future work. Next time I would plan a smaller scope from the start and spend more time on each part.

## License

Code is released under the MIT License (see `LICENSE`). Audio datasets keep
their original licences and are not redistributed here.
