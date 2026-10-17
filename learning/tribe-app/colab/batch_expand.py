"""batch_expand.py: expand the tribe v2 stimulus library from 9 to 20 stimuli.

colab only, won't run locally: it needs the tribe v2 model loaded on a
gpu runtime. run it after: setup cell → restart → version check → model load.

what it does
------------
for each entry in NEW_STIMULI (5 video, 4 audio, 2 text):
  1. downloads the source file from a stable archive.org / blender.org /
     wikimedia URL (text stimuli are read from drive instead).
  2. preprocesses it where needed: unzips caminandes 2 and strips the audio
     track from a trip to the moon, so that clip is video-only.
  3. clips 60 s starting at each stimulus's `start` second (chosen to skip
     logos, librivox disclaimers and unsuitable scenes), re-encoding video to
     H.264 and all audio to mp3.
  4. runs tribe inference and saves the prediction as {id}_preds.npy,
     shape (n_timesteps, 20484) on the fsaverage5 left/right surface.
  5. writes metadata rows (id, title, notes, npy_file, n_timesteps,
     n_vertices, credit) to new_rows.csv, to be merged locally into
     library/library_metadata.csv.

cells
-----
  1  mount drive; define LIB, CLIPS, RAW, CLIP_SECONDS
  2  NEW_STIMULI: id, title, modality, url/file, start, prep flags,
     notes, credit
  3  link check (wget --spider on every URL)
  4  prepare(): download → preprocess → clip
  5  run(): prepare → TRIBE predict → save .npy (skips finished stimuli)
  6  wordless gate (caminandes_2, trip_to_the_moon, pastoral_audio), then
     the full batch
  7  build new_rows.csv

outputs (all on drive under LIB, so they survive runtime disconnects (cries in corner from war flashbacks))
---------------------------------------------------------------------
  {id}_preds.npy        one per stimulus
  clips/{id}_clip.*     the exact 60 s clip each prediction was made from
  new_rows.csv          metadata for the 11 new stimuli
text excerpts are read from LIB/texts/ (raven_text.txt, origin_text.txt).

resuming
--------
run() skips any stimulus whose .npy already exists, so after a crash or
disconnect, rerun the setup → model load → cells 1, 2, 4, 5, then cell 6,
and only unfinished stimuli will run.

known behaviour and caveats
---------------------------
- run the wordless gate first. stimuli without speech risk failing
  whisperx alignment. all three passed (Oct 2026).
- whisperx hallucinates words in wordless audio (1 transcript chunk in
  caminandes 2, 3 words in the Pastoral symphony), so their text modality
  is not empty.
- text stimuli are synthesized to speech by tribe, so their timestep count
  depends on that speech (raven_text 56, origin_text 57), not on
  CLIP_SECONDS.
- duck_and_cover produced 61 timesteps (re-encoding rounding). harmless.
- archive.org storage servers occasionally return HTTP 500. retry later.
  prepare() deletes empty files left behind by failed downloads.
- video encoding (V-JEPA2) takes ~6–9 min per 60 s clip on an A100.

licensing
---------
every stimulus is public domain or CC-BY. the `credit` field holds the
attribution line used in the README and the deployed app.
"""

# cell 1: drive + paths
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
LIB   = Path("/content/drive/MyDrive/tribe-app/library")  # drive backup of the library
CLIPS = LIB / "clips"          # 60 s clips, kept in case the app ever shows them
RAW   = Path("/content/raw")   # full downloads stay on local disk (disposable)
for d in (LIB, CLIPS, RAW, LIB / "texts"):
    d.mkdir(parents=True, exist_ok=True)
CLIP_SECONDS = 60

# cell 2: the stimulus list
BLENDER = "(CC) Blender Foundation | studio.blender.org"

NEW_STIMULI = [
    # ---------- video ----------
    {"id": "sprite_fright", "title": "Sprite Fright", "modality": "video",
     "url": "https://archive.org/download/sprite-fright/Sprite%20Fright%20-%20Open%20Movie%20by%20Blender%20Studio-804p.mp4",
     "start": 75,
     "prep": [], "notes": "horror-comedy animation; multi-speaker British dialogue",
     "credit": BLENDER},
    {"id": "cosmos_laundromat", "title": "Cosmos Laundromat", "modality": "video",
     "url": "https://archive.org/download/cosmos-laundromat-first-cycle/Cosmos%20Laundromat%20-%20First%20Cycle.mp4",
     "start": 180,  # after the opening scene
     "prep": [], "notes": "cinematic animation; sparse dialogue",
     "credit": BLENDER},
    {"id": "duck_and_cover", "title": "Duck and Cover (1951)", "modality": "video",
     "url": "https://archive.org/download/0771_Duck_and_Cover_12_33_20_12/0771_Duck_and_Cover_12_33_20_12.mp4",
     "start": 180,  # narrated live-action, past the song + cartoon
     "prep": [], "notes": "live-action with single narrator; speech and visuals tightly congruent",
     "credit": "Public domain (U.S. Federal Civil Defense Administration), via Prelinger Archives"},
    {"id": "trip_to_the_moon", "title": "A Trip to the Moon (1902)", "modality": "video",
     "url": "https://archive.org/download/a-trip-to-the-moon-1902-colorized/A%20Trip%20to%20the%20Moon%20%281902%29%20colorized.mp4",
     "start": 360,
     "prep": ["strip_audio"], "notes": "silent film, audio removed; pure-visual stimulus",
     "credit": "Public domain (Georges Méliès, 1902)"},
    {"id": "caminandes_2", "title": "Caminandes 2: Gran Dillama", "modality": "video",
     "url": "https://download.blender.org/demo/movies/caminandes_gran_dillama.mp4.zip",
     "start": 15,   # past the opening logo
     "prep": ["unzip"], "notes": "wordless slapstick animation",
     "credit": BLENDER},

    # ---------- audio ----------
    {"id": "raven_audio", "title": "The Raven (audio)", "modality": "audio",
     "url": "https://archive.org/download/ravenandpoemspoe_pc_librivox/ravenandotherpoems_01_poe_64kb.mp3",
     "start": 11,   # skips the librivox intro; starts at "Once upon a midnight dreary"
     "prep": [], "notes": "metered poetry, single reader",
     "credit": "Public domain, LibriVox (read by Phil Chenevert)"},
    {"id": "earnest_audio", "title": "The Importance of Being Earnest, Act 1 (audio)", "modality": "audio",
     "url": "https://archive.org/download/being_earnest_librivox/Earnest_Act_1_64kb.mp3",
     "start": 300,  # past the cast list + setting, into the dialogue
     "prep": [], "notes": "full-cast dramatic reading; turn-taking voices",
     "credit": "Public domain, LibriVox full-cast recording"},
    {"id": "verwandlung_audio", "title": "Die Verwandlung, Ch. 2 (German audio)", "modality": "audio",
     "url": "https://archive.org/download/verwandlung_mw_0811_librivox/verwandlung_02_kafka.mp3",
     "start": 30,   # past the german librivox disclaimer + chapter title
     "prep": [], "notes": "German narration; second non-English language",
     "credit": "Public domain, LibriVox"},
    {"id": "pastoral_audio", "title": "Beethoven, Pastoral Symphony II (audio)", "modality": "audio",
     "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Ludwig_van_Beethoven_-_symphony_no._6_in_f_major_%27pastoral%27%2C_op._68_-_ii._andante_molto_mosso.ogg",
     "start": 60,   # past the quiet opening
     "prep": [], "notes": "orchestral, no speech; tests auditory vs speech-specific STG",
     "credit": "Public domain, Musopen"},

    # ---------- text ----------
    {"id": "raven_text", "title": "The Raven (text excerpt)", "modality": "text",
     "file": "raven_text.txt", "notes": "matches raven_audio window",
     "credit": "Public domain, Project Gutenberg #8893"},
    {"id": "origin_text", "title": "On the Origin of Species, intro (text excerpt)", "modality": "text",
     "file": "origin_text.txt", "notes": "expository prose",
     "credit": "Public domain, Project Gutenberg"},
]

missing = [s["id"] for s in NEW_STIMULI if s["modality"] != "text" and s["start"] is None]
assert not missing, f"set a start time for: {missing}"

# cell 3: link check
import subprocess
for s in NEW_STIMULI:
    if "url" in s:
        r = subprocess.run(["wget", "-q", "--spider", s["url"]])
        print("ok  " if r.returncode == 0 else "FAIL", s["id"])

# cell 4: download, preprocess, clip
import subprocess, urllib.parse


def prepare(stim):
    """Download, preprocess and clip one stimulus; return the path TRIBE should read.

    Args:
        stim: an entry from NEW_STIMULI. Uses keys "id", "modality"
            ("video" | "audio" | "text"), and for media also "url",
            "start" (seconds) and "prep" (list of flags: "unzip",
            "strip_audio").

    Returns:
        Path to the 60 s clip in CLIPS (.mp4 for video, .mp3 for audio),
        or to the excerpt in LIB/texts/ for text stimuli.

    Raises:
        subprocess.CalledProcessError: download or unzip failed (empty
            download files are removed first).
        FileNotFoundError: an unzipped archive contained no .mp4.
        RuntimeError: ffmpeg failed; the message includes ffmpeg's stderr.
    """
    sid, mod = stim["id"], stim["modality"]

    # text: nothing to download or clip
    if mod == "text":
        return LIB / "texts" / stim["file"]

    # 1. download (skip if a non-empty copy already exists)
    suffix = Path(urllib.parse.unquote(stim["url"])).suffix   # .mp4 / .mp3 / .ogg / .zip
    raw = RAW / f"{sid}_raw{suffix}"
    if not raw.exists() or raw.stat().st_size == 0:
        try:
            subprocess.run(["wget", "-q", stim["url"], "-O", str(raw)], check=True)
        except subprocess.CalledProcessError:
            raw.unlink(missing_ok=True)   # don't leave an empty file behind
            raise

    # 2. unzip (caminandes only); the .mp4's name inside isn't known ahead of time
    if "unzip" in stim["prep"]:
        subprocess.run(["unzip", "-o", "-q", str(raw), "-d", str(RAW / sid)], check=True)
        videos = sorted((RAW / sid).glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No .mp4 files found in {RAW / sid}")
        raw = videos[0]

    # 3. clip. re-encode rather than -c copy, so a non-zero start cuts exactly.
    if mod == "video":
        clip = CLIPS / f"{sid}_clip.mp4"
        cmd = ["ffmpeg", "-y", "-ss", str(stim["start"]), "-i", str(raw),
               "-t", str(CLIP_SECONDS), "-c:v", "libx264", "-preset", "veryfast"]
        if "strip_audio" in stim["prep"]:
            cmd += ["-an"]
        else:
            cmd += ["-c:a", "aac"]
    else:  # audio: re-encode everything to mp3 (handles the librivox .mp3s and the musopen .ogg alike)
        clip = CLIPS / f"{sid}_clip.mp3"
        cmd = ["ffmpeg", "-y", "-ss", str(stim["start"]), "-i", str(raw),
               "-t", str(CLIP_SECONDS), "-c:a", "libmp3lame", "-q:a", "2"]

    r = subprocess.run(cmd + [str(clip)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {sid}:\n{r.stderr[-800:]}")
    return clip

# cell 5: inference for one stimulus
import numpy as np, torch


def run(stim):
    """Run TRIBE inference on one stimulus and save the prediction to drive.

    Skips the stimulus if LIB/{id}_preds.npy already exists, so the batch
    loop can be rerun safely after a crash.

    Args:
        stim: an entry from NEW_STIMULI. Its "modality" picks the keyword
            passed to model.get_events_dataframe (video_path / audio_path /
            text_path).

    Side effects:
        Writes LIB/{id}_preds.npy, shape (n_timesteps, 20484), and clears
        the CUDA cache.
    """
    npy = LIB / f"{stim['id']}_preds.npy"
    if npy.exists():                      # lets you rerun after a crash without redoing work
        print(f"skip {stim['id']}")
        return
    src = prepare(stim)
    # builds video_path= / audio_path= / text_path= from the modality field
    df = model.get_events_dataframe(**{f"{stim['modality']}_path": str(src)})
    preds, segments = model.predict(events=df)
    np.save(npy, preds)
    print(stim["id"], preds.shape)
    torch.cuda.empty_cache()

# cell 6: wordless gate first, then the full batch
by_id = {s["id"]: s for s in NEW_STIMULI}

# gate: the three stimuli with no speech for whisperx to align
failed = {}
for sid in ["caminandes_2", "trip_to_the_moon", "pastoral_audio"]:
    try:
        run(by_id[sid])
    except Exception as e:
        failed[sid] = repr(e)
print("gate failures:", failed)

# full batch: only run once the gate prints {}; finished stimuli are skipped
failed = {}
for s in NEW_STIMULI:
    try:
        run(s)
    except Exception as e:
        failed[s["id"]] = repr(e)
print("failures:", failed)

# cell 7: metadata rows for library_metadata.csv
import pandas as pd

rows = []
for s in NEW_STIMULI:
    npy = LIB / f"{s['id']}_preds.npy"
    if not npy.exists():
        print("missing:", s["id"])
        continue
    shape = np.load(npy, mmap_mode="r").shape   # reads the shape without loading the array
    rows.append({
        "id": s["id"], "title": s["title"], "notes": s["notes"],
        "npy_file": npy.name,
        "n_timesteps": shape[0], "n_vertices": shape[1],
        "credit": s["credit"],
    })

new_rows = pd.DataFrame(rows)
new_rows.to_csv(LIB / "new_rows.csv", index=False)
print(new_rows[["id", "n_timesteps", "n_vertices"]].to_string())