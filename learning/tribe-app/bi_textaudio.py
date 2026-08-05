# okay, this is batch inferencing -- multimodal edition
# i ran this in google colab with the model loaded as well, keep that in mind -- you can't run it locally

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# defining the stimuli

STIMULI = [
    {
        "id": "dorian_audio",
        "title": "The Picture of Dorian Gray: Ch. 13 (audio)",
        "modality": "audio",
        "url": "https://archive.org/download/picture_doriangray_1012_librivox/pictureofdoriangray_13_wilde_64kb.mp3",
        "notes": "English prose, spoken (LibriVox Reading)",
    },
    {
        "id": "aeneid_audio",
        "title": "The Aeneid (Dryden's Translation): Book VIII part 1 (audio)",
        "modality": "audio",
        "url": "https://archive.org/download/aeneid_0810_librivox1/aeneid_15_vergil_64kb.mp3",
        "notes": "English verse, spoken (LibriVox Reading)",
    },
    {
        "id": "italian_audio",
        "title": "Trionfo di Bacco e Arianna (audio)",
        "modality": "audio",
        "url": "https://archive.org/download/multilingual_short_works_collection_016_1612_librivox/msw016_10_trionfodibaccoearianna_demedici_f_64kb.mp3",
        "notes": "Non-English (Italian) Speech Experiment (LibriVox Reading)"

    },
    {
        "id": "dorian_text",
        "title": "The Picture of Dorian Gray: Ch. 13 (text)",
        "modality": "text",
        "text_file": "text_dorian.txt",
        "notes": "English prose, text (goes with dorian_audio)",
    },
    {
        "id": "aeneid_text",
        "title": "The Aeneid (Dryden's Translation): Book VIII part 1 (text)",
        "modality": "text",
        "text_file": "text_aeneid.txt",
        "notes": "English verse, text (goes with aeneid_audio)"
    }

]

CLIP_SECONDS = 60
OUT_DIR = Path("library")
OUT_DIR.mkdir(exist_ok = True)

metadata_rows = []

for stim in STIMULI:
  print(f"n\=== {stim['title']} ({stim['modality']}) ===")
  npy_path = OUT_DIR / f"{stim['id']}_preds.npy"

  try:
    if stim["modality"] in ("video", "audio"):
      ext = "mp4" if stim["modality"] == "video" else "mp3"
      raw_path = OUT_DIR / f"{stim['id']}_raw.{ext}"
      clip_path = OUT_DIR / f"{stim['id']}_clip.{ext}"

      print("downloading...")
      subprocess.run(["wget", "-q", "--timeout=30", "--tries=2", stim["url"], "-O", str(raw_path)], check=True)

      print(f"clipping to {CLIP_SECONDS}s...")
      subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_path),
                 "-ss", "0", "-t", str(CLIP_SECONDS), "-c", "copy", str(clip_path)],
                check=True, capture_output=True,
            )

      print("running inference...")
      if stim["modality"] == "video":
        df = model.get_events_dataframe(video_path = str(clip_path))
      else:
        df = model.get_events_dataframe(audio_path = str(clip_path))

    elif stim["modality"] == "text":
      text_path = stim["text_file"]
      print(f"reading text from {text_path}...")
      print("running inference...")
      df = model.get_events_dataframe(text_path = text_path)

    else:
      raise ValueError(f"unknown modality: {stim['modality']}")

    preds, segments = model.predict(events = df)
    np.save(npy_path, preds)
    print(f"saved {npy_path.name} with shape {preds.shape}")

    metadata_rows.append({
        "id": stim["id"],
        "title": stim["title"],
        "modality": stim["modality"],
        "notes": stim["notes"],
        "npy_file": npy_path.name,
        "n_timesteps": preds.shape[0],
        "n_vertices": preds.shape[1],
    })

  except Exception as e:
    print(f"failed on {stim['id']}: {e}!!!")

# append to existing metadata library
meta_path = OUT_DIR / "library_metadata.csv"
new_rows = pd.DataFrame(metadata_rows)

if meta_path.exists():
  existing = pd.read_csv(meta_path)
  if "modality" not in existing.columns:
    existing["modality"] = "video"
  combined = pd.concat([existing, new_rows], ignore_index = True)

else:
  combined = new_rows

combined.to_csv(meta_path, index = False)
print("n\=== DONE ===")
print(combined[["id", "title", "modality", "n_timesteps"]])