# new skill! batch inference -- this is when you generate ml/ai predictions for large datasets offline at once as
#opposed to processing them real-time.

# this was run in google colab with the model loaded

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# i will now define the library so that each stimulus is labeled with a short id, title, video url,
# and notes about what it exercises (this data will then become the browsable library).

STIMULI = [
    {
        "id" :"sintel",
        "title" : "Sintel",
        "url" : "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
        "notes" : "fantasy animation! it contains rich visuals, an orchestral score and dialogue."
    },
    {
        "id" : "big_buck_bunny",
        "title" : "Big Buck Bunny",
        "url" : "https://archive.org/download/BigBuckBunny_328/BigBuckBunny_512kb.mp4",
        "notes" : "bright cartoon comedy -- musical and almost wordless",
    },
    {
        "id" : "tears_of_steel",
        "title" : "Tears of Steel",
        "url" : "https://archive.org/download/tweakers006325/tweakers006325.mp4",
        "notes" : "live action sci-fi with VFX! contains real human faces",
    },
    {
        "id" : "elephants_dream",
        "title" : "Elephants Dream",
        "url" : "https://archive.org/download/ElephantsDream/ed_1024.mp4",
        "notes" : "surreal abstract animation -- dialogue-heavy with unusual visuals",
    },
    {
        "id" : "caminandes",
        "title" : "Caminandes: Llama Drama",
        "url" : "https://archive.org/download/Caminandes1LlamaDrama/01_llama_drama_1080p.mp4",
        "notes" : "slapstick animation short -- wordless and comedic",
    }
]

CLIP_SECONDS = 60
OUT_DIR = Path("library")
OUT_DIR.mkdir(exist_ok = True)

metadata_rows = []

for stim in STIMULI:
  print(f"\n=== {stim['title']} ===")
  raw_path = OUT_DIR / f"{stim['id']}_raw.mp4"
  clip_path = OUT_DIR / f"{stim['id']}_clip.mp4"
  npy_path = OUT_DIR / f"{stim['id']}_preds.npy"

  try:
    # download vid
    print("downloading...")
    subprocess.run(
        ["wget", "-q", stim["url"], "-0", str(raw_path)],
        check = True,
    )

    # clip to first 60 seconds with ffmpeg
    print(f"clipping to {CLIP_SECONDS}s...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path),
         "-ss", "0", "-t", str(CLIP_SECONDS),
         "-c", "copy", str(clip_path)],
        check = True,
        capture_output = True,
    )

    # run tribe inference
    print("running inference...")
    df = model.get_events_dataframe(video_path = str(clip_path))
    preds, segments = model.predict(events = df)

    # save the prediction
    np.save(npy_path, preds)
    print(f"saved {npy_path.name} with shape {preds.shape}")

    # record metadata (note the actual timestep count per stimulus)
    metadata_rows.append({
        "id" : stim["id"],
        "title" : stim["title"],
        "notes" : stim["notes"],
        "npy_file" : npy_path.name,
        "n_timesteps" : preds.shape[0],
        "n_vertices" :preds.shape[1],
    })

  except Exception as e:
    # bc we don't want one failure to kill my baby
    print(f"failed on {stim['id']} : {e}!")

# writing the metadata table
meta_df = pd.DataFrame(metadata_rows)
meta_df.to_csv(OUT_DIR / "library_metadata.csv", index = False)
print("\n=== DONE! ===")
print(meta_df)
