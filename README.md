# tribe v2 interpretability dashboard

an interactive dashboard for poking at what meta fair's [tribe v2](https://huggingface.co/facebook/tribev2) brain-encoding model has learned: which brain regions it predicts will respond to a stimulus, when, and which inputs (text, audio or video) drive those predictions.

**live app → [tribe-v2-interpretability.streamlit.app](https://tribe-v2-interpretability.streamlit.app)**

![region explorer](docs/images/regions.png)

---

## what this is

tribe v2 takes a video, audio clip or text and predicts the fmri response across ~20k points on the cortical surface. out of the box, though, you get a giant array of anonymous numbers: nothing tells you *which regions* are responding, *what in the input* caused it, or whether standard interpretability tools can even be trusted on a model like this.

this project makes those predictions inspectable. it maps the model's vertices onto named anatomical regions (destrieux atlas), lets you browse and compare responses across a library of 20 stimuli, and runs attribution methods (occlusion and modality ablation) to ask what the model is actually relying on.

i built it as a 16-week, learn-by-doing project, going from "what is a tensor" to running captum attributions on a multimodal foundation model. 😌

## what you can do in the app

| page | what it shows |
|---|---|
| **library** | browse 20 stimuli (video, audio, text) and scrub through the predicted brain response over time |
| **compare timesteps** | two moments side by side, plus a difference map (with optional per-timestep mean removal) |
| **regions** | pick any destrieux region: see its predicted activity over the whole stimulus, and where it sits on the brain |
| **most active regions** | the inverse view: pick a moment and get the top-10 most active regions |
| **attribution** | which *moments* of a stimulus matter (time-window occlusion), and which *modality* each region leans on (modality ablation) |

![attribution page](docs/images/attribution.png)

![most active regions](docs/images/inverse.png)

## what i found

all of these are observations about tribe's *predicted* response on single clips, not claims about real brains. every effect is compared against a control region (postcentral sulcus, somatosensory), which serves as the noise floor.

- **plain gradient saliency isn't trustworthy here.** saliency maps for the superior temporal gyrus (stg, auditory/speech) and the occipital pole (visual) correlated at r = 0.88 across input timesteps. this means they were mostly measuring shared input magnitude, not region-specific drivers. integrated gradients with a clip-mean baseline removes that shared component.
- **the model's response to narration is region-selective.** occluding the real narration in a librivox recording shifts the predictions for stg and occipital, while the control region stays flat. one surprise: occluding narration *raises* stg's prediction, which is still an open question (see below).
- **there's no clean "one region, one modality" mapping.** on an audio-only clip, removing audio moves both stg *and* occipital the same way. on a live-action film clip (*tears of steel*), stg's prediction depends most on the **video** stream, and this was seen consistently across two different ablation baselines. occipital barely changes when any single modality is removed, which hints that the model can fill in from the other modalities.

![occlusion over time](docs/images/dorian_occlusion.png)

*occlusion on a dorian gray audiobook clip: after the librivox intro (~33 s), stg and occipital diverge from the flat control.*

### open questions
- why does occluding narration *raise* predicted stg? candidates: signed pooling over vertices and time cancelling effects out; the clip-mean baseline already containing speech; or the outputs being relative to a baseline.
- is stg's dependence on video about visual speech (faces, lip movement)? that's a hypothesis, not something shown here.
- is occipital's insensitivity a sign of redundancy across modalities? the test would be ablating modalities in *pairs*.

## how it works

```
colab (gpu)                                   streamlit app (cpu, local or hosted)
───────────                                   ────────────────────────────────────
stimulus ──► tribe v2 inference ──► .npy ──►  library / compare / regions / inverse
          └► captum attribution ──► .json/.npy ──►  attribution page
```

inference and attribution need a gpu, and tribe pulls in llama 3.2, v-jepa2 and wav2vec-bert, so they run offline in colab and save their results. the app only reads those precomputed files, which is why it can run on a free cpu host and stay responsive.

i didn't plan this split initially. my first version actually ran streamlit *inside* colab through a tunnel, but it kept hitting segfaults, port collisions and compute limits. separating the gpu work from the interactive part fixed all three.

## run it locally

needs python 3.11+.

```bash
git clone https://github.com/taraprahlad/tribe-dashboard.git
cd tribe-dashboard/learning/tribe-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

note: always launch `app.py`, not a file in `pages/`: the pages import shared helpers from `tribe_shared.py`.

## regenerate the predictions (colab)

you only need this to add stimuli or rerun attributions. the scripts in `colab/` are colab-only and won't run locally.

- `colab/batch_expand.py`: downloads, clips and runs tribe inference on a stimulus list, saving `{id}_preds.npy` plus metadata rows
- `colab/attribution_runs.py`: the region × modality ablation and time-window occlusion behind the attribution page

the environment is VERY fragile. what worked: an a100 runtime pinned to python 3.12.13, cuda torch 2.6.0 (cu124 wheels), and numpy 2.2.6 / sympy 1.13.1 installed **last** with `--force-reinstall --no-deps`, then a restart before importing anything. the details and failure signatures are in [`understanding-tribe.md`](../understanding-tribe.md).

## repo layout

```
tribe-app/
├── app.py              entry point: crash guards + page navigation
├── tribe_shared.py     paths and cached loaders shared by every page
├── pages/              library, compare, regions, inverse, attribution
├── colab/              gpu-side scripts (inference, attribution)
├── library/            20 prediction arrays, metadata (with credits), atlas, attribution results
└── docs/images/        screenshots and figures
```

## stimuli and credits

all 20 stimuli are public domain or creative commons. the exact credit line for each one is in `library/library_metadata.csv` and shown in the app.

- **blender foundation open movies** (cc-by, (cc) blender foundation | studio.blender.org): sintel, big buck bunny, tears of steel, elephants dream, sprite fright, cosmos laundromat, caminandes 2
- **librivox** (public domain recordings): dorian gray, the aeneid, the raven, the importance of being earnest, die verwandlung
- **project gutenberg** (public domain texts): dorian gray, the aeneid, the raven, on the origin of species
- **musopen** (public domain): beethoven, pastoral symphony, 2nd movement
- **prelinger archives** (public domain, u.s. government work): duck and cover (1951)
- **georges méliès** (public domain): a trip to the moon (1902), shown with its audio removed

**model:** tribe v2 by meta fair. [weights and model card](https://huggingface.co/facebook/tribev2) · [code](https://github.com/facebookresearch/tribev2). the model is released under **cc by-nc 4.0**, so the predictions in this repo are model outputs under that non-commercial license.

**built with:** streamlit, captum, nilearn, pytorch, numpy, pandas, matplotlib, whisperx.

## caveats

- predictions are for tribe's "average" subject, on the fsaverage5 mesh. brain renders display the left hemisphere.
- attribution results come from single clips and single regions pooled into one signed number, so make sure to read them as observations, and not general claims.
- whisper "hears" words in wordless clips (it found 3 in the beethoven piece), so the text modality isn't completely empty for non-speech stimuli.
- the app sleeps after 12 hours without visitors. if you see a wake-up button, give it a minute and thank you so much for taking a look at this. 🥹