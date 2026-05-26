# tribe-dashboard
# TRIBE v2 Interpretability Dashboard

> 🚧 **work in progress.** i'm building this as a side project to learn ML
> tooling, model interpretability, and neuroimaging visualization. it's going to
> be a little scrappy as it comes together — the roadmap below is for tracking the situation.

an interactive dashboard for exploring and interpreting [Meta's TRIBE v2](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/),
an open-source foundation model that predicts human brain responses (fMRI
activation) to images, audio, and text. the project builds tooling *around*
TRIBE v2 — it doesn't modify or redistribute the model itself — to make its
predictions visible and to ask *why* it predicts what it does.

## Why

tribe v2 is a powerful model but kind of opaque: i want to be able to feed it a stimulus and get 
back predicted brain activity. this dashboard makes those predictions explorable — rendering
them on a 3D brain, comparing stimuli side by side, and using attribution
methods to understand which input features drive activation in specific regions.

## tech stack

- **Python** — core language
- **PyTorch** — model inference and gradient-based attribution
- **nilearn** — brain atlas projection and neuroimaging visualization
- **Captum** — interpretability methods (saliency, Integrated Gradients)
- **Streamlit** — interactive web UI

## Roadmap

this doubles as my progress tracker. boxes get checked as features land (yay!).

**Foundations**
- [X] Python / Git / PyTorch tune-up
- [ ] fMRI & nilearn literacy

**Core pipeline**
- [ ] tRIBE v2 running, inference on a single stimulus
- [ ] understand & document the model's output format
- [ ] first brain visualization from a prediction
- [ ] minimal Streamlit app: input → inference → brain render

**Dashboard features**
- [ ] side-by-side stimulus comparison + difference view
- [ ] browsable stimulus library with cached predictions
- [ ] region drill-down (top-activating stimuli per region)
- [ ] multi-modal input (image / audio / text)

**Interpretability**
- [ ] gradient saliency maps
- [ ] integrated Gradients via Captum
- [ ] input ablation
- [ ] cross-stimulus representation analysis

**Ship**
- [ ] deploy (Hugging Face Spaces / Streamlit Cloud)
- [ ] polish README with screenshots
- [ ] write-up / blog post

## Getting started

> ⚠️ instructions are a placeholder right now and will firm up as the project does.

```bash
# Clone and set up a virtual environment
git clone https://github.com/<your-username>/tribe-dashboard.git
cd tribe-dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure secrets (copy the example and fill in your own values)
cp .env.example .env

# Run the app
streamlit run app.py
```

**note on compute:** TRIBE v2 needs a GPU. i develop on google colab; running
locally requires a CUDA-capable GPU. model weights are *not* included in this
repo — they're loaded at runtime from Hugging Face (`facebook/tribev2`).

## Project structure

```
tribe-dashboard/
├── app.py              # streamlit entry point
├── src/                # inference, visualization, interpretability modules
├── notebooks/          # exploratory work (Colab)
├── requirements.txt
├── .env.example        # template for secrets (no real values)
└── README.md
```

## License & attribution

- **TRIBE v2** is created by Meta's FAIR team and released under **CC BY-NC**
  (non-commercial). this project uses it for personal, non-commercial,
  educational purposes. all credit for the model goes to Meta.
- the **tooling code in this repo** is my own work.

## Acknowledgments

built on Meta FAIR's [TRIBE v2](https://github.com/facebookresearch/tribev2).
brain visualization powered by [nilearn](https://nilearn.github.io/);
interpretability via [Captum](https://captum.ai/).
