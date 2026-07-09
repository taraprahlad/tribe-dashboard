# tribe-dashboard
# tribe v2 interpretability dashboard ᯓ★

> 🚧 **work in progress** i'm building this as a side project to learn ML
> tooling, model interpretability, and neuroimaging visualization. it's going to
> be a little scrappy as it comes together — the roadmap below is for tracking the situation.

an interactive dashboard for exploring and interpreting [Meta's TRIBE v2](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/),
an open-source foundation model that predicts human brain responses (fmri
activation) to images, audio, and text. the project builds tooling *around*
tribe v2 — it doesn't modify or redistribute the model itself — to make its
predictions visible and to ask *why* it predicts what it does.

## why?

tribe v2 is a powerful model but kind of opaque: i want to be able to feed it a stimulus and get 
back predicted brain activity. this dashboard makes those predictions explorable — rendering
them on a 3D brain, comparing stimuli side by side, and using attribution
methods to understand which input features drive activation in specific regions.

## tech stack!

- **python** — core language
- **pytorch** — model inference and gradient-based attribution
- **nilearn** — brain atlas projection and neuroimaging visualization
- **captum** — interpretability methods (saliency, Integrated Gradients)
- **streamlit** — interactive web UI

## roadmap!

this is going to double as my progress tracker. boxes get checked as features land (yay!).

**foundations**
- [X] python / git / pytorch tune-up
- [X] fMRI & nilearn literacy

**core pipeline**
- [ ] tribe v2 running, inference on a single stimulus
- [ ] understand & document the model's output format
- [ ] first brain visualization from a prediction
- [ ] minimal Streamlit app: input → inference → brain render

**dashboard features**
- [ ] side-by-side stimulus comparison + difference view
- [ ] browsable stimulus library with cached predictions
- [ ] region drill-down (top-activating stimuli per region)
- [ ] multi-modal input (image / audio / text)

**interpretability**
- [ ] gradient saliency maps
- [ ] integrated gradients via captum
- [ ] input ablation
- [ ] cross-stimulus representation analysis

**ship**
- [ ] deploy (hugging face spaces / streamlit cloud)
- [ ] polish readme with screenshots
- [ ] write-up / blog post (might be better to link notion)

## let's get started!

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

**note on compute:** tribe v2 needs a gpu. i develop on google colab; running
locally requires a cuda-capable gpu. model weights are *not* included in this
repo — they're loaded at runtime from hugging face (`facebook/tribev2`).

## project structure!

```
tribe-dashboard/
├── app.py              # streamlit entry point
├── src/                # inference, visualization, interpretability modules
├── notebooks/          # exploratory work (Colab)
├── requirements.txt
├── .env.example        # template for secrets (no real values)
└── README.md
```

## license & attribution

- **tribe v2** is created by meta's FAIR team and released under **CC BY-NC**
  (non-commercial). this project uses it for personal, non-commercial,
  educational purposes. all credit for the model goes to meta.
- the **tooling code in this repo** is my own work.

## acknowledgments

built on Meta FAIR's [TRIBE v2](https://github.com/facebookresearch/tribev2).
brain visualization powered by [nilearn](https://nilearn.github.io/);
interpretability via [Captum](https://captum.ai/).
