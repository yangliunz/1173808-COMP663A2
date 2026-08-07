# COMP663 Assignment 2 — Classical Optimisation

Student: **1173808**  
Due: **21 August 2026, 5:00 p.m.**

## Structure

```
1173808-COMP663A2/
├── data/          # Put forest_cover_data.csv here (ignored by Git)
├── figures/       # Notebook figures
├── models/        # Final .pt or .pth model (ignored by Git)
├── notebooks/
│   └── 1173808_Assignment2.ipynb
├── submission/    # Local packaging workspace (ignored by Git)
├── requirements.txt
├── run.txt
└── GenAI_Acknowledgement.txt
```

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/1173808_Assignment2.ipynb
```

The supplied baseline network is in the notebook. Keep all project paths relative to `notebooks/`: `../data/`, `../figures/`, and `../models/`.

## Submission

Before submission, commit the executed notebook, final `.pt`/`.pth` model, `run.txt`, and `GenAI_Acknowledgement.txt`, then add the repository URL to `GitHub_URL.txt` for Akoraka.
