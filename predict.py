"""Generate Cover_Type predictions from the saved COMP663 Assignment 2 model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list[int], activation: str):
        super().__init__()
        activation_layer = getattr(nn, activation)
        layers: list[nn.Module] = []
        previous = input_size
        for width in hidden_layers:
            layers.extend([nn.Linear(previous, width), activation_layer()])
            previous = width
        layers.append(nn.Linear(previous, 5))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def predict(input_path: Path, output_path: Path, model_path: Path) -> None:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    features = checkpoint["feature_names"]
    frame = pd.read_csv(input_path)
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input feature columns: {sorted(missing)}")

    x = frame[features].astype("float32").copy()
    continuous = checkpoint["continuous_features"]
    x[continuous] = (x[continuous] - np.array(checkpoint["scaler_mean"])) / np.array(
        checkpoint["scaler_scale"]
    )

    architecture = checkpoint["architecture"]
    model = MLP(len(features), architecture["hidden_layers"], architecture["activation"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x.to_numpy(), dtype=torch.float32))
        labels = logits.argmax(dim=1).numpy() + 1

    pd.DataFrame({"Cover_Type": labels}).to_csv(output_path, index=False)
    print(f"Wrote {len(labels)} predictions to {output_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    args = [Path(arg) for arg in sys.argv[1:]]
    input_csv, output_csv, model_file = args or [
        root / "data" / "forest_cover_data.csv",
        root / "data" / "predictions.csv",
        root / "models" / "1173808_Assignment2_final.pt",
    ]
    if len(args) not in (0, 3):
        raise SystemExit("Usage: python predict.py INPUT.csv OUTPUT.csv MODEL.pt")
    predict(input_csv, output_csv, model_file)
