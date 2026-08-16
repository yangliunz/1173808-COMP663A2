#!/usr/bin/env python
# coding: utf-8

# # COMP663 Assignment 2 — Classical Optimisation
# 
# **Student ID:** 1173808  
# **Dataset:** `forest_cover_data.csv`  
# **Primary metric:** macro-F1
# 

# ## ENV & Libs Setup
# 

# In[23]:


from pathlib import Path
import ast
import random
import time
setup_started = time.perf_counter()

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from IPython.display import display
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    ParameterSampler,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42
if Path("/kaggle").exists() and not torch.cuda.is_available():
    raise RuntimeError("Kaggle GPU was not allocated; stop instead of running on CPU.")
DEVICE = torch.device(
      "mps" if torch.backends.mps.is_available()
      else "cuda" if torch.cuda.is_available()
      else "cpu"
  )
torch.set_num_threads(4)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd()
KAGGLE_DATA_PATH = Path("/kaggle/input/datasets/yangliunz/comp663-a2-forest-cove/forest_cover_data.csv")
kaggle_data_paths = ([KAGGLE_DATA_PATH] if KAGGLE_DATA_PATH.exists() else []) + list(Path("/kaggle/input").rglob("forest_cover_data.csv"))
DATA_PATH = kaggle_data_paths[0] if kaggle_data_paths else ROOT / "data" / "forest_cover_data.csv"
FIGURE_PATH = ROOT / "figures" / "performance_comparison.png"
MODEL_PATH = ROOT / "models" / "1173808_Assignment2_final.pt"
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20
SEARCH_EPOCHS = 20
FINAL_EPOCHS = 100
RANDOM_TRIALS = 12
BAYESIAN_TRIALS = 12
NAS_TRIALS = 12

if DEVICE.type == "cuda":
    RUNTIME_DEVICE = f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"
elif DEVICE.type == "mps":
    RUNTIME_DEVICE = "Apple Silicon MPS"
else:
    RUNTIME_DEVICE = "CPU"

def format_decimal(value):
    return np.format_float_positional(float(value), unique=True, trim="-")

def log_cell(name, started, configuration):
    print(f"{name}: device={RUNTIME_DEVICE}; configuration={configuration}; elapsed_seconds={format_decimal(time.perf_counter() - started)}")

log_cell("Environment setup", setup_started, f"torch={torch.__version__}, seed={SEED}, data={DATA_PATH}")


# ## Task 1 — Baseline model and candidate hyperparameters
# 
# ### 1.1 Preprocessing pipeline and feature engineering
# 
# As we found during assginment 1:
# 
# | Decision                 | Evidence from EDA                                                                                                                      | Action                                                                                                       | Reason / trade-off                                                                                                                                                                                              |
# | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | Missing values           | Zero missing values across all 15 columns (confirmed in Task 1.1)                                                                      | pass through                                                                                                 | Imputing when there is nothing to impute would add unnecessary complexity and risk introducing artificial patterns.                                                                                             |
# | Scaling / transformation | Distance features have large ranges and outliers (Task 1.3)                                                                            | Apply `StandardScaler` to the 10 continuous features and leave the 4 binary Wilderness_Area columns unscaled | Scaling puts continuous features on a comparable range. The fitted transformation stays inside the validation pipeline, which prevents leakage. Some models are notdistance-sensitive which will discuss later. |
# | Feature engineering      | Elevation, distance features, and wilderness area already show useful class separation; wilderness columns are already one-hot encoded | No additional feature engineering                                                                            | The existing features already contain useful information. Adding polynomial features would increase complexity without EDA evidence that they are needed.                                                       |
# | Class imbalance          | The largest and smallest classes have a 31:1 ratio (Task 1.2)                                                                          | Use macro-F1 and balanced accuracy for evaluation                                                            | These metrics make minority-class performance visible without changing the baseline training procedure.                                                                                                           |
# 
# 
# 

# #### Load data and check data integrity
# 

# In[24]:


# load data from CSV file
cell_started = time.perf_counter()
data = pd.read_csv(DATA_PATH)
target = "Cover_Type"

# filter out rows with missing values in the target column
data = data.dropna(subset=[target])
feature_names = [column for column in data.columns if column != target]
continuous_features = [
    column for column in feature_names if not column.startswith("Wilderness_Area")
]

# check data integrity
assert data.shape == (571_012, 15), data.shape
assert len(feature_names) == 14
assert set(data[target].unique()) == {1, 2, 3, 4, 5}
assert data.isna().sum().sum() == 0

# print data shape, feature names, and target value counts with percentages
print("Shape:", data.shape)
print("Features:", feature_names)
display(
    data[target]
    .value_counts()
    .sort_index()
    .rename("count")
    .to_frame()
    .assign(percentage=lambda frame: 100 * frame["count"] / len(data))
)
log_cell("Data loading", cell_started, f"data={DATA_PATH}")


# #### Data set split

# In[25]:


# split data into training, validation, and test sets
cell_started = time.perf_counter()
train_validation_frame, test_frame = train_test_split(
    data, test_size=TEST_FRACTION, stratify=data[target], random_state=SEED
)

train_frame, validation_frame = train_test_split(
    train_validation_frame,
    test_size=VALIDATION_FRACTION / (TRAIN_FRACTION + VALIDATION_FRACTION),
    stratify=train_validation_frame[target],
    random_state=SEED,
)

# validate the size of the splits and print the number of samples in each set
assert len(train_frame) + len(validation_frame) + len(test_frame) == len(data)
print(
    f"Train / validation / test: {len(train_frame):,} / {len(validation_frame):,} / {len(test_frame):,}"
)
log_cell("Data split", cell_started, "train=0.6, validation=0.2, test=0.2")


# ### 1.2 Primary and secondary evaluation metrics
# 
# As this dataset have a strong imblanaced class,
# 
# **Macro-F1** will be the primary metric to cover type has equal importance.
# 
# **Balanced accuracy** is the secondary metric because it reflect the model recalls on each class equally.
# 

# ### 1.3 Baseline model training and evaluation procedure
# 
# - (1) Initial the baseline mode with configuration in assignment requirement.
# 
# - (2) traning baseline model and compute metrics on validation dataset.
# 

# In[26]:


# Define the fixed baseline architecture required by the assignment.
cell_started = time.perf_counter()
class BaselineNN(nn.Module):
    """The architecture supplied in baselineNN.ipynb."""

    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 24),  # 14 input features to 24 hidden units
            nn.Sigmoid(),               # required hidden-layer activation
            nn.Linear(24, 12),          # second hidden layer
            nn.Sigmoid(),               # required hidden-layer activation
            nn.Linear(12, 5),           # one logit for each Cover_Type class
        )

    def forward(self, x):
        return self.layers(x)


# Create a fresh baseline model on the selected device.
baseline_model = BaselineNN(len(feature_names)).to(DEVICE)
print(baseline_model)
print(
    f"Parameters: {sum(parameter.numel() for parameter in baseline_model.parameters()):,}"
)
# Check the supplied architecture: weights and biases total 725 trainable parameters.
assert (
    sum(parameter.numel() for parameter in baseline_model.parameters()) == 725
), "Parameter count mismatch"
log_cell("Baseline architecture", cell_started, "layers=14-24-12-5, activation=Sigmoid, parameters=725")


# In[28]:


# Train the baseline model and evaluate it on the validation split.
cell_started = time.perf_counter()
BASELINE_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 512,
    "epochs": SEARCH_EPOCHS,
}

# Fit scaling values on training data only to prevent data leakage.
scaler = StandardScaler().fit(train_frame[continuous_features])


def prepare_baseline_data(frame):
    # Keep the original 14 feature columns and use float32 for PyTorch.
    features = frame[feature_names].astype("float32").copy()
    # Scale only continuous features; Wilderness_Area columns are already 0/1.
    features[continuous_features] = scaler.transform(features[continuous_features])
    # Change class labels from 1–5 to the 0–4 indices required by CrossEntropyLoss.
    return features.to_numpy(), frame[target].to_numpy(dtype=np.int64) - 1


# Apply the training-fitted scaler to both training and validation features.
x_train, y_train = prepare_baseline_data(train_frame)
x_validation, y_validation = prepare_baseline_data(validation_frame)

# Keep the supplied baseline loss unchanged.
loss_fn = nn.CrossEntropyLoss()
# Adam updates the baseline parameters using the fixed Task 1 settings.
optimizer = torch.optim.Adam(
    baseline_model.parameters(),
    lr=BASELINE_CONFIG["learning_rate"],
)

# Move the training arrays to PyTorch tensors once before the epoch loop.
x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
# Use the fixed seed so the mini-batch order is reproducible.
generator = torch.Generator(device=DEVICE).manual_seed(SEED)

# Training mode enables gradient calculation and parameter updates.
baseline_model.train()
for _ in range(BASELINE_CONFIG["epochs"]):
    # Shuffle the training rows once per epoch.
    order = torch.randperm(len(x_train_tensor), generator=generator, device=DEVICE)
    for start in range(0, len(order), BASELINE_CONFIG["batch_size"]):
        batch_index = order[start : start + BASELINE_CONFIG["batch_size"]]
        optimizer.zero_grad()  # clear gradients from the previous mini-batch
        loss = loss_fn(
            baseline_model(x_train_tensor[batch_index]), y_train_tensor[batch_index]
        )
        loss.backward()  # compute gradients by backpropagation
        optimizer.step()  # update weights and biases

# Evaluation mode and no_grad disable training updates for validation.
baseline_model.eval()
with torch.no_grad():
    validation_logits = baseline_model(
        torch.tensor(x_validation, dtype=torch.float32, device=DEVICE)
    )

# Use softmax to convert logits to probabilities, then take the argmax to get predictions.
probabilities = torch.softmax(validation_logits, dim=1)
validation_prediction = probabilities.argmax(dim=1).cpu().numpy()

# calculate macro-F1 and balanced accuracy scores for the validation set
validation_macro_f1 = f1_score(y_validation, validation_prediction, average="macro")
validation_balanced_accuracy = balanced_accuracy_score(
    y_validation, validation_prediction
)
# Metrics must be valid scores between zero and one.
assert 0 <= validation_macro_f1 <= 1
print("Validation macro-F1:", validation_macro_f1)
print("Validation balanced accuracy:", validation_balanced_accuracy)
log_cell("Baseline training and validation", cell_started, f"learning_rate={format_decimal(BASELINE_CONFIG['learning_rate'])}, batch_size={BASELINE_CONFIG['batch_size']}, epochs={BASELINE_CONFIG['epochs']}")


# **Baseline Model metrics on validation data set**
# 
# | Model      | Validation macro-F1 | Validation balanced accuracy |
# | ---------- | ------------------: | ---------------------------: |
# | BaselineNN |  0.5166125906224147 |           0.4830609876221151 |
# 

# ### 1.4 Unoptimised baseline performance using held-out test data
# 

# In[11]:


# evaluate the unoptimised baseline on the held-out test set
cell_started = time.perf_counter()
x_test, y_test = prepare_baseline_data(test_frame)

baseline_model.eval()
with torch.no_grad():
    test_logits = baseline_model(
        torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
    )
test_probabilities = torch.softmax(test_logits, dim=1)
test_prediction = test_probabilities.argmax(dim=1).cpu().numpy()

test_macro_f1 = f1_score(y_test, test_prediction, average="macro")
test_balanced_accuracy = balanced_accuracy_score(y_test, test_prediction)
assert 0 <= test_macro_f1 <= 1
print("Test macro-F1:", test_macro_f1)
print("Test balanced accuracy:", test_balanced_accuracy)
log_cell("Baseline test evaluation", cell_started, "model=BaselineNN")


# \*\*Baseline Model metrics on Test dataset
# | Model | Test macro-F1 | Test balanced accuracy |
# |---|---:|---:|
# | BaselineNN | 0.4717161780992363 | 0.7504576390688007 |
# 

# ### 1.5 Candidate hyperparameters
# 
# - (1) Learning rate: step size and convergence stability.
# - (2) Batch size: size of updates per epoch and effect training cost.
# - (3) Weight decay: regularises weights and may reduce overfitting.
# - (4) Epochs: controls the training budget and convergence time.
# - (5) NAS architectural design choices — number of hidden layers, hidden-layer widths, and activation functions: change model capacity and parameter count.
# 

# ### 1.6 Hyperparameter types and search ranges
# 
# | Hyperparameter | Type and range | Expected effect / cost |
# |---|---|---|
# | Learning rate | log continuous, 0.0001–0.01 | Controls update size; too large can be unstable. |
# | Batch size | categorical: 256, 512, 1024 | Larger batches use fewer updates per epoch. |
# | Weight decay | log continuous, 0.00001–0.001 | Regularises weights; too much can underfit. |
# | Epochs | fixed at 20 per trial | Keeps the comparison within a practical budget. |
# | Number of hidden layers | NAS search space: categorical 1, 2, or 3 | Changes model depth and parameter count. |
# | Hidden-layer width | NAS search space: categorical 16, 24, 32, 48, or 64 | Changes model capacity and training cost. |
# | Activation function | NAS search space: categorical Sigmoid or ReLU | Changes non-linear behaviour and convergence. |
# 

# ## Task 2 — Classical hyperparameter search
# 

# ### 2.1 Selected Random Search method and justification
# 
# Choose random search as it covers the learning-rate and weight-decay ranges more efficiently than a small grid since both ranges are meaningful on a log scale.
# 

# ### 2.2 Hyperparameters to optimise
# 
# Random search will focus on optimising `learning rate`, `batch size`, and `weight decay` from are the Task 1 candidates;
# `Epochs` and other architecture hyperparameters will be fixed during task 2 for a fair and justified comparision.
# 

# ### 2.3 Computational budget and justification
# 
# The budget is fixed as 12 trials in 20 epochs.
# This is small enough for CPU execution while testing different training settings.
# 

# ### 2.4 Apply Random Search
# Apply random search based on the task 1.6 hyperparameters table by 12 trails

# In[7]:


# Task 2 keeps the Task 1 baseline architecture fixed.
cell_started = time.perf_counter()
def train_and_evaluate(config):
    torch.manual_seed(SEED)
    model = BaselineNN(len(feature_names)).to(DEVICE)

    # Reuse the Task 1 train/validation arrays and training-fitted scaler.
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    model.train()
    for _ in range(config["epochs"]):
        order = torch.randperm(len(x_train_tensor), generator=generator, device=DEVICE)
        for start in range(0, len(order), config["batch_size"]):
            batch_index = order[start : start + config["batch_size"]]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train_tensor[batch_index]), y_train_tensor[batch_index])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_validation, dtype=torch.float32, device=DEVICE))
        probabilities = torch.softmax(logits, dim=1)
        prediction = probabilities.argmax(dim=1).cpu().numpy()

    return (
        f1_score(y_validation, prediction, average="macro"),
        balanced_accuracy_score(y_validation, prediction),
    )


# Random search samples configurations without evaluating every grid combination.
random_space = {
    "learning_rate": [0.0001, 0.001, 0.01],
    "batch_size": [256, 512, 1024],
    "weight_decay": [0.0, 0.00001, 0.0001, 0.001],
}
random_rows = []
for trial_number, sampled in enumerate(
    ParameterSampler(random_space, n_iter=RANDOM_TRIALS, random_state=SEED), 1
):
    config = {**sampled, "epochs": SEARCH_EPOCHS}
    print(f"Task 2 trial {trial_number}/{RANDOM_TRIALS}: learning_rate={format_decimal(config['learning_rate'])}, batch_size={config['batch_size']}, weight_decay={format_decimal(config['weight_decay'])}, epochs={config['epochs']}", flush=True)
    started = time.perf_counter()
    macro_f1, balanced_accuracy = train_and_evaluate(config)
    row = {
        "trial": trial_number,
        **config,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "parameters": 725,
        "seconds": time.perf_counter() - started,
    }
    random_rows.append(row)
    print(f"Task 2 trial {trial_number} result: macro-F1={format_decimal(row['macro_f1'])}, balanced accuracy={format_decimal(row['balanced_accuracy'])}, seconds={format_decimal(row['seconds'])}", flush=True)

random_table = pd.DataFrame(random_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
best_random = random_table.iloc[0].to_dict()
print("Task 2 complete:")
print(random_table.to_string(index=False, float_format=lambda value: np.format_float_positional(float(value), unique=True, trim="-")))
log_cell("Task 2 random search", cell_started, f"trials={RANDOM_TRIALS}, epochs_per_trial={SEARCH_EPOCHS}")


# ### 2.5 Evaluation procedure and primary metric
# 
# Every trial trains on the same 60% training dataset.
# Generate metrics macro-f1 balanced_accuracy only on same vlidation dataset.
# The scaler is only appied on training dataset only to prevent data leakage.
# 

# ### 2.6 Best configuration, score, and search time
# 
# | Item                                 |              Value |
# | ------------------------------------ | -----------------: |
# | Learning rate                        |               0.01 |
# | Batch size                           |                512 |
# | Weight decay                         |            0.00001 |
# | Epochs                               |                 20 |
# | Validation macro-F1                  |  0.620486164019339 |
# | Validation balanced accuracy         | 0.5615715029443511 |
# | Search time on Tesla T4 x2 (seconds) | 220.10828787800006 |
# 

# ### 2.7 Search results table
# 
# | Rank | Trial | Learning rate | Batch size | Weight decay | Macro-F1 | Balanced accuracy | Seconds |
# |---:|---:|---:|---:|---:|---:|---:|---:|
# | 1 | 7 | 0.01 | 512 | 0.00001 | 0.620486164019339 | 0.5615715029443511 | 19.434950119999996 |
# | 2 | 9 | 0.01 | 256 | 0.0 | 0.6058210695912665 | 0.5496740507572219 | 37.53823663899999 |
# | 3 | 11 | 0.01 | 256 | 0.00001 | 0.6042274044597084 | 0.5567435507062999 | 38.28666197899997 |
# | 4 | 12 | 0.01 | 1024 | 0.0001 | 0.526955832979515 | 0.4913293427281696 | 9.93340552199993 |
# | 5 | 5 | 0.001 | 512 | 0.0 | 0.45214492618835067 | 0.43567754133121106 | 18.69509913899998 |
# | 6 | 10 | 0.001 | 512 | 0.00001 | 0.4460991064559261 | 0.43167661114895833 | 19.573229812000022 |
# | 7 | 4 | 0.001 | 1024 | 0.0001 | 0.41715277351203195 | 0.4036995878770145 | 9.667997221999997 |
# | 8 | 1 | 0.01 | 1024 | 0.001 | 0.3631319448186237 | 0.35042005402198173 | 9.700595045 |
# | 9 | 6 | 0.001 | 1024 | 0.001 | 0.3138612797455258 | 0.3175054099652919 | 9.57150824300004 |
# | 10 | 8 | 0.0001 | 512 | 0.0 | 0.2971883716053794 | 0.3090180104120446 | 18.66795302700001 |
# | 11 | 2 | 0.0001 | 512 | 0.00001 | 0.29717649858291145 | 0.309005325363587 | 19.385552362999988 |
# | 12 | 3 | 0.0001 | 1024 | 0.0001 | 0.2947015511563337 | 0.30636080388925907 | 9.62775801700002 |
# 
# The table shows that learning rate = 0.01 gives the strongest trials, but its result still depends on batch size and weight decay.
# 

