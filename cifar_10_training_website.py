import copy
import time
from dataclasses import dataclass

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import pandas as pd


# =========================
# Page setup
# =========================
st.set_page_config(page_title="CIFAR-10 ResNet18 Trainer", layout="wide")
st.title("CIFAR-10 ResNet18 Training Website")
st.caption("A web app version of your notebook that runs the same overtraining and regularized experiments.")


# =========================
# Basic config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ExperimentConfig:
    regularize: bool
    overfit_subset: bool
    freeze_backbone: bool
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int


@st.cache_resource(show_spinner=False)
def get_datasets(regularize: bool):
    if regularize:
        train_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    val_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=val_transform)
    classes = train_dataset.classes
    return train_dataset, val_dataset, classes


def get_cifar10_loaders(regularize: bool, overfit_subset: bool, batch_size: int):
    base_train, val_dataset, classes = get_datasets(regularize)

    train_dataset = base_train
    if overfit_subset:
        small_size = int(0.15 * len(base_train))
        train_dataset = Subset(base_train, list(range(small_size)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, classes


def build_model(freeze_backbone: bool):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    return model.to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def run_experiment(config: ExperimentConfig):
    train_loader, val_loader, classes = get_cifar10_loaders(
        regularize=config.regularize,
        overfit_subset=config.overfit_subset,
        batch_size=config.batch_size,
    )

    model = build_model(freeze_backbone=config.freeze_backbone)
    criterion = nn.CrossEntropyLoss()

    if config.regularize:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    counter = 0
    history = []

    progress_bar = st.progress(0)
    status = st.empty()

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "gap": train_acc - val_acc,
            }
        )

        status.info(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
            f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}"
        )
        progress_bar.progress((epoch + 1) / config.epochs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if config.regularize and counter >= config.patience:
                status.warning("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    results = pd.DataFrame(history)
    return model, results, classes


def summarize_results(results: pd.DataFrame):
    last_row = results.iloc[-1]
    best_val_acc_row = results.loc[results["val_acc"].idxmax()]
    summary = {
        "epochs_ran": int(len(results)),
        "final_train_acc": float(last_row["train_acc"]),
        "final_val_acc": float(last_row["val_acc"]),
        "best_val_acc": float(best_val_acc_row["val_acc"]),
        "best_epoch": int(best_val_acc_row["epoch"]),
        "final_gap": float(last_row["gap"]),
    }
    return summary


# =========================
# Sidebar controls
# =========================
st.sidebar.header("Experiment Settings")
mode = st.sidebar.radio(
    "Choose experiment",
    ["Overtraining", "Regularized", "Compare Both"],
)

batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128, 256], index=2)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=15)
learning_rate = st.sidebar.selectbox("Learning rate", [1e-4, 3e-4, 1e-3], index=1)
weight_decay = st.sidebar.selectbox("Weight decay (regularized only)", [0.0, 1e-5, 1e-4, 1e-3], index=2)
patience = st.sidebar.slider("Early stopping patience", min_value=1, max_value=5, value=3)

run_button = st.sidebar.button("Run Experiment")

st.sidebar.markdown("---")
st.sidebar.write(f"Device: **{DEVICE}**")


# =========================
# Main explanation
# =========================
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("What this website does")
    st.write(
        "This app recreates your notebook as a website. It trains a pretrained ResNet18 on CIFAR-10 "
        "and lets you run the same two workflows: an overtraining setup and a regularized setup."
    )
with col_b:
    st.subheader("Experiment logic")
    st.write(
        "Overtraining uses a small subset of training data without regularization. "
        "Regularized training uses data augmentation, freezes the backbone, adds weight decay, "
        "and applies early stopping."
    )


if run_button:
    start = time.time()

    if mode == "Overtraining":
        st.header("Overtraining Experiment")
        config = ExperimentConfig(
            regularize=False,
            overfit_subset=True,
            freeze_backbone=False,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            weight_decay=0.0,
            patience=patience,
        )
        model, results, classes = run_experiment(config)
        summary = summarize_results(results)

        m1, m2, m3 = st.columns(3)
        m1.metric("Final Train Accuracy", f"{summary['final_train_acc']:.3f}")
        m2.metric("Final Validation Accuracy", f"{summary['final_val_acc']:.3f}")
        m3.metric("Train-Val Gap", f"{summary['final_gap']:.3f}")

        st.line_chart(results.set_index("epoch")[["train_acc", "val_acc"]])
        st.line_chart(results.set_index("epoch")[["train_loss", "val_loss"]])
        st.dataframe(results, use_container_width=True)

        if summary["final_gap"] > 0.10:
            st.warning("Train accuracy is much higher than validation accuracy. This suggests overtraining.")
        else:
            st.info("The train-validation gap is not very large yet.")

    elif mode == "Regularized":
        st.header("Regularized Experiment")
        config = ExperimentConfig(
            regularize=True,
            overfit_subset=False,
            freeze_backbone=True,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
        )
        model, results, classes = run_experiment(config)
        summary = summarize_results(results)

        m1, m2, m3 = st.columns(3)
        m1.metric("Best Validation Accuracy", f"{summary['best_val_acc']:.3f}")
        m2.metric("Best Epoch", summary["best_epoch"])
        m3.metric("Final Train-Val Gap", f"{summary['final_gap']:.3f}")

        st.line_chart(results.set_index("epoch")[["train_acc", "val_acc"]])
        st.line_chart(results.set_index("epoch")[["train_loss", "val_loss"]])
        st.dataframe(results, use_container_width=True)
        st.success("Regularization workflow completed.")

    else:
        st.header("Comparison: Overtraining vs Regularized")

        overfit_config = ExperimentConfig(
            regularize=False,
            overfit_subset=True,
            freeze_backbone=False,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            weight_decay=0.0,
            patience=patience,
        )
        regularized_config = ExperimentConfig(
            regularize=True,
            overfit_subset=False,
            freeze_backbone=True,
            batch_size=batch_size,
            epochs=epochs,
            lr=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
        )

        st.subheader("Running overtraining setup")
        _, overfit_results, _ = run_experiment(overfit_config)
        overfit_results = overfit_results.copy()
        overfit_results["experiment"] = "Overtraining"

        st.subheader("Running regularized setup")
        _, regularized_results, _ = run_experiment(regularized_config)
        regularized_results = regularized_results.copy()
        regularized_results["experiment"] = "Regularized"

        combined = pd.concat([overfit_results, regularized_results], ignore_index=True)

        st.subheader("Accuracy comparison")
        pivot_acc = combined.pivot(index="epoch", columns="experiment", values="val_acc")
        st.line_chart(pivot_acc)

        st.subheader("Loss comparison")
        pivot_loss = combined.pivot(index="epoch", columns="experiment", values="val_loss")
        st.line_chart(pivot_loss)

        st.subheader("Detailed results")
        st.dataframe(combined, use_container_width=True)

    elapsed = time.time() - start
    st.caption(f"Completed in {elapsed:.1f} seconds.")
else:
    st.info("Choose an experiment in the sidebar, then click 'Run Experiment'.")


st.markdown("---")
st.subheader("How to run locally")
st.code("pip install streamlit torch torchvision pandas\nstreamlit run cifar10_training_website.py", language="bash")

st.subheader("How this matches your notebook")
st.markdown(
    """
- Uses **CIFAR-10** as the dataset.
- Uses **pretrained ResNet18**.
- Includes an **overtraining experiment** with a small subset.
- Includes a **regularized experiment** with augmentation, frozen backbone, weight decay, and early stopping.
- Shows **epoch-by-epoch train/validation accuracy and loss** in a web interface.
    """
)
