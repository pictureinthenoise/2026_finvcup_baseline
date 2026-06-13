import json
import argparse
import yaml
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import TurnTakingTrainDataset, build_collate_fn
from src.models import MultimodalTurnTakingModel

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

@dataclass
class TrainSampleMulti:
    conv_id: str
    end_idx: int
    # (BC, I, T) or any configured order
    label_vec: Tuple[int, ...]

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--max_eval_batches", type=int, required=True)
    return p.parse_args()

def list_conv_ids(labels_dir: Path) -> List[str]:
    return sorted([p.stem for p in labels_dir.glob("*.npy")])

def split_conversation_ids(
    conv_ids: Sequence[str], valid_ratio: float, seed: int
) -> Dict[str, List[str]]:
    conv_ids = list(conv_ids)
    random.Random(seed).shuffle(conv_ids)
    valid_size = max(1, int(len(conv_ids) * valid_ratio))
    valid_ids = sorted(conv_ids[:valid_size])
    train_ids = sorted(conv_ids[valid_size:])
    return {"train": train_ids, "valid": valid_ids}

def build_train_samples_multitask(
    labels_dir: Path,
    conv_ids: Sequence[str],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    label_ids: Dict[str, int],
    target_labels: Sequence[str] = ("BC", "I", "T"),
    max_samples: Optional[int] = None,
) -> List[TrainSampleMulti]:
    samples: List[TrainSampleMulti] = []
    target_id_list = [int(label_ids[k]) for k in target_labels]
    for conv_id in conv_ids:
        labels = np.load(labels_dir / f"{conv_id}.npy")
        max_end = labels.shape[0] - target_chunks
        for end_idx in range(context_chunks, max_end + 1, stride):
            future = labels[end_idx : end_idx + target_chunks]
            y_vec = tuple(int(any(int(x) == tid for x in future)) for tid in target_id_list)
            samples.append(TrainSampleMulti(conv_id=conv_id, end_idx=end_idx, label_vec=y_vec))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples

def compute_multilabel_metrics(labels, probs, label_names=None) -> Dict[str, float]:
    """
    Label order is `[C, NA, I, BC, T]`. So, interruption labels (`I`) have index `2` and back-channel labels 
    (`BC`) have index `3`.
    """
    THRESH = [0.5, 0.5, 0.5, 0.5, 0.5] # Class thresholds
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if labels.ndim != 2 or probs.ndim != 2:
        raise ValueError(f"Expected 2D labels/probs, got {labels.shape} and {probs.shape}")
    if labels.shape != probs.shape:
        raise ValueError(f"Shape mismatch: labels {labels.shape} vs probs {probs.shape}")

    n_labels = labels.shape[1]
    if label_names is None:
        label_names = [f"label{i}" for i in range(n_labels)]
    if len(label_names) != n_labels:
        raise ValueError(f"label_names length {len(label_names)} != n_labels {n_labels}")

    out: Dict[str, float] = {}
    per_acc, per_f1, per_auc = [], [], []
    
    for i, name in enumerate(label_names):
        y = labels[:, i]
        p = probs[:, i]
        pred = (p >= THRESH[i]).astype(int)
        # pred = (p >= 0.5).astype(int)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, zero_division=0))
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, p))
        else:
            auc = 0.5
        out[f"{name}_accuracy"] = acc
        out[f"{name}_f1"] = f1
        out[f"{name}_roc_auc"] = auc
        per_acc.append(acc)
        per_f1.append(f1)
        per_auc.append(auc)

    out["macro_accuracy"] = float(np.mean(per_acc))
    out["macro_f1"] = float(np.mean(per_f1))
    out["macro_roc_auc"] = float(np.mean(per_auc))
    # Alias for backward-compatible save_metric/print flow
    out["accuracy"] = out["macro_accuracy"]
    out["f1"] = out["macro_f1"]
    out["roc_auc"] = out["macro_roc_auc"]
    return out

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    use_amp: bool,
    label_names: list[str] | None = None,
    max_batches: int | None = None,
):
    model.eval()
    all_labels, all_probs = [], []
    for bi, batch in enumerate(tqdm(data_loader, desc="eval", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        waveform = batch["waveform"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        context_labels = batch["context_labels"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(
                waveform=waveform,
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_labels=context_labels,
            )

        probs = torch.sigmoid(logits)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    labels_np = (
        torch.as_tensor(all_labels).numpy() if len(all_labels) > 0 else np.array([])
    )
    if labels_np.ndim == 2:
        return compute_multilabel_metrics(all_labels, all_probs, label_names=label_names)
    raise RuntimeError("The baseline has been solidified into a multi-label training setup; the evaluation process should not enter the binary classification branch.")

def main():
    # --- MANUAL CONFIG ---
    # LABELS_DIR = Path("C:/Users/LenovoPC/Documents/GitHub/aiml-finvolution-2026-teach-ai-when-to-speak/input/train/labels")
    # VALID_RATIO = 0.1
    # SEED = 42
    # CONTEXT_CHUNKS = 375
    # TARGET_CHUNKS = 25
    # STRIDE = 5
    # LABELS = {
    #   "C": 0,
    #   "T": 1,
    #   "BC": 2,
    #   "I": 3,
    #   "NA": 4,
    #   "positive_ids": [1, 2, 3],
    #   "multi_targets": ["C", "NA", "I", "BC", "T"]
    # }
    # MULTI_TARGETS = list(LABELS.get("multi_targets", ["C", "NA", "I", "BC", "T"]))

    args = parse_args()
    cfg = load_config(args.config)

    TRAIN_AUDIO_DIR = Path(cfg["paths"]["train_audio_dir"])
    TRAIN_TEXT_DIR = Path(cfg["paths"]["train_text_dir"])
    TRAIN_LABELS_DIR = Path(cfg["paths"]["train_labels_dir"])
    VALID_RATIO = float(cfg["split"]["valid_ratio"])
    SEED = int(cfg["seed"])
    CONTEXT_CHUNKS = int(cfg["context_chunks"])
    TARGET_CHUNKS = int(cfg["target_chunks"])
    STRIDE = int(cfg["stride"])
    LABELS = cfg["labels"]
    MULTI_TARGETS = list(cfg.get("labels", {}).get("multi_targets", ["C", "NA", "I", "BC", "T"]))
    METRIC_LABEL_NAMES = [x.lower() for x in MULTI_TARGETS]
    
    conv_ids = list_conv_ids(TRAIN_LABELS_DIR)
    split_ids = split_conversation_ids(
        conv_ids=conv_ids,
        valid_ratio=float(VALID_RATIO),
        seed=int(SEED)
    )
    _, valid_ids = split_ids["train"], split_ids["valid"]

    valid_samples = build_train_samples_multitask(TRAIN_LABELS_DIR, conv_ids, CONTEXT_CHUNKS, TARGET_CHUNKS, STRIDE, LABELS, MULTI_TARGETS, None)
    ds = TurnTakingTrainDataset(
        samples=valid_samples,
        train_audio_dir=TRAIN_AUDIO_DIR,
        train_text_dir=TRAIN_TEXT_DIR,
        train_labels_dir=TRAIN_LABELS_DIR,
        context_chunks=CONTEXT_CHUNKS,
        target_chunks=TARGET_CHUNKS,
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    bs = int(cfg["train"]["eval_batch_size"])
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = MultimodalTurnTakingModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    use_amp = bool(cfg["train"].get("use_amp", False))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["segment_id"] + MULTI_TARGETS

    limit = 500 # Set arbirarily high to process all segments
    done = 0
    rows: list[dict] = []

    metrics = evaluate(model, loader, device, use_amp, METRIC_LABEL_NAMES, args.max_eval_batches)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write(json.dumps(metrics))
if __name__ == "__main__":
    main()
