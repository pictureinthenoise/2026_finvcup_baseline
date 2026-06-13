import argparse
import yaml
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.data import TurnTakingTrainDataset, build_collate_fn
from src.models import MultimodalTurnTakingModel

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
    
    conv_ids = list_conv_ids(LABELS_DIR)
    split_ids = split_conversation_ids(
        conv_ids=conv_ids,
        valid_ratio=float(VALID_RATIO),
        seed=int(SEED)
    )
    _, valid_ids = split_ids["train"], split_ids["valid"]

    valid_samples = build_train_samples_multitask(TRAIN_LABELS_DIR, conv_ids, CONTEXT_CHUNKS, TARGET_CHUNKS, STRIDE, LABELS, MULTI_TARGETS, None)
    valid_dataset = TurnTakingTrainDataset(
        samples=valid_samples,
        train_audio_dir=TRAIN_AUDIO_DIR,
        train_text_dir=TRAIN_TEXT_DIR,
        train_labels_dir=TRAIN_LABELS_DIR,
        context_chunks=CONTEXT_CHUNKS,
        target_chunks=TARGET_CHUNKS,
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
    )
    print("Done creating validation dataset.")

if __name__ == "__main__":
    main()
