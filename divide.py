import os
import shutil
import random

SOURCE_DIR = "quantum_states"
TARGET_DIR = "bloch_dataset"
CATEGORIES = sorted(["entangled", "mixed", "pure"])
TRAIN_RATIO = 0.8

for split in ["train", "test"]:
    for cat in CATEGORIES:
        os.makedirs(os.path.join(TARGET_DIR, split, cat), exist_ok=True)

for cat in CATEGORIES:
    src_path = os.path.join(SOURCE_DIR, cat)
    all_files = [f for f in os.listdir(src_path) if f.endswith(".png")]
    random.shuffle(all_files)

    split_idx = int(len(all_files) * TRAIN_RATIO)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    for f in train_files:
        shutil.copy(
            os.path.join(src_path, f),
            os.path.join(TARGET_DIR, "train", cat, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(src_path, f),
            os.path.join(TARGET_DIR, "test", cat, f)
        )

print("Dataset split complete")
