import os
import numpy as np
import pandas as pd
from feature_extraction import extract_features

RAW_DIR = "data/raw_images"
PROFILE_DIR = "data/profiles"

def train_profiles():
    os.makedirs(PROFILE_DIR, exist_ok=True)

    for person in os.listdir(RAW_DIR):
        person_dir = os.path.join(RAW_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        features = []
        for img in os.listdir(person_dir):
            if img.lower().endswith(("jpg","png","jpeg")):
                try:
                    feats = extract_features(os.path.join(person_dir, img))
                    features.append(feats)
                except:
                    pass

        if not features:
            continue

        arr = np.array(features)
        profile = pd.DataFrame({
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0)
        })

        profile.to_csv(os.path.join(PROFILE_DIR, f"{person}.csv"), index=False)
