import os
import numpy as np
import pandas as pd
from feature_extraction import extract_features

PROFILE_DIR = "data/profiles"

def fuzzy_match(image_path):
    input_features = extract_features(image_path)

    best_score = -1
    best_person = None

    for file in os.listdir(PROFILE_DIR):
        if not file.endswith(".csv"):
            continue

        profile = pd.read_csv(os.path.join(PROFILE_DIR, file))
        mean = profile["mean"].values
        std = profile["std"].values
        std[std == 0] = 0.0001

        score_vec = np.exp(-((input_features - mean) ** 2) / (2 * std ** 2))
        score = np.mean(score_vec)

        if score > best_score:
            best_score = score
            best_person = file.replace(".csv", "")

    return best_person, round(best_score * 100, 2)
