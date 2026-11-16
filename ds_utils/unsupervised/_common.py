"""Helpers shared by the unsupervised learning utilities."""

import numpy as np
import pandas as pd


def _extract_cardinality(labels):
    labels_df = pd.DataFrame(np.transpose(labels), columns=["labels"])
    cardinality = labels_df["labels"].value_counts().sort_index().reset_index()
    return cardinality


def _extract_magnitude(X, labels, cluster_centers, distance_function):
    data = pd.DataFrame({"point": list(X), "label": labels})
    data["center"] = data["label"].apply(lambda label: cluster_centers[label])
    data["distance"] = data.apply(lambda row: distance_function(row["point"], row["center"]), axis=1)
    magnitude = data.groupby("label")["distance"].sum()
    return magnitude
