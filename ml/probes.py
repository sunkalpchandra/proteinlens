"""Supervised probes on frozen representations.

A probe is a logistic-regression classifier trained on a fixed representation
matrix; probe quality measures how linearly accessible a property is in that
representation. The language model is never updated.

Task design note: our train/val/test splits group proteins by family, so family
membership itself cannot be a cross-split probe target (test families are
unseen classes by construction). Probes therefore target properties that cut
across families — enzyme/non-enzyme, EC top class, subcellular localization —
which is precisely the "does the geometry generalize beyond homology"
question worth asking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeTask:
    name: str
    labels: pd.Series          # aligned with corpus rows; NaN = excluded
    min_class_count: int = 40  # classes below this (in train) are dropped

    def prepare(self, splits: pd.Series) -> tuple[pd.Series, list[str]]:
        """Return usable labels (NaN elsewhere) and the retained class list."""
        train_labels = self.labels[splits == "train"].dropna()
        counts = train_labels.value_counts()
        classes = counts[counts >= self.min_class_count].index.tolist()
        usable = self.labels.where(self.labels.isin(classes))
        return usable, classes


def build_tasks(df: pd.DataFrame) -> list[ProbeTask]:
    enzyme = df["is_enzyme"].map({True: "enzyme", False: "non-enzyme"})
    ec = df["ec_class"].where(df["is_enzyme"])
    localization = df["localization"]
    return [
        ProbeTask("enzyme_vs_nonenzyme", enzyme, min_class_count=100),
        ProbeTask("ec_class", ec, min_class_count=40),
        ProbeTask("subcellular_localization", localization, min_class_count=80),
    ]


def run_probe(
    representation: np.ndarray,
    task: ProbeTask,
    splits: pd.Series,
    seed: int = 42,
    max_iter: int = 3000,
) -> dict:
    """Train on train split, model-select nothing (fixed C), report test metrics."""
    usable, classes = task.prepare(splits)
    mask = usable.notna().to_numpy()
    train_mask = mask & (splits == "train").to_numpy()
    test_mask = mask & (splits == "test").to_numpy()

    scaler = StandardScaler().fit(representation[train_mask])
    x_train = scaler.transform(representation[train_mask])
    x_test = scaler.transform(representation[test_mask])
    y_train = usable[train_mask].to_numpy()
    y_test = usable[test_mask].to_numpy()

    clf = LogisticRegression(max_iter=max_iter, C=1.0, random_state=seed)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)

    metrics = {
        "task": task.name,
        "n_classes": len(classes),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    try:
        if len(classes) == 2:
            positive = list(clf.classes_).index(classes[0])
            metrics["auroc"] = float(
                roc_auc_score((y_test == classes[0]).astype(int), proba[:, positive])
            )
        else:
            metrics["auroc_macro_ovr"] = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro",
                              labels=list(clf.classes_))
            )
    except ValueError:
        pass  # a class can be absent from the test split; AUROC is then undefined
    return metrics
