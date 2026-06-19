"""Predictive router: label generation + a small classifier (build-spec §5/§16).

Strategy B picks the cheapest sufficient tier **upfront** from the query alone —
no cheap-then-strong double spend. This module has three pure-ish parts:

1. **Label generation (Appendix B).** :func:`label_cheapest_correct` is the pure,
   no-key-testable core: given which tiers were correct on an item, pick the
   cheapest correct one (or the cheapest tier if none was — escalating wouldn't
   have helped, §17). :func:`generate_labels` runs every tier (@api) to produce
   those per-item grades.
2. **The classifier.** :func:`train` fits a deterministic sklearn estimator
   (LogisticRegression default, KNeighborsClassifier fallback) over
   ``embedding → tier``. :class:`PredictiveRouter` bundles the fitted estimator
   with its provenance (embedder, prompt version, label-run ids) and turns a query
   into ``(tier, p_strong)``.
3. **Persistence.** :func:`save_router` / :func:`load_router` round-trip a
   ``PredictiveRouter`` via joblib (trained artifacts are gitignored, never
   committed).

Determinism: the *training* is deterministic (fixed solver + ``random_state``);
the *labels* depend on graded model output, so ``label_run_ids`` are recorded for
reproducibility (build-spec §9/§11).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from frugalroute.benchmarks import BenchItem, grade
from frugalroute.embed import DEFAULT_EMBEDDER, embed
from frugalroute.generate import generate
from frugalroute.prompts import PROMPT_VERSION


# ----------------------------------------------------------------------------
# Label generation (Appendix B) — the pure core is no-key testable.
# ----------------------------------------------------------------------------
def label_cheapest_correct(per_tier_correct: dict[str, bool], tiers: Sequence[str]) -> str:
    """Return the cheapest tier (earliest in ``tiers``) that was correct.

    If **no** tier was correct, return the cheapest tier (``tiers[0]``):
    escalating wouldn't have helped, so the cheapest is the right label
    (build-spec Appendix B / §17). The returned tier is always a member of
    ``tiers``. Raises ``ValueError`` on an empty tier list.
    """
    if not tiers:
        raise ValueError("tiers must be non-empty to assign a label.")
    for tier in tiers:
        if per_tier_correct.get(tier, False):
            return tier
    return tiers[0]


@dataclass
class LabelRun:
    """One labeled item with the per-tier grades that produced its label.

    ``run_id`` records the graded-run provenance (build-spec §9/§11): every item
    labeled in a single :func:`generate_labels` pass shares one id.
    """

    item_id: str
    label: str  # the chosen tier (model id)
    per_tier_correct: dict[str, bool]
    run_id: str


def _label_run_id(benchmark: str, tiers: Sequence[str], item_ids: Sequence[str]) -> str:
    """A deterministic id for one labeling pass (prompt + benchmark + tiers + items)."""
    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode())
    digest.update(b"|")
    digest.update(benchmark.encode())
    digest.update(b"|")
    digest.update("|".join(tiers).encode())
    digest.update(b"|")
    digest.update("|".join(item_ids).encode())
    return f"labels-{benchmark}-{digest.hexdigest()[:12]}"


def generate_labels(
    client: Any, items: Sequence[BenchItem], tiers: Sequence[str], benchmark: str
) -> list[LabelRun]:
    """Run every tier on every item, grade, and label the cheapest correct tier.

    (@api — exercised live, not in the no-key suite; the pure
    :func:`label_cheapest_correct` is the unit-tested core.) A refused tier counts
    as incorrect for labeling (it produced no usable answer). All items in this
    pass share one ``run_id`` for provenance.
    """
    tiers = list(tiers)
    run_id = _label_run_id(benchmark, tiers, [item.id for item in items])
    runs: list[LabelRun] = []
    for item in items:
        per_tier_correct: dict[str, bool] = {}
        for tier in tiers:
            result = generate(client, tier, item.question, benchmark)
            correct = (not result.refused) and grade(benchmark, result.text, item.gold)
            per_tier_correct[tier] = bool(correct)
        label = label_cheapest_correct(per_tier_correct, tiers)
        runs.append(
            LabelRun(
                item_id=item.id,
                label=label,
                per_tier_correct=per_tier_correct,
                run_id=run_id,
            )
        )
    return runs


# ----------------------------------------------------------------------------
# Classifier training + the predictive router
# ----------------------------------------------------------------------------
def train(
    embeddings: npt.NDArray[np.float32] | Sequence[Sequence[float]],
    labels: Sequence[str],
    tiers: Sequence[str],
    *,
    kind: str = "logreg",
    random_state: int = 0,
) -> Any:
    """Fit a deterministic sklearn classifier over ``embedding → tier label``.

    ``kind="logreg"`` (default) uses LogisticRegression; ``kind="knn"`` uses
    KNeighborsClassifier (build-spec §19-E fallback). Determinism comes from a
    fixed solver and ``random_state``. If the labels collapse to a single tier
    (no decision to learn), a constant classifier is fit instead — sklearn's
    estimators reject single-class training. Raises ``ValueError`` on an unknown
    ``kind`` or empty training data.
    """
    features = np.asarray(embeddings, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2-D array (n_items, dim).")
    if len(labels) != features.shape[0]:
        raise ValueError(
            f"labels length ({len(labels)}) must match embeddings rows ({features.shape[0]})."
        )

    label_list = list(labels)
    if len(set(label_list)) < 2:
        # Only one tier was ever the answer — nothing to discriminate. A constant
        # predictor keeps predict_proba well-defined without a degenerate fit.
        from sklearn.dummy import DummyClassifier

        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(features, label_list)
        return clf

    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(random_state=random_state, max_iter=1000)
    elif kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        estimator = KNeighborsClassifier(n_neighbors=min(5, len(label_list)))
    else:
        raise ValueError(f"Unknown classifier kind {kind!r}; expected 'logreg' or 'knn'.")

    estimator.fit(features, label_list)
    return estimator


@dataclass
class PredictiveRouter:
    """A fitted predictive router with its reproducibility provenance.

    ``clf`` is a fitted sklearn estimator (typed ``Any`` because sklearn ships no
    stubs). ``tiers`` is the ordered cheap→strong list it was trained over.
    ``embedder_name``/``prompt_version``/``label_run_ids`` pin exactly how the
    training labels and features were produced (build-spec §9/§11).
    """

    clf: Any
    tiers: list[str]
    embedder_name: str
    prompt_version: str
    label_run_ids: list[str]

    @property
    def clf_kind(self) -> str:
        """The fitted estimator's class name (recorded via the saved ``clf``)."""
        return type(self.clf).__name__

    def predict_tier_from_embedding(
        self, vector: npt.NDArray[np.float32] | Sequence[float], theta: float | None = None
    ) -> tuple[str, float]:
        """Map one embedding to ``(tier, p_strong)``; raise on degenerate input.

        ``p_strong`` is P(needs the strong tier) — the probability mass on
        ``tiers[-1]`` (the split-01 ``RouteResult.p_strong`` field / UI decision
        margin). For the 2-tier case, route to strong iff ``p_strong > theta``
        (default ``theta=0.5``); θ sweeps the predictive frontier. For >2 tiers,
        route to the argmax tier. Raises ``ValueError`` on a non-finite embedding
        or a predicted tier outside ``tiers`` (no silent mis-route — split-04 R11).
        """
        arr = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("Degenerate embedding (contains NaN/inf); cannot predict a tier.")

        proba = np.asarray(self.clf.predict_proba(arr), dtype=np.float64)[0]
        classes = [str(c) for c in self.clf.classes_]
        strong = self.tiers[-1]
        p_strong = float(proba[classes.index(strong)]) if strong in classes else 0.0

        if len(self.tiers) == 2:
            threshold = 0.5 if theta is None else theta
            tier = strong if p_strong > threshold else self.tiers[0]
        else:
            tier = classes[int(np.argmax(proba))]

        if tier not in self.tiers:
            raise ValueError(
                f"Predicted tier {tier!r} is not in the router's tiers {self.tiers!r}."
            )
        return tier, p_strong

    def predict_tier(
        self, query: str, theta: float | None = None, *, embedder: Any = None
    ) -> tuple[str, float]:
        """Embed ``query`` (local model) and predict ``(tier, p_strong)``.

        ``embedder`` is injectable for no-key tests (a fake with ``.encode``);
        it defaults to the lazily-loaded local model.
        """
        vector = embed([query], embedder=embedder)[0]
        return self.predict_tier_from_embedding(vector, theta)


def save_router(router: PredictiveRouter, path: str | Path) -> None:
    """Persist a ``PredictiveRouter`` to ``path`` via joblib (creates parents)."""
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(router, target)


def load_router(path: str | Path) -> PredictiveRouter:
    """Load a ``PredictiveRouter`` saved by :func:`save_router`.

    Raises ``FileNotFoundError`` if the path is missing and ``TypeError`` if the
    file does not hold a ``PredictiveRouter`` (guarding against arbitrary pickles).
    """
    import joblib

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Router artifact not found: {source}")
    router = joblib.load(source)
    if not isinstance(router, PredictiveRouter):
        raise TypeError(f"{source} does not contain a PredictiveRouter (got {type(router)!r}).")
    return router


# Build-spec §13 library surface: ``from frugalroute import Router``.
Router = PredictiveRouter

__all__ = [
    "DEFAULT_EMBEDDER",
    "LabelRun",
    "PredictiveRouter",
    "Router",
    "generate_labels",
    "label_cheapest_correct",
    "load_router",
    "save_router",
    "train",
]
