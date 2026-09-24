"""Keep-or-discard decision for raw IQ recordings (the "keep storage low"
part of the project plan, 13 Jul).

Raw IQ is large: a 10-minute pass at 1 Msps from an RTL-SDR is ~1.2 GB.
After detection, each recording gets one of three verdicts:

    detected       the model is confident a satellite signal is there
                   -> the raw IQ is kept where it was recorded
    uncertain      the model isn't sure (confidence inside the uncertain
                   band, default 0.3-0.7) -> the raw IQ is ALWAYS kept, in
                   <output_dir>/uncertain/, and the capture waits for a
                   person to review it (web UI Review page, src/review.py)
    not_detected   the model is confident there is nothing
                   -> handled by the policy below

The uncertain level exists so that a false negative never deletes real
data (Khoa's flow, Milestone 2 report). Policies for not_detected:

    keep-all           never touch the file
    archive-negatives  move it to <output_dir>/rejected/ (reversible)
    delete-negatives   delete it (the JSON sidecar, spectrogram, waterfall
                       image + matrix and the database row are kept)

The score comes from the waterfall model, then the IQ model, then the
rule detector - and the database row records which one was used, so a
decision can always be traced. The rule detector alone never leads to a
deletion: without an ML prediction a negative is "uncertain".
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

POLICIES = ("keep-all", "archive-negatives", "delete-negatives")
DEFAULT_POLICY = "keep-all"
#: None = use the model's own decision threshold (tuned when it was trained,
#: stored in its metadata). A number overrides it. Only used when the
#: uncertain band is switched off.
DEFAULT_KEEP_THRESHOLD = None
#: Confidence in [low, high) = uncertain. None = two-level (old behaviour).
DEFAULT_UNCERTAIN_BAND: Tuple[float, float] = (0.3, 0.7)

DETECTED, UNCERTAIN, NOT_DETECTED = "detected", "uncertain", "not_detected"
VERDICTS = (DETECTED, UNCERTAIN, NOT_DETECTED)
UNCERTAIN_DIR = "uncertain"
REJECTED_DIR = "rejected"


@dataclass
class RetentionDecision:
    keep: bool
    source: str            # "ml", "waterfall-ml", "rule" or "none"
    score: Optional[float]
    reason: str
    verdict: str = field(default="")

    def __post_init__(self):
        if not self.verdict:
            self.verdict = DETECTED if self.keep else NOT_DETECTED


@dataclass
class RetentionOutcome:
    action: str            # "kept", "kept for review", "archived", "deleted", "kept (policy keep-all)", "kept (error: ...)"
    final_path: Optional[str]
    decision: RetentionDecision


def parse_band(value) -> Optional[Tuple[float, float]]:
    """Config value -> (low, high) or None. Accepts [0.3, 0.7], "0.3,0.7", None/"off"."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "off", "none", "no"}):
        return None
    if isinstance(value, str):
        value = [v for v in value.replace(";", ",").split(",") if v.strip()]
    low, high = (float(v) for v in value)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"uncertain band must satisfy 0 <= low <= high <= 1, got {low}, {high}")
    return low, high


def decide(*, ml_detection=None, rule_detection=None, threshold=DEFAULT_KEEP_THRESHOLD,
           uncertain_band: Optional[Tuple[float, float]] = None) -> RetentionDecision:
    """Score the recording. ML confidence wins when the model made a prediction.

    With ``uncertain_band=(low, high)``: confidence >= high -> detected,
    < low -> not_detected, otherwise uncertain.

    Without a band (two levels): ``threshold=None`` follows the model's own
    yes/no decision (its tuned threshold); a number compares the confidence
    against that number instead."""
    if ml_detection is not None and getattr(ml_detection, "ml_confidence_score", None) is not None:
        score = float(ml_detection.ml_confidence_score)
        if uncertain_band is not None:
            low, high = uncertain_band
            if score >= high:
                return RetentionDecision(True, "ml", score, f"ML confidence {score:.2f} >= {high:.2f}: satellite", DETECTED)
            if score < low:
                return RetentionDecision(False, "ml", score, f"ML confidence {score:.2f} < {low:.2f}: no satellite",
                                         NOT_DETECTED)
            return RetentionDecision(True, "ml", score,
                                     f"ML confidence {score:.2f} between {low:.2f} and {high:.2f}: uncertain - "
                                     "kept for review", UNCERTAIN)
        if threshold is None:
            model_says = getattr(ml_detection, "ml_detection_result", None)
            keep = bool(model_says) if model_says is not None else score >= 0.5
            return RetentionDecision(keep, "ml", score,
                                     f"ML confidence {score:.2f}: model says {'satellite' if keep else 'no satellite'} "
                                     "(its tuned threshold)")
        keep = score >= threshold
        return RetentionDecision(keep, "ml", score,
                                 f"ML confidence {score:.2f} {'>=' if keep else '<'} threshold {threshold:.2f}")
    if rule_detection is not None:
        flagged = bool(rule_detection.detected)
        score = float(rule_detection.confidence_score)
        if uncertain_band is not None and not flagged:
            # The rule detector misses most real signals; never delete on its word alone.
            return RetentionDecision(True, "rule", score,
                                     f"no ML prediction and the rule detector found nothing (confidence {score:.2f}) - "
                                     "uncertain, kept for review", UNCERTAIN)
        return RetentionDecision(flagged, "rule", score,
                                 f"no ML prediction; rule detector {'flagged' if flagged else 'did not flag'} a candidate "
                                 f"(confidence {score:.2f})")
    return RetentionDecision(True, "none", None, "no detector result - kept by default",
                             UNCERTAIN if uncertain_band is not None else DETECTED)


def _move_with_sidecar(iq_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / iq_path.name
    shutil.move(str(iq_path), str(target))
    sidecar = iq_path.with_suffix(".json")
    if sidecar.exists():
        shutil.move(str(sidecar), str(target_dir / sidecar.name))
    return target


def apply(decision: RetentionDecision, iq_path, *, policy: str = DEFAULT_POLICY,
          archive_dir: Optional[Path] = None, uncertain_dir: Optional[Path] = None) -> RetentionOutcome:
    """Carry out the policy on the recording file. Never raises for I/O
    problems - a failure to move/delete leaves the file in place and says so."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown retention policy {policy!r}; choose one of {', '.join(POLICIES)}")
    iq_path = Path(iq_path)
    if policy == "keep-all":
        return RetentionOutcome("kept (policy keep-all)", str(iq_path), decision)
    try:
        if decision.verdict == UNCERTAIN:
            target = _move_with_sidecar(iq_path, Path(uncertain_dir) if uncertain_dir else iq_path.parent / UNCERTAIN_DIR)
            return RetentionOutcome("kept for review", str(target), decision)
        if decision.keep:
            return RetentionOutcome("kept", str(iq_path), decision)
        if policy == "archive-negatives":
            target = _move_with_sidecar(iq_path, Path(archive_dir) if archive_dir else iq_path.parent / REJECTED_DIR)
            return RetentionOutcome("archived", str(target), decision)
        iq_path.unlink()
        return RetentionOutcome("deleted", None, decision)
    except OSError as exc:
        return RetentionOutcome(f"kept (error: {exc})", str(iq_path), decision)
