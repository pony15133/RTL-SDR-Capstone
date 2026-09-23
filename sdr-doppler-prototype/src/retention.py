"""Keep-or-discard decision for raw IQ recordings (the "keep storage low"
part of the project plan, 13 Jul).

Raw IQ is large: 60 s at 2.4 Msps from an RTL-SDR is ~290 MB. After
detection, each recording is scored and handled according to a policy:

    keep-all           never touch the file (default - safe while the
                       model is still being validated)
    archive-negatives  move recordings judged "no satellite" into an
                       archive folder (reversible; purge it when you're
                       confident)
    delete-negatives   delete recordings judged "no satellite" (and their
                       JSON sidecar is kept, so the attempt stays documented)

The score comes from the ML model when it produced a prediction,
otherwise from the rule-based detector - and the database row records
which one was used, so a decision can always be traced.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

POLICIES = ("keep-all", "archive-negatives", "delete-negatives")
DEFAULT_POLICY = "keep-all"
DEFAULT_KEEP_THRESHOLD = 0.5


@dataclass
class RetentionDecision:
    keep: bool
    source: str            # "ml" or "rule"
    score: Optional[float]
    reason: str


@dataclass
class RetentionOutcome:
    action: str            # "kept", "archived", "deleted", "kept (policy keep-all)", "kept (error: ...)"
    final_path: Optional[str]
    decision: RetentionDecision


def decide(*, ml_detection=None, rule_detection=None, threshold: float = DEFAULT_KEEP_THRESHOLD) -> RetentionDecision:
    """Score the recording. ML confidence wins when the model made a prediction."""
    if ml_detection is not None and getattr(ml_detection, "ml_confidence_score", None) is not None:
        score = float(ml_detection.ml_confidence_score)
        keep = score >= threshold
        return RetentionDecision(keep, "ml", score,
                                 f"ML confidence {score:.2f} {'>=' if keep else '<'} threshold {threshold:.2f}")
    if rule_detection is not None:
        keep = bool(rule_detection.detected)
        score = float(rule_detection.confidence_score)
        return RetentionDecision(keep, "rule", score,
                                 f"no ML prediction; rule detector {'flagged' if keep else 'did not flag'} a candidate "
                                 f"(confidence {score:.2f})")
    return RetentionDecision(True, "none", None, "no detector result - kept by default")


def apply(decision: RetentionDecision, iq_path, *, policy: str = DEFAULT_POLICY,
          archive_dir: Optional[Path] = None) -> RetentionOutcome:
    """Carry out the policy on the recording file. Never raises for I/O
    problems - a failure to move/delete leaves the file in place and says so."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown retention policy {policy!r}; choose one of {', '.join(POLICIES)}")
    iq_path = Path(iq_path)
    if policy == "keep-all":
        return RetentionOutcome("kept (policy keep-all)", str(iq_path), decision)
    if decision.keep:
        return RetentionOutcome("kept", str(iq_path), decision)
    try:
        if policy == "archive-negatives":
            target_dir = Path(archive_dir) if archive_dir else iq_path.parent / "rejected"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / iq_path.name
            shutil.move(str(iq_path), str(target))
            sidecar = iq_path.with_suffix(".json")
            if sidecar.exists():
                shutil.move(str(sidecar), str(target_dir / sidecar.name))
            return RetentionOutcome("archived", str(target), decision)
        iq_path.unlink()
        return RetentionOutcome("deleted", None, decision)
    except OSError as exc:
        return RetentionOutcome(f"kept (error: {exc})", str(iq_path), decision)
