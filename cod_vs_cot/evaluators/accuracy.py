"""
accuracy.py  –  universal scorer for CoD-vs-CoT results.

Handles
  • Multiple-choice   (A/B/C/D  or  0-3  or  letter strings)
  • Yes/No, True/False, 0/1
  • Free-form answers (fallback)

JSON schema expected per record
  {
    "answers": [pred1, pred2, …],   # strings
    "gold":    ... ,                # string | int | bool
    "latency": 1.23                 # seconds (float)
  }
"""

from collections import Counter
import json, pathlib, re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lower-case & squeeze whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())

def _majority(preds) -> str:
    counts = Counter(_norm(p) for p in preds)
    return counts.most_common(1)[0][0]

# ---------------------------------------------------------------------------
# multiple-choice (A/B/C/D  ↔  0-3)
# ---------------------------------------------------------------------------

CHOICE2IDX = {"a": 0, "b": 1, "c": 2, "d": 3}
IDX2CHOICE = {v: k for k, v in CHOICE2IDX.items()}

MC_LETTER_RE = re.compile(r"\b([abcd])\b", flags=re.I)
MC_DIGIT_RE  = re.compile(r"\b([0-3])\b")

def _mc_to_int(text: str) -> Optional[int]:
    """
    Extract *last* choice reference from text and return it as 0-3.
    Accepts letters A-D or digits 0-3.  Returns None if nothing found.
    """
    t = _norm(text)

    letters = MC_LETTER_RE.findall(t)
    if letters:
        return CHOICE2IDX[letters[-1]]

    digits = MC_DIGIT_RE.findall(t)
    if digits:
        return int(digits[-1])

    return None

# ---------------------------------------------------------------------------
# yes / no parsing
# ---------------------------------------------------------------------------

_YES = {"yes", "true", "1"}
_NO  = {"no",  "false", "0"}

YN_RE = re.compile(r"\b(yes|no|true|false|[01])\b", flags=re.I)

def _yesno_to_int(obj) -> Optional[int]:
    """
    Map various yes/no representations to 1/0.  Accepts:
      • bool   → 1 / 0
      • int    → 1 / 0  (only if value in {0,1})
      • str    → 'yes'/'no'/'true'/'false'/'1'/'0'
    Returns None when unrecognised.
    """
    if isinstance(obj, bool):
        return int(obj)

    if isinstance(obj, (int, float)) and obj in (0, 1):
        return int(obj)

    if isinstance(obj, str):
        t = _norm(obj)
        if t in _YES:
            return 1
        if t in _NO:
            return 0

    return None

def _parse_yesno_pred(text: str) -> Optional[int]:
    """Return last yes/no mention in text as 1/0, else None."""
    tokens = YN_RE.findall(_norm(text))
    if not tokens:
        return None
    return 1 if tokens[-1] in _YES else 0

# ---------------------------------------------------------------------------
# main scoring routine
# ---------------------------------------------------------------------------

def _canonical_pair(pred_raw: str, gold_raw) -> Tuple[Optional[int], Optional[int]]:
    """
    Attempt MC first, then Yes/No; return (pred_int, gold_int) in same domain,
    else (None, None).  Domains are disjoint, so safe.
    """
    # ----- multiple choice
    gold_mc = _mc_to_int(str(gold_raw))
    if gold_mc is not None:
        return _mc_to_int(pred_raw), gold_mc

    # ----- yes / no
    gold_yn = _yesno_to_int(gold_raw)
    if gold_yn is not None:
        return _parse_yesno_pred(pred_raw), gold_yn

    return None, None

def score_file(path: str):
    """
    Returns dict with accuracy, avg latency, n, and task-file stem.
    """
    records = json.loads(pathlib.Path(path).read_text())

    total = correct = 0
    latency_sum = 0.0

    for rec in records:
        pred_raw = (_majority(rec["answers"])
                    if len(rec["answers"]) > 1
                    else rec["answers"][0])
        gold_raw = rec["gold"]

        pred_int, gold_int = _canonical_pair(pred_raw, gold_raw)

        if pred_int is not None and gold_int is not None:
            hit = pred_int == gold_int
        else:
            # fuzzy string fallback
            hit = _norm(pred_raw).startswith(_norm(str(gold_raw)))

        correct     += int(hit)
        total       += 1
        latency_sum += rec["latency"]

    return {
        "accuracy":     correct / total if total else 0.0,
        "latency_avg":  latency_sum / total if total else 0.0,
        "n":            total,
        "task":         pathlib.Path(path).stem,
    }

# ---------------------------------------------------------------------------
# CLI helper (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, pandas as pd
    rows = [score_file(fp) for fp in sys.argv[1:]]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
