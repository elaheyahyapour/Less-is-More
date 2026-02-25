import re, json
from typing import Optional, Dict, Any

STRICT_ALLOWED = {"stop": "stop", "slow down": "slow down", "proceed": "proceed"}



def _extract_single_json_object(text: Optional[str]) -> Optional[str]:

    if text is None:
        return None
    s = text.strip()

    fence = re.findall(r"```(?:json)?\s*({.*?})\s*```", s, flags=re.S | re.I)
    if len(fence) == 1:
        return fence[0].strip()
    if len(fence) > 1:
        return None


    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    candidate = m.group(0).strip()
    if '"action"' not in candidate and "'action'" not in candidate:
        return None
    head, tail = s.split(candidate, 1)
    if head.strip() or tail.strip():
        return None
    return candidate


def parse_action_json_strict(text: Optional[str]) -> Dict[str, Any]:

    try:
        blob = _extract_single_json_object(text)
        if blob is None:
            return {"predicted": None, "valid": False, "error": "no-json"}

        obj = json.loads(blob)
        if not isinstance(obj, dict) or "action" not in obj:
            return {"predicted": None, "valid": False, "error": "bad-schema"}

        act = str(obj["action"]).strip().lower()
        if act not in STRICT_ALLOWED:
            return {"predicted": None, "valid": False, "error": f"bad-action:{act}"}

        return {"predicted": STRICT_ALLOWED[act], "valid": True, "error": None}
    except Exception as e:
        return {"predicted": None, "valid": False, "error": f"parse-exc:{e.__class__.__name__}"}


def evaluate_action_json_strict(response_text: Optional[str], y_true: str) -> Dict[str, Any]:

    out = parse_action_json_strict(response_text)
    pred = out["predicted"]
    correct = (pred == y_true) if out["valid"] else False
    return {
        "predicted": pred,
        "correct": correct,
        "valid": out["valid"],
        "confidence": None,
        "error": out["error"],
    }


_STOP_PAT = re.compile(
    r"\bstop(ping)?\b|full\s*stop|hard\s*brake|brake\b|halt\b|emergency\s*stop",
    flags=re.I,
)
_SLOW_PAT = re.compile(
    r"slow\s*down|decelerate|yield\b|reduce\s*speed|caution\b|proceed\s*with\s*caution|ease\s*off",
    flags=re.I,
)
_PROCEED_PAT = re.compile(
    r"\bproceed\b|\bgo\b|keep\s*going|continue|drive\s*on|move\s*ahead",
    flags=re.I,
)

def _normalize_action(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().lower()
    if s in STRICT_ALLOWED:
        return STRICT_ALLOWED[s]
    if _STOP_PAT.search(s):
        return "stop"
    if _SLOW_PAT.search(s):
        return "slow down"
    if _PROCEED_PAT.search(s):
        return "proceed"
    return None

def _extract_reason_from_json(obj: Any) -> Optional[str]:
    if isinstance(obj, dict) and "reason" in obj and isinstance(obj["reason"], str):
        return obj["reason"].strip()
    return None

def parse_action_lenient(text: Optional[str]) -> Dict[str, Any]:

    if not text:
        return {"predicted": None, "reason": None, "valid": False, "error": "empty"}

    s = text.strip()
    reason = None

    json_blob = None
    for m in re.finditer(r"\{.*?\}", s, flags=re.S):
        candidate = m.group(0)
        if re.search(r'"action"\s*:', candidate, flags=re.I):
            json_blob = candidate
            break
    if json_blob:
        try:
            obj = json.loads(json_blob)
            if isinstance(obj, dict) and "action" in obj:
                pred = _normalize_action(str(obj["action"]))
                reason = _extract_reason_from_json(obj)
                if pred:
                    return {"predicted": pred, "reason": reason, "valid": True, "error": None}
                return {"predicted": None, "reason": reason, "valid": False, "error": "bad-action"}
        except Exception:
            pass

    m = re.search(r"\bACTION\s*[:\-]\s*([A-Za-z ]+)", s, flags=re.I)
    if m:
        pred = _normalize_action(m.group(1))
        m2 = re.search(r"\bREASON\s*[:\-]\s*(.+)", s, flags=re.I)
        if m2:
            reason = m2.group(1).strip()
        if pred:
            return {"predicted": pred, "reason": reason, "valid": True, "error": None}
        return {"predicted": None, "reason": reason, "valid": False, "error": "bad-action"}

    pred = _normalize_action(s)
    if pred:
        return {"predicted": pred, "reason": None, "valid": True, "error": None}

    return {"predicted": None, "reason": None, "valid": False, "error": "unparsed"}


def evaluate_action(
    response_text: Optional[str],
    y_true: str,
    mode: str = "auto",
    conservative_default: str = "slow down",
) -> Dict[str, Any]:

    if mode not in {"strict", "lenient", "auto"}:
        mode = "auto"

    if mode in {"strict", "auto"}:
        strict_out = parse_action_json_strict(response_text)
        if strict_out["valid"]:
            pred = strict_out["predicted"]
            return {
                "predicted": pred,
                "correct": (pred == y_true),
                "valid": True,
                "confidence": None,
                "error": None,
                "reason": None,
            }
        if mode == "strict":
            return {
                "predicted": None,
                "correct": False,
                "valid": False,
                "confidence": None,
                "error": strict_out["error"],
                "reason": None,
            }

    lenient_out = parse_action_lenient(response_text)
    pred = lenient_out["predicted"]
    reason = lenient_out.get("reason")
    valid = lenient_out["valid"]

    if pred is None:
        pred = STRICT_ALLOWED.get(conservative_default.lower(), "slow down")
        return {
            "predicted": pred,
            "correct": (pred == y_true),
            "valid": False,
            "confidence": None,
            "error": lenient_out["error"],
            "reason": reason,
        }

    return {
        "predicted": pred,
        "correct": (pred == y_true),
        "valid": valid,
        "confidence": None,
        "error": lenient_out["error"],
        "reason": reason,
    }


__all__ = [
    "parse_action_json_strict",
    "evaluate_action_json_strict",
    "parse_action_lenient",
    "evaluate_action",
]
