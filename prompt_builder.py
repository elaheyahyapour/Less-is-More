import json, math

ACTIONS = ["stop", "slow down", "proceed"]

def _present(x):
    return (x is not None) and not (isinstance(x, float) and math.isnan(x))

from typing import Dict, Optional

def _counts_to_str(obj_counts: Optional[Dict[str, int]], max_items: int = 6) -> str:
    if not obj_counts:
        return "none"
    items = sorted(obj_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_items]
    return ", ".join(f"{k}:{v}" for k, v in items)


def build_structured_prompt(
    caption: str,
    ttc: float,
    risk: float,
    complexity: Optional[str] = None,
    min_distance: Optional[float] = None,
    obj_counts: Optional[Dict[str, int]] = None,
    cues: Optional[Dict[str, bool]] = None
) -> str:

    cues = cues or {}
    counts_str = _counts_to_str(obj_counts)

    return (
        "You are an autonomous driving policy reasoner.\n"
        "Given the scene metrics and cues, select the safest immediate ACTION.\n\n"
        f"CAPTION: {caption}\n"
        f"TTC_SECONDS: {ttc:.2f}\n"
        f"RISK_SCORE: {risk:.2f}\n"
        f"SCENE_COMPLEXITY: {complexity or 'UNKNOWN'}\n"
        f"NEAREST_OBJECT_DISTANCE_M: {('%.1f' % min_distance) if min_distance is not None else 'UNKNOWN'}\n"
        f"OBJECT_COUNTS: {counts_str}\n"
        f"CUES: stop_sign={bool(cues.get('stop_sign'))}, traffic_light={bool(cues.get('traffic_light'))}\n\n"
        "Output EXACTLY this JSON (no extra text):\n"
        '{"action":"STOP|SLOW DOWN|PROCEED"}\n'
        "Decision guidance (apply the first rule that matches):\n"
        "- If stop_sign==True -> STOP, unless TTC>3.5 AND RISK<0.15.\n"
        "- If traffic_light==True and CAPTION suggests 'red' -> STOP; if 'yellow/amber' -> SLOW DOWN.\n"
        "- If TTC ≤ 2.0 OR RISK ≥ 0.40 -> STOP.\n"
        "- Else if TTC ≤ 3.0 OR RISK ≥ 0.28 -> SLOW DOWN.\n"
        "- Else -> PROCEED.\n"
        "Return ONLY the JSON."
    )

def build_unstructured_prompt_v2(caption: str) -> str:

    allowed = " | ".join(ACTIONS)
    return (
        "You are a driving safety classifier.\n\n"
        f"Scene Description: {caption}\n\n"
        "Respond in EXACTLY this format (two lines):\n"
        f"ACTION: <one of {allowed}>\n"
        "REASON: <one short sentence>"
    )

def build_structured_prompt_v2(
    caption: str,
    ttc: float,
    risk: float,
    complexity: Optional[str] = None,
    min_distance: Optional[float] = None,
    obj_counts: Optional[Dict[str, int]] = None,
    cues: Optional[Dict[str, bool]] = None
) -> str:

    cues = cues or {}
    counts_str = _counts_to_str(obj_counts)
    stop_sign        = bool(cues.get("stop_sign", False))
    stop_sign_close  = bool(cues.get("stop_sign_close", False))
    traffic_light    = bool(cues.get("traffic_light", False))
    red_light        = bool(cues.get("red_light", False))

    return (
        "You are an autonomous driving policy reasoner.\n"
        "Decide the safest immediate action and explain briefly, grounded in the metrics.\n\n"
        f"CAPTION: {caption}\n"
        f"TTC_SECONDS: {ttc:.2f}\n"
        f"RISK_SCORE: {risk:.2f}\n"
        f"SCENE_COMPLEXITY: {complexity or 'UNKNOWN'}\n"
        f"NEAREST_OBJECT_DISTANCE_M: {('%.1f' % min_distance) if min_distance is not None else 'UNKNOWN'}\n"
        f"OBJECT_COUNTS: {counts_str}\n"
        f"CUES: stop_sign={bool(cues.get('stop_sign'))}, "
        f"stop_sign_close={bool(cues.get('stop_sign_close'))}, "
        f"traffic_light={bool(cues.get('traffic_light'))}, "
        f"red_light={bool(cues.get('red_light'))}\n\n"
        "Return ONLY one single-line JSON object with these exact keys:\n"
        '{"action":"stop|slow down|proceed","reason":"<short one sentence>"}\n'
        "Do not include markdown, code fences, extra text, or additional keys.\n"
        "Decision order (apply the first rule that matches):\n"
        "- if red_light==true -> stop, unless ttc_seconds>3.5 and risk_score<0.15.\n"
        "- if stop_sign_close==true -> stop, unless ttc_seconds>3.5 and risk_score<0.15.\n"
        "- if stop_sign==true -> stop, unless ttc_seconds>3.5 and risk_score<0.15.\n"
        "- if ttc_seconds <= 2.0 or risk_score >= 0.40 -> stop.\n"
        "- if ttc_seconds <= 3.0 or risk_score >= 0.28 -> slow down.\n"
        "- else -> proceed."
    )


def build_structured_prompt_v3(
    caption, ttc, risk, complexity=None, min_distance=None, obj_counts=None, cues=None
) -> str:
    cues = cues or {}
    counts_str = ", ".join(
        f"{k}:{v}" for k, v in sorted((obj_counts or {}).items(), key=lambda kv:(-kv[1], kv[0]))[:3]
    ) or "none"

    stop_state = "close" if cues.get("stop_sign_close") else ("present" if cues.get("stop_sign") else "none")
    tl_state   = "red" if cues.get("red_light") else ("present" if cues.get("traffic_light") else "none")

    return (
        "You are an autonomous driving policy reasoner.\n"
        f"CAPTION: {caption}\n"
        f"METRICS: TTC={ttc:.2f}s, RISK={risk:.2f}, NEAREST={('%.1f' % min_distance) if min_distance is not None else 'UNK'}m, "
        f"COMPLEXITY={complexity or 'UNKNOWN'}, OBJS={counts_str}\n"
        f"SAFETY_CUES: stop_sign={stop_state}, traffic_light={tl_state}\n\n"
        'Return ONLY one JSON line: {"action":"stop|slow down|proceed","reason":"<short one sentence>"}\n'
        "Decision order:\n"
        "- red light -> stop (unless TTC>3.5 and RISK<0.15)\n"
        "- stop_sign=close -> stop (same exception)\n"
        "- stop_sign=present -> stop if TTC<=3.0 or RISK>=0.30\n"
        "- if TTC<=2.0 or RISK>=0.40 -> stop\n"
        "- else if TTC<=3.0 or RISK>=0.28 -> slow down\n"
        "- else -> proceed"
    )
