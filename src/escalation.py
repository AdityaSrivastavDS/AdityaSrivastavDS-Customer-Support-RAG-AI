from dataclasses import dataclass

@dataclass
class EscalationResult:
    risk_score: float
    should_escalate: bool
    reason: str

def escalation_risk(sentiment_label: str, emotion_label: str, neg_streak: int, threshold: float = 0.6, neg_streak_escalate: int = 2) -> EscalationResult:
    # Transparent, simple scoring:
    # Base from sentiment
    base = 0.0
    if sentiment_label == "negative":
        base += 0.5
    elif sentiment_label == "neutral":
        base += 0.2
    else:
        base += 0.05

    # Emotion bumps
    anger_like = {"anger", "annoyance", "disgust", "fear", "sadness"}
    if emotion_label in anger_like:
        base += 0.3

    # History bump
    if neg_streak >= 1:
        base += 0.15 * neg_streak

    # Clamp to [0, 1]
    risk = min(1.0, base)
    should_escalate = risk >= threshold or neg_streak >= neg_streak_escalate
    reason = "negative streak" if neg_streak >= neg_streak_escalate else ("high risk score" if should_escalate else "low risk")
    return EscalationResult(risk_score=risk, should_escalate=should_escalate, reason=reason)
