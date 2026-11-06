# octscreen/app/triage.py
"""Class + confidence → triage level. All policy comes from configs/app.yaml → policy."""
LEVELS = ("urgent", "routine", "none", "review")


def validate_policy(policy, class_names):
    """Fail fast at startup if the config doesn't cover the model's classes."""
    triage = dict(policy.triage)
    missing = [c for c in class_names if c not in triage]
    bad = {c: lvl for c, lvl in triage.items() if lvl not in LEVELS[:3]}
    thr = float(policy.confidence_threshold)
    problems = []
    if missing:
        problems.append(f"policy.triage has no entry for {missing}")
    if bad:
        problems.append(f"policy.triage has invalid levels {bad} (allowed: urgent, routine, none)")
    if not 0.0 < thr < 1.0:
        problems.append(f"policy.confidence_threshold must be between 0 and 1, got {thr}")
    if problems:
        raise ValueError("Invalid app policy: " + "; ".join(problems))


def decide(class_name, confidence, policy, forced_review=False):
    if forced_review or float(confidence) < float(policy.confidence_threshold):
        return "review"
    return str(policy.triage[class_name])
