"""Structural difference metrics for plan comparison.

Provides Jaccard distance on action sequences to measure structural
novelty between successive plan attempts. Used by B4 replanning to
enforce the anti-repetition invariant: each replan attempt must be
structurally different from all prior attempts.
"""

from __future__ import annotations


def action_signature(action: dict) -> str:
    """Extract a canonical signature from an action dict.

    Uses tool_name + sorted param keys as the identity.  Two actions
    with the same tool and same parameter names are considered the
    same structural step regardless of parameter values.
    """
    tool = action.get("tool_name", "")
    params = action.get("params", {})
    if isinstance(params, dict):
        param_keys = ",".join(sorted(params.keys()))
    else:
        param_keys = ""
    return f"{tool}({param_keys})"


def plan_signature_set(actions: list[dict]) -> set[str]:
    """Convert a plan's action list into a set of action signatures."""
    return {action_signature(a) for a in actions}


def jaccard_distance(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard distance between two signature sets.

    Returns 0.0 when identical, 1.0 when fully disjoint.
    Empty-vs-empty is defined as 0.0 (no distance).
    """
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - len(intersection) / len(union)


def plan_distance(plan_a: list[dict], plan_b: list[dict]) -> float:
    """Jaccard distance between two plans' action sequences."""
    return jaccard_distance(
        plan_signature_set(plan_a),
        plan_signature_set(plan_b),
    )


def min_distance_from_prior(
    candidate: list[dict],
    prior_attempts: list[list[dict]],
) -> float:
    """Minimum Jaccard distance from a candidate to any prior attempt.

    Returns 1.0 if there are no prior attempts (maximally novel).
    """
    if not prior_attempts:
        return 1.0
    return min(plan_distance(candidate, prior) for prior in prior_attempts)


def all_pairwise_distances(plans: list[list[dict]]) -> list[float]:
    """All pairwise Jaccard distances between a list of plans.

    Returns a flat list of distances for statistical analysis.
    """
    distances = []
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            distances.append(plan_distance(plans[i], plans[j]))
    return distances
