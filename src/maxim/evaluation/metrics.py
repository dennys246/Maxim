def success_rate(results):
    return sum(r.get("valid", 0) for r in results) / max(len(results), 1)

def average_score(results):
    scores = [r.get("score", 0) for r in results]
    return sum(scores) / max(len(scores), 1)