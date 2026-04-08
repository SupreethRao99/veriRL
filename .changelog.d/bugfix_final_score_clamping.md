Fixed score clamping to ensure all task scores are strictly in [0.01, 0.99]:

Server-side (evaluator.py) — authoritative clamping:
- breakdown["compile"] = 0.99 (was 1.0, now clamped)
- breakdown["sim"] = max(0.01, min(ratio, 0.99)) per test pass rate
- breakdown["timing"] = max(0.01, min(..., 0.99)) for all task variants
- breakdown["area"] = max(0.01, min(ratio, 0.99)) per reference cells
- final_score = max(0.01, min(0.99, weighted_sum)) before return

Inference-side (inference.py) — defensive clamping before output:
- All step rewards: reward = safe_score(result.reward or 0.01)
- All final_scores: final_score = safe_score(obs.final_score or 0.01)
- Validation: apply safe_score() before checking task validity
- All values in [END] rewards array are pre-clamped via safe_score()

Models (models.py) — documentation:
- Updated final_score descriptions to clarify [0.01, 0.99] range
- Removed strict Pydantic validators (trust server guarantee)
