Fixed inference.py output format and score validation issues:
- Clamp final_score to (0, 1) in all code paths (lines 251, 307, 328)
- Move task_id validation inside try block to ensure [START] is always logged before errors
- Initialize env=None to safely handle env.close() in finally block
- Log failed env.step() calls with error details to prevent format mismatch
- All rewards and scores now strictly within (0.01, 0.99) per validator requirement
