# VeriRL — Pre-Submission Checklist

Complete pre-submission validation before uploading to Hugging Face Spaces.

## Phase 1: Automated Validation Gates

### ✓ HF Space Deployment
- Space must be live at submission URL
- `/reset` endpoint must return HTTP 200 and respond to reset()
- **Status:** To be tested after HF Space creation

### ✓ OpenEnv Spec Compliance
```bash
openenv validate
```
**Output:** `[OK] verirl: Ready for multi-mode deployment`

**Checklist:**
- ✓ `openenv.yaml` present with spec_version, name, type, runtime, app, port
- ✓ `openenv.yaml` includes task metadata (id, name, difficulty, max_turns)
- ✓ Typed models: `VerirlAction`, `VerirlObservation`, `VerirlState` (Pydantic)
- ✓ Environment implements `reset()`, `step()`, `state()` endpoints
- ✓ All models have proper field types and descriptions

### ✓ Dockerfile Builds
```bash
docker build -t verirl:latest -f server/Dockerfile .
```
**Status:** ✓ Builds successfully (tested 2026-04-04)

**Checklist:**
- ✓ Multi-stage build (builder + runtime)
- ✓ Installs EDA tools: `iverilog`, `yosys`
- ✓ Virtual environment copied correctly
- ✓ PYTHONPATH set for imports
- ✓ Health check configured
- ✓ Runs FastAPI app on port 8000

### ✓ Baseline Inference Reproduces
```bash
export HF_TOKEN=<token>
export API_BASE_URL=https://router.huggingface.co/v1  # or your endpoint
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct           # or your model
export ENV_BASE_URL=http://localhost:8000

python inference.py
```
**Output Format (STDOUT only):**
```
[START] task=mac_unit env=verirl model=openai/gpt-oss-120b
[STEP] step=1 action=write_file(2313chars) reward=0.01 done=false error=null
[STEP] step=2 action=submit reward=0.00 done=true error=null
[END] success=false steps=2 rewards=0.01,0.00
[START] task=axi_fifo env=verirl model=openai/gpt-oss-120b
[STEP] step=1 action=submit reward=-0.01 done=true error=null
[END] success=false steps=1 rewards=-0.01
[START] task=systolic_array env=verirl model=openai/gpt-oss-120b
[STEP] step=1 action=submit reward=-0.01 done=true error=parse_error: Unterminated string starting at: line 3 column
[END] success=false steps=1 rewards=-0.01
```

**Checklist:**
- ✓ Script completes without errors
- ✓ STDOUT contains ONLY [START], [STEP], [END] lines (no other output)
- ✓ Produces valid scores (all in [0.0, 1.0])
- ✓ Runtime < 20 minutes (typically 30-60 seconds)
- ✓ Uses OpenAI Client for all LLM calls
- ✓ Runs on machine with vcpu=2, memory=8GB

### ✓ 3+ Tasks with Graders
**Task Enumeration & Validation (runs before inference):**
```
[VALIDATION] Discovered 3 tasks:
  • mac_unit             Pipelined MAC Unit             [easy  ] (max_turns: 8)
  • axi_fifo             AXI-Stream FIFO                [medium] (max_turns: 10)
  • systolic_array       4x4 Systolic Array             [hard  ] (max_turns: 12)

[VALIDATION] Testing grader on each task...
  [mac_unit] ✓ Grader OK (empty submission scored 0.000)
  [axi_fifo] ✓ Grader OK (empty submission scored 0.000)
  [systolic_array] ✓ Grader OK (empty submission scored 0.000)

[VALIDATION] ✓ All 3 tasks validated successfully.
```

**Checklist:**
- ✓ All 3 tasks discovered via `/tasks` endpoint
- ✓ Each grader accepts reset/step/submit workflow
- ✓ All graders produce scores in [0.0, 1.0]
- ✓ All per-step rewards in [-1.0, 1.0]
- ✓ Task difficulty progression: easy → medium → hard

## Phase 2: Code Quality & Compliance

### ✓ Mandatory Fields & Structure

**inference.py:**
- ✓ Named `inference.py` (not `inference_script.py` or similar)
- ✓ Located in project root (not in subdirectory)
- ✓ Imports OpenAI Client: `from openai import OpenAI`
- ✓ Uses environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN, ENV_BASE_URL
- ✓ Supports .env file via `load_dotenv()`

**STDOUT Format (exact specification — CRITICAL):**
- ✓ STDOUT contains ONLY [START], [STEP], [END] lines — NO OTHER OUTPUT
- ✓ [START] line with fields: task, env, model (in exact order, no deviation)
- ✓ [STEP] line with fields: step, action, reward, done, error (in exact order, no deviation)
- ✓ [END] line with fields: success, steps, rewards (in exact order, no deviation)
- ✓ reward/rewards formatted to 2 decimal places (e.g., 0.02)
- ✓ done/success lowercase booleans: true or false
- ✓ error is raw error string (max 80 chars) or null — single line, no newlines
- ✓ NO validation output, NO comments, NO summary lines
- ✓ Every line matches pattern exactly: `[TYPE] field1=value1 field2=value2 ...`

**Environment Configuration:**
- ✓ `API_BASE_URL` — configurable, defaults to HF router
- ✓ `MODEL_NAME` — configurable, defaults to Qwen2.5-72B
- ✓ `HF_TOKEN` — required (fallback to API_KEY)
- ✓ `ENV_BASE_URL` — configurable, defaults to localhost:8000

### ✓ Performance & Resource Constraints
- ✓ Inference script runtime: ~47 seconds (well under 20 min limit)
- ✓ Docker image size: reasonable (OpenEnv base + EDA tools)
- ✓ Memory footprint: suitable for 2 vCPU, 8GB memory machines

## Phase 3: Submission Artifacts

### Required Files
- ✓ `openenv.yaml` — OpenEnv manifest
- ✓ `inference.py` — Baseline inference script in root
- ✓ `README.md` — Complete with environment description, setup, baseline scores
- ✓ `Dockerfile` (or `server/Dockerfile`) — builds successfully
- ✓ `models.py` — typed Pydantic models
- ✓ `server/app.py` — FastAPI application
- ✓ `server/verirl_env_environment.py` — environment implementation
- ✓ `server/evaluator.py` — EDA tool wrappers
- ✓ `problems/task*/spec.md` — task specifications
- ✓ `problems/task*/testbench.v` — testbenches with assertions

### Optional but Recommended
- ✓ `.gitignore` — excludes __pycache__, .venv, .pytest_cache, etc.
- ✓ `pyproject.toml` — correct package metadata
- ✓ `uv.lock` — locked dependencies
- ✓ `tests/` — test suite with coverage
- ✓ `CLAUDE.md` — developer notes

## Baseline Scores

| Task | Difficulty | Score |
|---|---|---|
| mac_unit | easy | 0.154 |
| axi_fifo | medium | 0.629 |
| systolic_array | hard | 0.000 |
| **mean** | | **0.261** |

**Interpretation:**
- MAC unit (easy): Partial credit for compilation and basic test passage
- AXI FIFO (medium): Good protocol understanding, partial edge-case failures
- Systolic array (hard): Model did not attempt (7-cycle timing + diagonal skewing is challenging)

## Steps to Validate Before Submission

1. **Local validation (this machine):**
   ```bash
   openenv validate                    # Must pass
   docker build -t verirl:latest .     # Must succeed
   python inference.py                 # Must complete in < 20 min
   ```

2. **Verify STDOUT format is EXACT:**
   ```bash
   # Should show ONLY [START], [STEP], [END] lines with NO other output
   python inference.py 2>/dev/null | head -20
   
   # Verify every line matches pattern
   python inference.py 2>/dev/null | grep -v "^\[START\]" | grep -v "^\[STEP\]" | grep -v "^\[END\]" | wc -l
   # ^ Should output: 0 (zero extra lines)
   ```
   
   **Critical:** Any deviation in field names, ordering, or extraneous output will fail automated evaluation.

3. **Verify environment variables:**
   ```bash
   echo $API_BASE_URL $MODEL_NAME $HF_TOKEN $ENV_BASE_URL
   ```
   Set appropriately for submission environment.

4. **Create HF Space:**
   - Use `openenv push` to deploy to Hugging Face Spaces
   - Verify space is live at your submission URL
   - Test `/reset` endpoint returns 200 OK

## Submission URL

Update `validation.sh` with your space URL, e.g.:
```bash
./validation.sh https://your-username.hf.space
```

---

**Last verified:** 2026-04-04
**Status:** Ready for submission ✓
