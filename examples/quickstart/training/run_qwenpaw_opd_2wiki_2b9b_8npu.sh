#!/usr/bin/env bash
set -xeuo pipefail

# Full online-OPD run: 2B student + 9B teacher, QwenPaw agent, 8 NPUs (FSDP2).
# Design and decisions: opd/OPD_TRAIN_2B9B.md.
#
# Topology on one node with 8 Ascend cards:
#   - student pool (4 cards): FSDP2 actor (DP=4) + vLLM(Ascend) rollout (TP=1, DP=4)
#   - teacher pool (4 cards): standalone vLLM teacher (TP=1, 4 replicas), forward-only
#   - QwenPaw service: CPU side, drives the student through the Gateway (serialized)
#
# Unlike the 4p smoke run, the teacher here is a DIFFERENT checkpoint (9B). TEACHER_MODEL_PATH
# is required and checked against MODEL_PATH: a silent teacher==student makes k3 start at ~0
# and the run trains nothing.
#
# Prereqs:
#   - ray start --head; CANN + torch_npu + vllm-ascend
#   - QwenPaw up with a CUSTOM provider (freeze_url=false), API unter /api
#   - agent workspace may contain ./kb_<qid>/ for every qid (created below if QWENPAW_WORKSPACE set)
#   - train parquet produced by ALIGN_RECHECK.md (62 filtered minus overlong ones)
#
# Env overrides: MODEL_PATH, TEACHER_MODEL_PATH (required), TRAIN_FILE, DATA_ROOT,
#   QWENPAW_API_URL, QWENPAW_PROVIDER_ID, QWENPAW_WORKSPACE, KB_ROOT, and all below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIAGENT_ROOT="${UNIAGENT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
if [ -d "${UNIAGENT_ROOT}/verl" ] && [ -n "$(ls -A "${UNIAGENT_ROOT}/verl" 2>/dev/null)" ]; then
  VERL_ROOT=${VERL_ROOT:-${UNIAGENT_ROOT}/verl}
else
  VERL_ROOT=${VERL_ROOT:-${UNIAGENT_ROOT}/../verl}
fi

# --- NPU runtime env ----------------------------------------------------------
if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export PYTHONPATH="${UNIAGENT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

# --- Settings ----------------------------------------------------------------
project_name=${PROJECT_NAME:-"qwenpaw-opd-2wiki-2b9b"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H%M)_exp"}

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen3.5-2B"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-""}

DATA_ROOT=${DATA_ROOT:-"/mnt/share_space/opd/data/2wiki"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/qa/tool/train_filtered_59.parquet"}
TEST_FILE=${TEST_FILE:-"${TRAIN_FILE}"}

QWENPAW_API_URL=${QWENPAW_API_URL:-"http://localhost:8088/api"}
QWENPAW_PROVIDER_ID=${QWENPAW_PROVIDER_ID:-"opd-gateway"}
QWENPAW_WORKSPACE=${QWENPAW_WORKSPACE:-""}
KB_ROOT=${KB_ROOT:-"${DATA_ROOT}"}

CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${HOME}/logs/${project_name}/${exp_name}"}

TOOL_PARSER=${TOOL_PARSER:-"qwen3_xml"}
GATEWAY_COUNT=${GATEWAY_COUNT:-4}

# --- Length budget: must satisfy teacher max_model_len >= prompt + response + 1 ---
max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-24576}
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

# --- Algorithm: pure on-policy distillation (no task reward, no policy gradient) ---
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}

# --- Parallelism: 4 student cards + 4 teacher cards ------------------------------
gen_tp=${GEN_TP:-1}
train_tp=${TP:-1}
train_pp=${PP:-1}
train_cp=${CP:-1}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-4}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.85}
teacher_enforce_eager=${TEACHER_ENFORCE_EAGER:-True}
teacher_chunked_prefill=${TEACHER_CHUNKED_PREFILL:-False}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.45}

# --- Batching: one on-policy session per prompt (QwenPaw is serialized) -----------
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-4}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-1}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-4}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
test_freq=${TEST_FREQ:--1}
total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-4}
actor_lr=${ACTOR_LR:-1e-6}
logger=${LOGGER:-"['console']"}

# --- Preflight ---------------------------------------------------------------
die() { echo "FATAL: $*" >&2; exit 1; }

[ -n "${TEACHER_MODEL_PATH}" ] || die "TEACHER_MODEL_PATH is required (do not let the teacher default to the student)"
[ -d "${MODEL_PATH}" ] || die "student weights not found: ${MODEL_PATH}"
[ -d "${TEACHER_MODEL_PATH}" ] || die "teacher weights not found: ${TEACHER_MODEL_PATH}"
if [ "$(readlink -f "${MODEL_PATH}")" = "$(readlink -f "${TEACHER_MODEL_PATH}")" ]; then
  die "student and teacher resolve to the same directory; k3 would start at ~0"
fi
[ -f "${TRAIN_FILE}" ] || die "train parquet not found: ${TRAIN_FILE} (see ALIGN_RECHECK.md section 20/21)"
n_cards=$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")
[ "${n_cards}" -ge "$((NGPUS_PER_NODE + TEACHER_NGPUS_PER_NODE))" ] \
  || die "ASCEND_RT_VISIBLE_DEVICES has ${n_cards} cards, need $((NGPUS_PER_NODE + TEACHER_NGPUS_PER_NODE))"

# --- Data: sanity check the training subset -----------------------------------
python3 - "${TRAIN_FILE}" <<'PY'
import sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
print(f"train rows = {len(df)}")
if len(df) < 20:
    raise SystemExit("too few training rows")
PY

# --- QwenPaw workspace: expose ./kb_<qid>/ for each training qid ------------------
if [ -n "${QWENPAW_WORKSPACE}" ]; then
  mkdir -p "${QWENPAW_WORKSPACE}"
  TRAIN_FILE="${TRAIN_FILE}" KB_ROOT="${KB_ROOT}" QWENPAW_WORKSPACE="${QWENPAW_WORKSPACE}" python3 -c "
import os
from pathlib import Path
import pandas as pd
kb_root = Path(os.environ['KB_ROOT'])
workspace = Path(os.environ['QWENPAW_WORKSPACE'])
qids = pd.read_parquet(os.environ['TRAIN_FILE'])['extra_info'].map(lambda info: info['qid'])
for qid in sorted(set(map(str, qids))):
    src = kb_root / 'kb_episodes' / qid
    dst = workspace / f'kb_{qid}'
    if not src.is_dir():
        raise SystemExit(f'missing KB dir: {src}')
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=True)
print(f'linked {len(set(map(str, qids)))} kb_<qid> dirs into {workspace}')
"
fi

# --- Launch -------------------------------------------------------------------
start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=${num_warmup_batches} \
    transfer_queue.enable=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_model_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.runner_fqn=uni_agent.agents.qwenpaw.agent.run_qwenpaw \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.dispatch_mode=inline_async \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.max_concurrent_sessions=1 \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.trajectory_selection=longest \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.runner_kwargs.qwenpaw_api_url=${QWENPAW_API_URL} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.runner_kwargs.provider_id=${QWENPAW_PROVIDER_ID} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.runner_kwargs.task_timeout=${QWENPAW_TASK_TIMEOUT:-1800} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qwenpaw.runner_kwargs.poll_interval=${QWENPAW_POLL_INTERVAL:-2.0} \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=True \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.offload_policy=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    distillation.enabled=True \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE} \
    distillation.distillation_loss.loss_mode=k3 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL_PATH}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.mode=async \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util} \
    distillation.teacher_models.teacher_model.inference.prompt_length=${max_prompt_length} \
    distillation.teacher_models.teacher_model.inference.response_length=${max_response_length} \
    distillation.teacher_models.teacher_model.inference.max_model_len=$((max_prompt_length + max_response_length + 1)) \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${actor_ppo_max_token_len} \
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=${teacher_chunked_prefill} \
    distillation.teacher_models.teacher_model.inference.free_cache_engine=False \
    distillation.teacher_models.teacher_model.inference.enforce_eager=${teacher_enforce_eager} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=${logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.device=npu \
    trainer.balance_batch=True \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.test_freq="${test_freq}" \
    "$@" 2>&1 | tee "logs/qwenpaw-opd-2b9b-${start_time}.log"
