#!/usr/bin/env bash
set -xeuo pipefail

# Minimal uni-agent GRPO demo: react_simple task + Qwen3-Coder-1.7B.
# Single node, 8 Ascend NPU cards, Megatron (mbridge) backend + vLLM(Ascend) colocate.
#
# Prereqs (paths auto-derived from script location; run from the uni-agent repo root):
#   - `ray start --head` running (this script executes verl directly, not `ray job submit`)
#   - CANN + torch_npu + vllm-ascend installed; mbridge in PYTHONPATH
#   - Model downloaded to $MODEL_PATH (default ~/models/Qwen3-Coder-1.7B)
#   - Data prepared once:
#       python3 examples/quickstart/training/prepare_react_simple_data.py
#
# verl is a git submodule of uni-agent (volcengine/verl @ 78bba31d). If the
# submodule is not checked out locally but verl exists as a sibling directory
# (e.g. ~/Desktop/uniagent/verl), the script falls back to it automatically.
#
# Everything below is overridable via env, e.g.:
#   MODEL_PATH=/data/models/Qwen3-Coder-1.7B ./examples/quickstart/training/run_react_simple_npu_megatron_vllm.sh

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

# --- Demo settings -------------------------------------------------------------
project_name=${PROJECT_NAME:-"react_simple-npu-megatron-demo"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H%M)_exp"}

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen3-Coder-1.7B"}
TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/react_simple/train.parquet"}
TEST_FILE=${TEST_FILE:-"${HOME}/data/react_simple/test.parquet"}
TASK_CONFIG=${TASK_CONFIG:-"${UNIAGENT_ROOT}/examples/quickstart/training/task_config_react_simple.yaml"}

CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${HOME}/logs/${project_name}/${exp_name}"}

TOOL_PARSER=${TOOL_PARSER:-"qwen3_coder"}   # tool-call parser; MUST match the model chat template
GATEWAY_COUNT=${GATEWAY_COUNT:-8}           # gateway actors fronting the rollout engine
CONCURRENCY=${CONCURRENCY:-64}              # max in-flight rollout sessions
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}

# --- Algorithm -----------------------------------------------------------------
adv_estimator=${ADV_ESTIMATOR:-grpo}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_ratio_c=${CLIP_RATIO_C:-10.0}
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}
loss_mode=${LOSS_MODE:-vanilla}

# react_simple prompts are tiny; keep the response budget for multi-turn tool loops.
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}

# --- Parallelism: 8 NPU cards, Megatron TP=4/PP=1/CP=1 (DP=2), vLLM TP=2 ------
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-2}
train_tp=${TP:-4}
train_pp=${PP:-1}
train_cp=${CP:-1}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# --- Batching: 8-row dataset; each prompt sampled n_resp_per_prompt times -------
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-4}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
test_freq=${TEST_FREQ:-1}
total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-10}
logger=${LOGGER:-"['console']"}   # add 'wandb' to log curves (needs login)

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
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
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    +actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model'] \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.95} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K:--1} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=${logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.test_freq="${test_freq}" \
    "$@"
