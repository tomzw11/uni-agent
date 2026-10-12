# qwenpaw — uni-agent + verl GRPO 训练 Demo（零 QwenPaw 改动）

> 与 [react_simple_demo_runbook.md](react_simple_demo_runbook.md)（已跑通的 react_simple
> 基线）互补：本文档记录 qwenpaw demo 的方案、数据准备（含完整 GSM8K 转换流程）与训练用法。
> 训练链路、NPU 保守参数与排障沿用 react_simple 基线。

## 1. 目标与方案决策

在单机 8 卡 Ascend NPU 上，用 **QwenPaw（远程部署服务）** 作为 agent 跑通
**uni-agent + verl** 训练链路（GRPO / Megatron 后端 / vLLM rollout / colocate_async）。

**核心约束：不改 QwenPaw 源码**，只走其现有 REST API：

| 步骤 | QwenPaw 现有 API | 作用 |
| --- | --- | --- |
| 1 | `PUT /models/{provider_id}/config` | 把 provider `base_url` 重指到 Gateway session URL（`auto_discover=False` 阻止后台模型发现任务） |
| 2 | `POST /console/chat/task` | 后台启动一次 ReAct 回合（dict payload，含 `timeout`） |
| 3 | `GET /console/chat/task/{task_id}` | 轮询至 `finished`，取最终文本 |
| 4 | `POST session.reward_info_url` | 上报 reward（`report_reward=True`） |

Provider `base_url` 是实例全局的 → runner 并发必须 `max_concurrent_sessions=1` 串行。
QwenPaw 的 OpenAI 聊天适配器在 `base_url` 后拼接 `/chat/completions`，与 Gateway 路由
`/sessions/{session_id}/v1/chat/completions` 精确匹配，这是方案成立的基础。

**重要约束：provider 必须是自定义 provider（`freeze_url=False`）**。内置 provider
（`openai` / `modelscope` / `dashscope` 等）`freeze_url=True`，`PUT config` 的
`base_url` 更新会被**静默忽略**（PUT 返回 200 但 base_url 不变）。创建方式：
`POST /models/custom-providers`（`chat_model=OpenAIChatModel`，手动加一个模型，
任意名字——Gateway 不校验模型名），训练脚本的 `QWENPAW_PROVIDER_ID` 填该自定义
provider id。不要对该 provider 触发 `auto_discover`（runner 已固定 `False`，避免
探测 Gateway 的 `/models` 失败）。

**决策点**（已确认）：

| 项 | 选择 |
| --- | --- |
| QwenPaw 侧改动 | 零改动（Provider 重配 + 后台任务 API） |
| reward | 现在就接 GSM8K 规则 reward（`#### <number>` 匹配） |
| 模型 / 环境 | Qwen3-Coder-1.7B / 单机 8 卡 NPU（同 react_simple 基线） |
| 数据 | 冒烟 toy 10 条 + 完整 GSM8K（本地转换，见 §3） |

## 2. 训练链路

```
verl trainer (v1, colocate_async)
  → AgentFrameworkRolloutAdapter（uni_agent.framework.entry 注入）
  → gateway actor（session 管理，注入 base_url + reward_info_url）
  → run_qwenpaw（qwenpaw runner，uni_agent.agents.qwenpaw.agent）
      → PUT /models/{provider_id}/config（base_url → session URL）
      → POST /console/chat/task（运行 QwenPaw ReAct 回合）
          → QwenPaw 每次 LLM 调用都打到 Gateway → vLLM → 轨迹被记录
      → GET /console/chat/task/{task_id} 轮询
      → GSM8KReward 计算 → POST session.reward_info_url
  → verl 更新策略
```

数据行结构（与冒烟/完整 GSM8K 一致）：`prompt`（system + user）+
`extra_info.tools_kwargs.task`（`name=qwenpaw`、`metadata.ground_truth`）。

## 3. 数据准备

### 3.1 冒烟数据（链路验证）

[prepare_qwenpaw_data.py](prepare_qwenpaw_data.py)：10 条 toy GSM8K 风格数学题
（train 8 / val 2），输出 `~/data/qwenpaw/{train,test}.parquet`。

```bash
python3 examples/quickstart/training/prepare_qwenpaw_data.py
```

用途：先在 NPU 上把整条链路（provider 重配 → Gateway → vLLM → reward 上报）验证通，
toy 题保证出现正 reward，便于确认 reward 信号正常。

### 3.2 完整 GSM8K 数据转换（重点）

**注意：远程完整数据集是本地的**。远程 NPU 机上已存有完整 GSM8K 的本地 HF 格式副本
（`openai/gsm8k` main split），转换脚本读本地路径，**运行时不会下载任何数据**。

[convert_gsm8k_data.py](convert_gsm8k_data.py)：把本地完整 GSM8K 转成与冒烟数据
**完全相同的行结构**（复用 `prepare_qwenpaw_data.py` 的 `SYSTEM_PROMPT`/`build_rows`），
runner / reward / 训练脚本对两种数据零改动。

**转换流程**：

```
本地 GSM8K（HF 格式副本，question + answer）
  → 正则抽取 answer 中的 "#### <number>" 数字 → ground_truth
  → 组装 prompt = [system(输出格式约束), user(question)]
  → 组装 extra_info.tools_kwargs.task = {name: qwenpaw, metadata: {ground_truth}}
  → 输出 ~/data/qwenpaw_gsm8k/{train,test}.parquet
```

**执行**（远程，仓库根目录）：

```bash
# 全量转换（train 7473 / val 1319）
python3 examples/quickstart/training/convert_gsm8k_data.py \
    --local-dataset-path /data/datasets/gsm8k \
    --local-save-dir ~/data/qwenpaw_gsm8k

# 首次全量跑建议先子采样控制步数（如 train 256 / val 64）
python3 examples/quickstart/training/convert_gsm8k_data.py \
    --local-dataset-path /data/datasets/gsm8k \
    --num-train 256 --num-val 64
```

> `--local-dataset-path` 指向远程本地的 HF 格式 GSM8K 副本；若省略才会回退去 HF Hub
> 下载（仅本地开发用，远程必须走本地路径）。

**喂给训练脚本**：训练脚本默认读 `~/data/qwenpaw/`（冒烟）；完整 GSM8K 数据通过
`TRAIN_FILE` / `TEST_FILE` 环境变量覆盖：

```bash
TRAIN_FILE=~/data/qwenpaw_gsm8k/train.parquet \
TEST_FILE=~/data/qwenpaw_gsm8k/test.parquet \
  bash examples/quickstart/training/run_qwenpaw_npu_megatron_vllm.sh
```

## 4. 训练启动与参数

[run_qwenpaw_npu_megatron_vllm.sh](run_qwenpaw_npu_megatron_vllm.sh)：从 react_simple
脚本派生，agent_runners 换成 qwenpaw runner，并固化 NPU 保守参数
（`enable_chunked_prefill=False` / `free_cache_engine=False`，见 react_simple runbook §6）。

```bash
QWENPAW_API_URL=http://<qwenpaw-host>:8088 \
QWENPAW_PROVIDER_ID=<custom-provider-id> \
MODEL_PATH=~/models/Qwen3-Coder-1.7B \
  bash examples/quickstart/training/run_qwenpaw_npu_megatron_vllm.sh
```

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `QWENPAW_API_URL` | `http://localhost:8088` | 远程 QwenPaw 服务地址 |
| `QWENPAW_PROVIDER_ID` | `openai` | 被重配 base_url 的 provider，**必须是自定义 provider id**（内置 openai 是 `freeze_url=True`，重配无效） |
| `QWENPAW_TASK_TIMEOUT` | 600 | 后台任务服务端超时（秒） |
| `QWENPAW_POLL_INTERVAL` | 2.0 | 任务轮询间隔（秒） |
| `TRAIN_FILE` / `TEST_FILE` | `~/data/qwenpaw/{train,test}.parquet` | 数据文件（完整 GSM8K 用 §3.2 覆盖） |

runner 侧固定参数：`dispatch_mode=inline_async`、`max_concurrent_sessions=1`（provider
base_url 全局性，必须串行）、`trajectory_selection=longest`、`report_reward=True`。

## 5. 训练前联通检查（QwenPaw 侧准备是否就绪）

[check_qwenpaw_connectivity.py](check_qwenpaw_connectivity.py) 是独立联检脚本，训练前
逐项确认 QwenPaw 侧已就绪，失败项直接给出原因：

```bash
# 基础检查：health + agent + provider 可重配
python3 examples/quickstart/training/check_qwenpaw_connectivity.py \
    --base-url http://<qwenpaw-host>:8088 --provider-id <custom-provider-id>

# 端到端（可选）：本地起 mock OpenAI 端点，真实跑一个 console chat 任务，
# 验证 provider 重配后 ReAct 回合的 LLM 调用确实打到新 base_url；结束后恢复原 base_url
python3 examples/quickstart/training/check_qwenpaw_connectivity.py \
    --base-url http://<qwenpaw-host>:8088 --provider-id <custom-provider-id> --e2e
```

检查项与判定：

| 检查 | 端点 | 失败条件 |
| --- | --- | --- |
| 服务就绪 | `GET /api/healthz` | 503（仍在启动）或不可达 |
| agent 可用 | `GET /agents` | `X-Agent-Id` 解析的 agent 不存在 / disabled / `available_in_chat=false`；无 `active_model` 给出告警 |
| provider 可重配 | `GET /models` | provider 不存在 / `freeze_url=true`（内置 provider 静默忽略 base_url 更新）/ `enabled=false` |
| e2e（`--e2e`） | PUT config + POST/GET chat task | mock 未收到任何 `/chat/completions`（ReAct 回合没打到重配后的 base_url）/ 任务未 `finished` |

e2e 模式会临时把 provider base_url 指向本地 mock（`finally` 中恢复原值），不影响生产；
但训练期间该自定义 provider 不能同时用于生产会话。

## 6. reward 机制

`run_qwenpaw` 完成回合后计算 reward（[agent.py](../../../uni_agent/agents/qwenpaw/agent.py)）：

- 有 `ground_truth` 且回合完成 → `GSM8KReward().compute(content, ground_truth)`
  （`#### <number>` 匹配，兜底取最后数字，1.0/0.0）
- 完成但无 `ground_truth` → 1.0（冒烟兜底）
- 未完成 → 0.0

`report_reward=True` 时 POST 到 `session.reward_info_url`，框架优先采用 runner 上报的
reward。

## 7. 验证成功信号

- agent 日志出现 `qwenpaw runner done: sample_index=... finished=True reward=... reward_posted=True`
- verl 日志训练 step 正常推进，reward 均值可见
- 训练前模型常不会输出 `#### <number>` 格式 → 初始 reward 偏低属预期，随训练应上升
