# react_simple — 最小 uni-agent + verl GRPO 训练 Demo

## 目的

在单机多卡 Ascend NPU 上，用自定义的极小任务跑通 **uni-agent + verl** 的训练链路
（GRPO / Megatron 后端 / vLLM rollout / colocate_async）。数据与 reward 均为自定义，
只用于确认框架链路可用，不做任何真实任务评估。

链路：`verl trainer (colocate_async)` → `AgentFrameworkRolloutAdapter` → gateway → react agent
（调用 vLLM）→ `task_runner.run_task`（local sandbox 执行命令）→ `TaskResult.reward`
POST 回 `session.reward_info_url` → verl 学习。

## 文件清单

框架改动（uni-agent 仓库内，最小改动）：

| 文件 | 改动 |
| --- | --- |
| `uni_agent/tasks/react_simple/task.py` | 新增：极简 Task（shell 命令 stdout 匹配即 reward=1） |
| `uni_agent/tasks/registry.py` | `TASK_MODULES` 注册 `react_simple`（一行） |

Demo 文件（uni-agent 仓库内，examples/ 下）：

| 文件 | 作用 |
| --- | --- |
| `examples/quickstart/training/prepare_react_simple_data.py` | 生成 train(8)/test(2) parquet，每行带 `extra_info.tools_kwargs.task` |
| `examples/quickstart/training/task_config_react_simple.yaml` | 任务配置：react agent + local sandbox，只暴露 shell/submit |
| `examples/quickstart/training/run_react_simple_npu_megatron_vllm.sh` | 训练启动脚本（本文件同目录） |
| `examples/quickstart/training/react_simple_demo.md` | 本说明 |

verl 本体与 uni-agent 的 framework/entry、task_runner、sandbox、agent **零改动**，
全部走 verl 官方扩展点注入。

## 环境准备（远程 NPU 机一次性）

- 系统：Ascend NPU（8 卡），CANN ≥ 8.0
- Python 包：`torch`、`torch_npu`、`vllm-ascend`、`pandas`、`pyarrow`、`mbridge`（Megatron-Bridge）
- `ray start --head`（脚本直接执行 verl，不用 `ray job submit`）
- 模型下载到 `~/models/Qwen3-Coder-1.7B`

## 部署

```bash
# verl 是 uni-agent 的 git 子模块（volcengine/verl @ 78bba31d），必须递归 clone
git clone --recurse-submodules https://github.com/tomzw11/uni-agent.git
cd uni-agent && git submodule update --init --recursive
git -C verl rev-parse HEAD        # 应输出 78bba31d1a6b95084d83f8bddd59c190ca704144
```

> 本地开发机器若子模块未 checkout，而 verl 放在 uni-agent 同级目录（如
> `~/Desktop/uniagent/verl`），启动脚本会自动 fallback 到同级目录，无需额外配置。

## 运行

```bash
# 1. 生成训练/验证数据（默认写入 ~/data/react_simple/）
python3 examples/quickstart/training/prepare_react_simple_data.py

# 2. 启动训练（从仓库根目录；所有参数可用环境变量覆盖）
MODEL_PATH=~/models/Qwen3-Coder-1.7B \
  bash examples/quickstart/training/run_react_simple_npu_megatron_vllm.sh
```

常用覆盖项（默认值见脚本顶部）：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `~/models/Qwen3-Coder-1.7B` | 模型路径 |
| `TRAIN_FILE` / `TEST_FILE` | `~/data/react_simple/train|test.parquet` | 数据文件 |
| `TP` / `PP` / `CP` | 4 / 1 / 1 | Megatron 并行（8 卡 → DP=2） |
| `GEN_TP` | 2 | vLLM 张量并行 |
| `N_RESP_PER_PROMPT` | 4 | 每 prompt 采样数 |
| `TOTAL_EPOCHS` | 10 | 8 行数据 ≈ 10 步 |
| `LOGGER` | `['console']` | 可加 `['console','wandb']` 看曲线 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0..7` | NPU 设备 |

## 验证成功信号

- agent 日志目录出现 `react_simple done: reward=... finished=...`
- verl 日志出现训练 step：reward 均值、policy loss 更新
- reward 均值随训练提升（8 个确定性命令，模型应能学会 shell 输出）

## 常见问题

- **并行度报错**：8 卡须满足 `world % (TP×PP×CP) == 0`；改 `TP` 需同步检查。
- **mbridge 报错**：Megatron-Bridge 未装或版本不匹配时，先用 `USE_MBRIDGE=False` 降级排查。
- **vLLM 起不来**：确认 `vllm-ascend` 与 CANN 配套；colocate 下显存不足时调低
  `actor_rollout_ref.rollout.gpu_memory_utilization`（脚本内 0.7）。
- **task 未注册**：确认数据行内 `extra_info.tools_kwargs.task.name == react_simple`，
  且 `uni_agent/tasks/registry.py` 已注册。
