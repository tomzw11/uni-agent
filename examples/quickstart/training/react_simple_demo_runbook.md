# react_simple — uni-agent + verl GRPO 训练 Demo 全流程总结

> 与 [react_simple_demo.md](react_simple_demo.md)（使用说明）互补：本文档记录方案
> 决策、框架改动、部署步骤与完整的排障过程，作为后续在此环境扩展训练的基线。

## 1. 目标与方案决策

在单机 8 卡 Ascend NPU 上，用自定义的最小任务跑通 **uni-agent + verl** 训练链路
（GRPO / Megatron 后端 / vLLM rollout / colocate_async）。数据与 reward 均为自定义，
只验证链路可用，不做真实任务评估。

**为什么不用 GSM8K**：GSM8K 是纯数学数据，没有 `extra_info.tools_kwargs.task` 字段，
不走 uni-agent 的 agent/sandbox/task 框架，reward 走 naive manager —— 无法演示
uni-agent。因此自造极简 react 任务 `react_simple`。

**决策点**（已确认）：

| 项 | 选择 | 理由 |
| --- | --- | --- |
| 任务 | 自造 `react_simple` | 必须走 agent 框架的最小闭环 |
| 模型 | Qwen3-Coder-1.7B（dense） | 小模型，冒烟快；MoE 35B 太重 |
| 环境 | 单机 8 卡 NPU，Megatron 后端 | 用户指定 |
| rollout | vLLM colocate 同卡，`gen_tp=2` | 与训练共享 NPU，节省资源 |

## 2. 训练链路

```
verl trainer (v1, colocate_async)
  → AgentFrameworkRolloutAdapter（uni_agent.framework.entry 注入）
  → gateway actor（session 管理，注入 base_url + reward_info_url）
  → task_runner.run_task（react_simple task）
      → react agent（调用 vLLM 生成）
      → LocalSandbox（宿主 bash 执行命令）
  → TaskResult.reward POST 回 session.reward_info_url
  → verl 更新策略
```

数据行结构（每行）：`prompt`（system+user）+ `extra_info.tools_kwargs.task`（
`name=react_simple`、`sandbox.provider=local`、`metadata.command/expected`）。

## 3. 框架改动清单（uni-agent 仓库内，最小改动）

| 文件 | 改动 | 说明 |
| --- | --- | --- |
| `uni_agent/tasks/react_simple/task.py` | 新增 | 极简 Task：agent 执行 shell 命令，stdout 匹配 `expected` 即 reward=1 |
| `uni_agent/tasks/registry.py` | 1 行 | `TASK_MODULES` 注册 `react_simple` |
| `uni_agent/sandbox/local.py` | +41 行 | `supports_shell=True` + `open_shell()` 返回宿主 `bash -lc` 会话，绕开 tmux/root |

verl 本体与 uni-agent 的 framework/entry、task_runner、react agent **零改动**，
全部走 verl 官方扩展点注入。

Demo 文件（examples/quickstart/training/ 下）：

| 文件 | 作用 |
| --- | --- |
| `prepare_react_simple_data.py` | 生成 train(8)/test(2) parquet |
| `task_config_react_simple.yaml` | 任务配置：react agent + local sandbox |
| `run_react_simple_npu_megatron_vllm.sh` | 训练启动脚本（已验证的默认参数见 §6） |
| `react_simple_demo.md` / `react_simple_demo_runbook.md` | 使用说明 / 本文档 |

## 4. 部署与运行（远程 NPU 机）

```bash
git clone --recurse-submodules https://github.com/tomzw11/uni-agent.git
cd uni-agent && git submodule update --init --recursive
git -C verl rev-parse HEAD        # 应输出 78bba31d1a6b95084d83f8bddd59c190ca704144

python3 examples/quickstart/training/prepare_react_simple_data.py
MODEL_PATH=~/models/Qwen3-Coder-1.7B \
  bash examples/quickstart/training/run_react_simple_npu_megatron_vllm.sh
```

前置：CANN + torch_npu + vllm-ascend + mbridge + `ray start --head`。

## 5. 排障记录（按主题合并）

### P1. 工具 `shell` 启动失败 → tmux/root 权限

**现象**：`tool 'shell' failed to start (installer exit 100)` +
`dpkg lock Permission denied, are you root?`，随后训练 step 报
`AssertionError: number of items: [0] < k_partitions: [2]`（batch 空）。

**根因**：LocalSandbox 默认 `supports_shell=False`，shell 工具走 TmuxShell 分支，
沙箱内无 tmux 时尝试 `apt-get install tmux`，宿主无 root → 安装失败 → 工具不可用
→ 所有 episode 无产出 → rollout batch 空。

**修复**：给 LocalSandbox 实现原生 shell 会话（`supports_shell=True` + `open_shell()`
复用宿主 `bash -lc`），shell 工具自动走 SandboxShell 分支，不再依赖 tmux/root。
本机验证：`echo hello` 正常返回。

### P2. 远程代码不同步 → SyntaxError

**现象**：ray task 报 `RayTaskError(SyntaxError)`，session 全失败。

**根因**：远程 `uni-agent-main` 代码与本地不一致（手动同步遗漏/损坏）。
用 `python3 -m compileall -q uni_agent` 可一行定位坏文件。

**修复**：统一远程代码版本（git 拉取或完整同步），确保 `local.py`、`shell.py`
等与本地一致。

### P3. Megatron fusion 算子 → NPU vector core 崩溃

**现象**：`ERR00100 PTA call acl api failed` / `vector core execution is abnormal`
/ `runtime result = 507035`，训练进程退出。

**根因**：启动脚本照搬官方 GPU 脚本的 flash 类 fusion
（`masked_softmax_fusion` / `bias_activation_fusion` / `bias_dropout_fusion` /
`gradient_accumulation_fusion`），NPU(mbridge) 上不支持。

**修复**：以上 fusion 全部关闭（脚本默认已改）。

### P4. vLLM-Ascend graph 模式 → MTE 地址越界

**现象**：vLLM 侧 `acl_graph.py` 同步失败，`The DDR address of the MTE instruction
is out of range`，vLLM Engine 崩溃 → agent 请求 `127.0.0.1:8000`
`Connection refused` → 32 个 session 全失败。

**根因**：vllm-ascend 用静态 graph 重放；uni-agent 的 multi-turn 变长请求 +
`enable_chunked_prefill=True` 让每个调度步的 batch 组成（prefill 块混排、KV block
布局）高度动态，偏离 graph 捕获时的静态 shape 假设 → MTE 搬运地址越界。
（标准 verl 数据 shape 规整，故同版本组合下 verl 正常。）

**修复**（已验证有效）：

```bash
actor_rollout_ref.rollout.enable_chunked_prefill=False
actor_rollout_ref.rollout.free_cache_engine=False
```

`enforce_eager=True` 不可用（vllm-ascend 依赖 graph 模式），保持 `False`。

### P5. `no tool call found` 告警（正常现象）

**现象**：训练日志出现 `no tool call found` 告警。

**说明**：未训练模型不常输出工具调用格式，react agent 按 `_NO_TOOL_CALL_RETRY`
提示语重试有限次后结束该轮。这是 GRPO 训练初期预期现象——只有调用工具才有
reward，训练后模型会逐步学会工具调用格式，告警比例下降。非错误。

## 6. 最终验证状态

- 链路已跑通：8 prompt × 4 采样，训练 step 正常推进，reward 回传正常。
- 已固化到启动脚本的保守参数（NPU）：
  - Megatron fusion 全关（`apply_rope/masked_softmax/bias_activation/bias_dropout/gradient_accumulation`）
  - `enforce_eager=False`（vllm-ascend 保持 graph）
- 经命令行覆盖验证、建议固化进脚本的参数：
  - `enable_chunked_prefill=False`
  - `free_cache_engine=False`

## 7. 已知注意点

- 并行度：8 卡须满足 `world % (TP×PP×CP) == 0`；当前 TP=4/PP=1/CP=1 → DP=2。
- 数据量不是 `items: [0]` 报错的直接原因；该报错根因是 rollout 无有效产出（见 P1/P4）。
- 远程部署优先保持代码版本一致（git pull），避免手动同步引入 P2 类问题。
- 升级 vllm-ascend/CANN 后，P4 的保守参数可能需要复核（graph 行为可能变化）。
