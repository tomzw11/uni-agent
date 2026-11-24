# QwenPaw 对接 uni-agent：协议与配置参考

> 范围：只描述**协议、接口与配置**，不涉及部署步骤和训练运维。
> 面向读者：需要在 uni-agent 侧把已部署的 QwenPaw 服务接成 Agent Runner 的工程人员。
> 结论来源：QwenPaw `2.3.1b1` 源码、uni-agent `uni_agent/agents/qwenpaw/` 实现，以及本机历史实验记录。

---

## 1. 组件与版本

| 组件 | 已验证版本 | 说明 |
| --- | --- | --- |
| QwenPaw | `2.3.1b1` | 本仓库 `QwenPaw/src/qwenpaw/__version__.py`；历史 4P 实验使用 `2.3.0`，协议一致 |
| QwenPaw Python | `3.11` | `QwenPaw/.python-version`；部署要求 `>=3.11,<3.14` |
| QwenPaw 前端构建 | Node `22` | 仅源码方式部署且需要 Web Console 时需要；API 对接本身不依赖 |
| uni-agent | 当前工作区 `98f37f3` | Runner 契约遵循 RFC #5790 |
| verl | `0.9.0.dev`（`78bba31d`） | uni-agent/NPU 训练侧要求的版本 |
| vLLM | OpenAI 兼容服务 | 必须开启自动工具调用：`--enable-auto-tool-choice` + `--tool-call-parser` |
| 工具解析器 | `qwen3_xml` / `qwen3_coder` | Qwen3.5 系列用 `qwen3_xml`，Qwen3-Coder 用 `qwen3_coder` |

**版本相关变更**

- QwenPaw `2.3.x` 的 provider 路由前缀是 `/models`；更早版本（如 `2.2.x`）是 `/providers`。
- QwenPaw `2.3.x` 没有对外的 `/v1/chat/completions` 入口。uni-agent 走的是「provider 重配 + 后台任务」的现有 REST API。
- 所有运行时接口都挂载在 `/api` 前缀下（见 §8）。

---

## 2. 链路与协议分层

```
verl trainer
  └─ uni-agent Framework
       ├─ Gateway.create_session()
       │    → base_url        = http://<gateway>/sessions/<session_id>/v1
       │    → reward_info_url = http://<gateway>/sessions/<session_id>/reward_info
       │
       └─ run_qwenpaw(session=SessionHandle, raw_prompt=..., tools_kwargs=...)
            │
            │ ① PUT  {qwenpaw_api_url}/models/{provider_id}/config
            │      {base_url: session.base_url, auto_discover: false}
            │
            │ ② POST {qwenpaw_api_url}/console/chat/task
            │      {channel, user_id, session_id, input[], timeout}
            │      ← {task_id, timeout}
            │
            │ ③ GET  {qwenpaw_api_url}/console/chat/task/{task_id}
            │      ← {status, result}
            │
            │ ④ POST {session.reward_info_url}      （report_reward=true 时）
            │      {"reward_info": {"reward", "finished", "extra_info"}}
            │
            └─ TaskResult(reward, finished, extra_info)

QwenPaw ReAct 循环内部：
  POST {provider.base_url}/chat/completions
    = POST http://<gateway>/sessions/<session_id>/v1/chat/completions
    → Gateway 记录 token 级轨迹 → vLLM → 返回 OpenAI 兼容响应
```

三条协议边界：

| 边界 | 协议 | 方向 |
| --- | --- | --- |
| uni-agent Runner ↔ QwenPaw | QwenPaw REST（JSON） | 双向 |
| QwenPaw ↔ Gateway | OpenAI 兼容 `/v1/chat/completions` | QwenPaw 发起 |
| uni-agent Runner → Gateway | 会话奖励回传 HTTP POST | Runner 发起 |

关键点：**QwenPaw 不返回 logprob**。token 级 logprob 由 Gateway 在转发 LLM 调用时记录，QwenPaw 只返回最终文本。

---

## 3. QwenPaw 侧前置状态

对接前 QwenPaw 必须满足以下状态。这些是配置前提，不是部署步骤。

### 3.1 Provider

| 项 | 要求 | 原因 |
| --- | --- | --- |
| `provider_id` | 必须是**自定义 provider** | 内置 provider（`openai`/`modelscope`/`dashscope`）`freeze_url=true` |
| `freeze_url` | 必须为 `false` | `freeze_url=true` 时 `PUT /config` 返回 200，但 `base_url` 被**静默忽略** |
| `enabled` | `true` | 关闭状态下 chat 会被拒绝 |
| `chat_model` | `OpenAIChatModel` | 走 OpenAI 兼容适配器，才会把 `base_url` 拼成 `/chat/completions` |
| `auto_discover` | 重配时固定 `false` | Gateway 没有可用的 `/models` 目录，避免后台探测任务失败 |

创建自定义 provider（QwenPaw 2.3.x）：

```bash
curl -s -X POST "http://<qwenpaw-host>:8088/api/models/custom-providers" \
  -H "Content-Type: application/json" \
  -H "X-Agent-Id: default" \
  -d '{
    "id": "opd-gateway",
    "name": "OPD Gateway",
    "default_base_url": "http://127.0.0.1:8000/v1",
    "chat_model": "OpenAIChatModel",
    "models": [{"id": "any-model-name", "name": "any-model-name"}]
  }'
```

字段参考（`CreateCustomProviderRequest`）：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | provider id，训练脚本的 `provider_id` 必须与它一致 |
| `name` | 是 | 展示名 |
| `default_base_url` | 否 | 初始 `base_url`；训练时会被 Runner 覆盖 |
| `api_key` / `api_key_prefix` | 否 | 上游模型鉴权；Gateway 场景一般留空 |
| `chat_model` | 否 | 默认 `OpenAIChatModel` |
| `models` | 否 | 模型列表；模型名不参与 Gateway 校验，可任意填写 |

> 该 provider 在训练期间会被重指向 Gateway，**不能同时用于生产会话**。

### 3.2 Agent

| 项 | 要求 |
| --- | --- |
| `agent_id` | 通过请求头 `X-Agent-Id` 解析，默认 `default` |
| 存在性 | 必须存在于 `GET /api/agents` |
| 状态 | `enabled=true` 且 `available_in_chat != false` |
| `active_model` | 未设置会告警；建议指向训练 provider + 任意模型名 |

### 3.3 网络

- uni-agent 主机 → QwenPaw `:8088` 可达。
- **QwenPaw 主机 → 训练机 Gateway 的内网 IP:port 可达**。这条是轨迹采集成立的前提。
- QwenPaw 默认监听 `127.0.0.1:8088`；跨机时必须改成 `0.0.0.0` 或可被访问的地址，并视情况开启鉴权（`Authorization: Bearer <token>`）。

---

## 4. 协议 A：uni-agent → QwenPaw REST

### 4.1 通用约定

| 项 | 值 |
| --- | --- |
| Base URL | `http(s)://<qwenpaw-host>:<port>/api` |
| Content-Type | `application/json` |
| 鉴权 | `Authorization: Bearer <token>`（仅开启访问控制时） |
| Agent 选择 | `X-Agent-Id: <agent_id>` |
| 成功状态码 | `200` / `201`；其余由客户端抛 `RuntimeError` |

### 4.2 接口清单

| 方法 | 路径（相对 `/api`） | 作用 |
| --- | --- | --- |
| GET | `/healthz` | 就绪探测；未就绪返回 503 |
| GET | `/agents` | 列出 agent，校验 `X-Agent-Id` |
| GET | `/models` | 列出 provider，校验 `freeze_url` / `enabled` |
| POST | `/models/custom-providers` | 创建自定义 provider |
| PUT | `/models/{provider_id}/config` | 重配 provider（训练时指向 Gateway session） |
| POST | `/console/chat/task` | 启动一次后台 ReAct 回合 |
| GET | `/console/chat/task/{task_id}` | 轮询任务状态 / 取结果 |

### 4.3 Provider 重配

请求：

```http
PUT /api/models/opd-gateway/config
Content-Type: application/json
X-Agent-Id: default

{
  "base_url": "http://<gateway-host>:<port>/sessions/<session_id>/v1",
  "auto_discover": false
}
```

可选字段（`ProviderConfigRequest`）：

| 字段 | 说明 |
| --- | --- |
| `enabled` | 是否启用 |
| `api_key` | 上游模型密钥 |
| `name` | 仅自定义 provider 可改显示名 |
| `chat_model` | 自定义 provider 只接受受支持的协议名 |
| `generate_kwargs` | 透传给生成调用的参数（如 `temperature`、`max_tokens`） |
| `custom_headers` | 每次上游请求附带的 HTTP 头 |
| `auth_mode` | `api_key` / `auth_token`，仅 Anthropic 兼容协议 |
| `auto_discover` | 是否触发模型发现；对接 Gateway 必须 `false` |

响应：`ProviderInfo`（含 `id`、`base_url`、`is_custom`、`freeze_url`、`enabled` 等）。

> `200` 不代表重配生效。内置 provider 会静默忽略 `base_url`。验收方式是再 `GET /api/models` 检查实际 `base_url`。

### 4.4 启动后台 Chat 任务

请求：

```http
POST /api/console/chat/task
Content-Type: application/json
X-Agent-Id: default

{
  "channel": "console",
  "user_id": "default",
  "session_id": "uniagent-3f2a9c1b7d40",
  "input": [
    {
      "role": "user",
      "type": "message",
      "content": [
        {"type": "text", "text": "What is 2+2?", "status": "created"}
      ]
    }
  ],
  "timeout": 600
}
```

响应：

```json
{"task_id": "task-4f1c2a9b8e70", "timeout": 600}
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `channel` | 是 | 固定 `console` |
| `user_id` | 是 | 会话归属用户，Runner 固定 `default` |
| `session_id` | 是 | QwenPaw 会话键；Runner 未传时生成 `uniagent-<uuid12>` |
| `input` | 是 | 消息数组，元素含 `role` / `type` / `content[]` |
| `content[].type` | 是 | `text` |
| `content[].text` | 是 | 用户消息正文 |
| `content[].status` | 是 | 固定 `created` |
| `timeout` | 否 | 服务端执行预算（秒）；缺省用服务端默认值，非法值返回 400 |
| `model_slot_override` | 否 | 会话级模型覆盖，格式 `{"provider_id": "...", "model": "..."}` |

`session_id` 语义：QwenPaw 用它恢复会话历史。**评测/训练每个 episode 必须使用全新 session_id**，复用会导致历史轨迹泄漏。

### 4.5 轮询任务

```http
GET /api/console/chat/task/task-4f1c2a9b8e70
```

运行中：

```json
{"status": "running", "started_at": 1794534650.123}
```

完成：

```json
{
  "status": "finished",
  "started_at": 1794534650.123,
  "result": {
    "output": "The answer is 4. #### 4"
  }
}
```

状态语义：

| status | 客户端行为 |
| --- | --- |
| `running` 或缺失 | 继续轮询 |
| `finished` | 取 `result`，递归抽取最终文本 |
| 其他 | 视为错误 |

最终文本抽取规则：从 `result` 中依次找 `output` → `text` → `content[{type:text}]`，递归取第一个非空字符串。

### 4.6 客户端超时

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `request_timeout` | 600s | 单次 HTTP 调用总超时 |
| `connect_timeout` | 30s | 建连超时 |
| `task_timeout` | 600s | 服务端任务预算，同时作为客户端轮询 deadline 基准 |
| `poll_interval` | 2s | 轮询间隔 |
| 客户端 deadline | `task_timeout + 30s` | 留缓冲等服务端先取消 |

---

## 5. 协议 B：QwenPaw → Gateway（OpenAI 兼容）

### 5.1 URL 拼接规则

`PUT /config` 写入的 `base_url` 会被 QwenPaw 的 OpenAI 适配器原样使用，并追加 `/chat/completions`：

```
provider.base_url = http://<gateway>/sessions/<session_id>/v1
实际请求          = POST http://<gateway>/sessions/<session_id>/v1/chat/completions
```

这正是 Gateway 的路由，因此方案成立。

### 5.2 请求

```http
POST /sessions/<session_id>/v1/chat/completions
Content-Type: application/json

{
  "model": "any-model-name",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "execute_shell_command",
        "description": "...",
        "parameters": {"type": "object", "properties": {...}}
      }
    }
  ],
  "stream": true,
  "temperature": 0.6,
  "max_tokens": 32768
}
```

说明：

- `model` 不参与 Gateway 路由校验，可任意。
- `tools` 由 QwenPaw 侧工具配置生成，uni-agent 不干预。
- `stream` 由 QwenPaw 适配器决定；Gateway 支持流式和非流式。
- 采样参数来自 provider 的 `generate_kwargs`；**未显式设置 `temperature` 时可能落到 vLLM 默认值 1.0**，评测场景必须在 provider 配置里固定 `temperature: 0`。

### 5.3 响应与轨迹

Gateway 返回标准 OpenAI 响应，并在内部记录：

- 每轮请求的 token 序列
- 生成 token 的 logprob
- 工具调用与最终文本

QwenPaw 只消费 `choices[].message.content`（或流式增量），不感知也不返回 logprob。

### 5.4 Gateway 会话端点

| 端点 | 作用 |
| --- | --- |
| `POST /sessions/{id}/v1/chat/completions` | OpenAI 兼容聊天 |
| `POST /sessions/{id}/v1/messages` | Anthropic 兼容消息 |
| `POST /sessions/{id}/reward_info` | Runner 回传奖励 |

`SessionHandle.base_url` 与 `reward_info_url` 由 Gateway 在 `create_session()` 时注入，Runner 无需自行拼接。

---

## 6. 协议 C：奖励回传

`report_reward=true` 时，Runner 在回合结束后调用：

```http
POST /sessions/<session_id>/reward_info
Content-Type: application/json

{
  "reward_info": {
    "reward": 1.0,
    "finished": true,
    "extra_info": {
      "qwenpaw_session_id": "uniagent-3f2a9c1b7d40",
      "agent_id": "default",
      "qwenpaw_api_url": "http://<qwenpaw-host>:8088/api",
      "provider_id": "opd-gateway",
      "content": "The answer is 4. #### 4",
      "ground_truth": "4"
    }
  }
}
```

框架在 finalize 时**优先采用 Runner 上报的 reward**；上报失败只告警，不回滚回合。

奖励计算位置在 Runner 进程内：

| 条件 | 行为 |
| --- | --- |
| 回合完成且有 `ground_truth`，`answer_type="em"` | `ExactMatchReward` |
| 回合完成且有 `ground_truth`，其他 | `GSM8KReward`（匹配 `#### <number>`，兜底取最后一个数字） |
| 回合完成但无 `ground_truth` | `1.0`（仅冒烟） |
| 回合未完成 | `0.0` |

---

## 7. uni-agent 配置参考

### 7.1 Runner YAML

```yaml
agent_runners:
  qwenpaw:
    runner_fqn: uni_agent.agents.qwenpaw.agent.run_qwenpaw
    runner_kwargs:
      qwenpaw_api_url: "http://10.0.0.5:8088/api"   # 注意 /api 后缀
      provider_id: "opd-gateway"                    # 必须是自定义 provider
      agent_id: "default"
      auth_token: null
      request_timeout: 600.0
      connect_timeout: 30.0
      task_timeout: 600.0
      poll_interval: 2.0
      report_reward: true
    dispatch_mode: inline_async
    max_concurrent_sessions: 1
    trajectory_selection: longest
```

### 7.2 参数表

| 参数 | 位置 | 默认 | 必填/约束 |
| --- | --- | --- | --- |
| `runner_fqn` | runner | — | `uni_agent.agents.qwenpaw.agent.run_qwenpaw` |
| `qwenpaw_api_url` | `runner_kwargs` | `http://localhost:8088` | 实际必须带 `/api`（见 §8） |
| `provider_id` | `runner_kwargs` | `""` | 必填；必须是自定义 provider |
| `agent_id` | `runner_kwargs` | `default` | 必须存在且可用 |
| `auth_token` | `runner_kwargs` | `null` | 开启访问控制时填 Bearer token |
| `request_timeout` | `runner_kwargs` | `600.0` | 秒 |
| `connect_timeout` | `runner_kwargs` | `30.0` | 秒 |
| `task_timeout` | `runner_kwargs` | `600.0` | 秒；长轨迹需调大 |
| `poll_interval` | `runner_kwargs` | `2.0` | 秒 |
| `report_reward` | `runner_kwargs` | `false` | 训练建议 `true` |
| `reward_fn` | `runner_kwargs` | `null` | 覆盖默认奖励函数 |
| `dispatch_mode` | runner | — | 固定 `inline_async` |
| `max_concurrent_sessions` | runner | — | 固定 `1` |
| `trajectory_selection` | runner | — | 建议 `longest` |

### 7.3 并发约束

`provider.base_url` 是 QwenPaw 实例级全局状态（`ProviderManager` 进程级单例）。同一实例上并发 episode 会互相覆盖端点，导致轨迹串号。因此：

- 当前 Runner 复用同一个 `provider_id`，训练侧必须 `max_concurrent_sessions: 1`。
- 评测侧可以启动多个独立 QwenPaw 实例按 shard 分题并行，不受此约束。
- QwenPaw 原生支持会话级模型隔离：为每个 session 创建独立自定义 provider，并在 `POST /console/chat/task` 里带 `model_slot_override: {"provider_id": "...", "model": "..."}`。**当前 Runner 未使用该能力**，属于可选的后续优化方向。

### 7.4 其它硬约束

| 约束 | 后果 |
| --- | --- |
| `provider_id` 为空 | Runner 直接抛 `ValueError` |
| `session.base_url` 为空 | Runner 直接抛 `ValueError` |
| provider 是内置 provider | `PUT config` 静默无效，轨迹仍打到原端点 |
| `auto_discover` 未关 | QwenPaw 后台探测 Gateway `/models` 失败 |
| QwenPaw 访问不到 Gateway | ReAct 回合无法发起 LLM 调用，任务超时 |
| 复用 `session_id` | 历史轨迹泄漏进新 episode |

### 7.5 输入格式

`run_qwenpaw` 接收框架注入的 `raw_prompt` 和 `tools_kwargs`。`raw_prompt` 支持三种形态。

形态 1，纯字符串：

```json
"What is 2+2?"
```

形态 2，OpenAI messages 列表：

```json
[
  {"role": "system", "content": "You must answer with #### <number>."},
  {"role": "user", "content": "What is 2+2?"}
]
```

形态 3，dict：

```json
{"prompt": "What is 2+2?", "ground_truth": "4"}
```

抽取规则（`_extract_user_message`）：

- 字符串：原样使用。
- messages 列表：把 `system` 和 `user` 的文本块按出现顺序用 `\n\n` 拼接成**一条 user 消息**。
- dict：依次取 `prompt` → `text` → `content`。
- 其他类型：抛 `ValueError`。

注意：形态 2 里的 system 消息会被合并进 user 文本，QwenPaw 侧不会再保留独立 system 消息。这是与白盒 ReAct 评测 harness 的已知差异，判读准确率时需标注。

训练数据行结构（parquet，`prepare_qwenpaw_data.py` / `convert_gsm8k_data.py` 生成）：

```json
{
  "prompt": [
    {"role": "system", "content": "Solve the problem. End with #### <number>."},
    {"role": "user", "content": "Natalia sold clips to 48 friends in April, ..."}
  ],
  "extra_info": {
    "tools_kwargs": {
      "task": {
        "name": "qwenpaw",
        "metadata": {
          "ground_truth": "72",
          "answer_type": "gsm8k"
        }
      }
    }
  }
}
```

字段语义：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `prompt` | 是 | 字符串 / messages 列表 / dict，见上 |
| `extra_info.tools_kwargs.task.name` | 建议 | 任务名，`qwenpaw` |
| `...task.metadata.ground_truth` | 否 | 标准答案；缺失时完成即给 1.0（仅冒烟） |
| `...task.metadata.answer_type` | 否 | `"em"` 触发精确匹配，其余走 GSM8K 数字匹配 |

`ground_truth` 抽取优先级：`tools_kwargs.task.metadata.ground_truth` → `raw_prompt.ground_truth`。

`tools_kwargs` 不会被传给 QwenPaw，也不会触发 uni-agent 本地 Toolbox/Sandbox。工具集完全由 QwenPaw 服务侧配置决定。

---

## 8. 已知问题：`/api` 前缀不一致

### 8.1 事实

QwenPaw `2.3.x` 在 `app/_app.py` 里统一挂载：

```python
app.include_router(api_router, prefix="/api")
```

provider router 自身前缀是 `/models`，console chat 任务路由是 `/console/chat/task`。因此真实路径是：

```
/api/models/{provider_id}/config
/api/console/chat/task
/api/console/chat/task/{task_id}
/api/healthz
/api/agents
```

而 uni-agent 侧 `QwenPawClient._request()` 的拼接方式是：

```python
url = f"{self._cfg.qwenpaw_api_url.rstrip('/')}/{path.lstrip('/')}"
# path 分别是 models/{id}/config、console/chat/task、console/chat/task/{task_id}
```

它**不会自动补 `/api`**。所以 `qwenpaw_api_url` 必须显式写成 `http://<host>:8088/api`。

### 8.2 现存不一致点

| 位置 | 现状 | 与 2.3.x 实际路由的关系 |
| --- | --- | --- |
| `QwenPawConfig.qwenpaw_api_url` 默认值 | `http://localhost:8088` | 缺 `/api` |
| `agent.py` 文档字符串示例 | `http://10.0.0.5:8088` | 缺 `/api` |
| `run_qwenpaw_npu_megatron_vllm.sh` | `QWENPAW_API_URL` 默认 `http://localhost:8088` | 缺 `/api` |
| `qwenpaw_demo_runbook.md` | 示例均不带 `/api` | 缺 `/api` |
| `check_qwenpaw_connectivity.py` | healthz 用 `api/healthz`，models/chat 用 `models`、`console/chat/task` | 同一条 base 上混用两种口径 |

### 8.3 联检脚本为什么必然失败

`check_qwenpaw_connectivity.py` 用同一个 `--base-url` 拼所有路径：

- healthz：`{base}/api/healthz`
- agents：`{base}/agents`
- models：`{base}/models`
- chat task：`{base}/console/chat/task`

两种取值都会有一组检查失败：

| `--base-url` | healthz | models | 结果 |
| --- | --- | --- | --- |
| `http://host:8088` | `/api/healthz` ✅ | `/models` ❌ 404 | provider 检查失败 |
| `http://host:8088/api` | `/api/api/healthz` ❌ 404 | `/api/models` ✅ | health 检查失败 |

因此该脚本对 QwenPaw `2.3.x` 无法全绿，不能作为「QwenPaw 侧未就绪」的判据。

### 8.4 影响

1. **训练直接失败在第一个 episode 之前**。`configure_provider` 返回 404，`run_qwenpaw` 捕获异常并返回 `finish_reason="error"`，日志出现：
   `QwenPaw API PUT models/<id>/config returned 404: {"detail":"Not Found"}`。
2. **联检产生假阴性**，容易误判为 provider 配置错误、QwenPaw 未启动或网络不通。
3. **不影响 QwenPaw 与 Gateway 的协议本身**。只要 `qwenpaw_api_url` 带 `/api`，§4、§5 的链路即成立。
4. **历史实验能在远程跑通，说明运行时实际传入的地址与 2.3.x 路由匹配**（通过环境变量或脚本覆盖）；代码里的默认值和文档示例不能直接照抄。

### 8.5 使用建议（本次不改代码）

- 所有 YAML / 环境变量显式写 `qwenpaw_api_url: "http://<host>:8088/api"`。
- 手工联检时按 `/api` 口径逐条验证，不要依赖联检脚本的默认路径组合。
- 若后续要修，最小改法是统一为「base 带 `/api`」，联检脚本里的 `api/healthz` 改为 `healthz`；或在 `QwenPawClient` 内做一次前缀归一化。

---

## 9. 常见错误判读

| 现象 | 根因 | 定位 |
| --- | --- | --- |
| `PUT models/<id>/config returned 404` | `qwenpaw_api_url` 缺 `/api` | §8 |
| `provider '<id>' not found` | provider 未创建，或 id 拼错 | `GET /api/models` |
| `PUT config` 返回 200 但 `base_url` 不变 | provider `freeze_url=true`（内置 provider） | `GET /api/models` 看 `freeze_url` |
| `GET /api/healthz` 返回 503 | QwenPaw 仍在启动，`startup_ready` 未置位 | 等待或检查启动日志 |
| `agent '<id>' not found` / `available_in_chat=false` | `X-Agent-Id` 不对或 agent 被禁用 | `GET /api/agents` |
| chat task 一直 `running` 直到超时 | QwenPaw 访问不到 Gateway，或上游 vLLM 卡住 | 查 QwenPaw 日志与 Gateway 访问日志 |
| 400 `auto tool choice requires ...` | vLLM 未开自动工具调用 | 加 `--enable-auto-tool-choice` 与正确的 `--tool-call-parser` |
| 输出格式大量错误、准确率异常偏低 | provider 未固定 `temperature`，或 system prompt 被 QwenPaw 改写 | 固定 `generate_kwargs.temperature`，核对 harness 差异 |
| 轨迹出现重复长片段 | QwenPaw 流式超时后重试未回滚 | 训练前做重复扫描，或调大流超时 |
