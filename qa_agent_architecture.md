# NeMo-RL QA Agent 多轮检索问答架构设计方案

本文档面向 NeMo-RL 0.6.0 下的 QA-RL 考试任务：使用 GRPO 训练 `Qwen/Qwen3.5-9B-Base`，让模型在回答技术培训题前可以多轮检索 `/data/docs` 中的内部资料，并在最终答案中输出 `\boxed{...}`。

考试 PDF 中的硬约束以如下内容为准：

- 官方仓库：`https://github.com/wccdev/nemo-rl-lab-exam`。
- 模型：`Qwen/Qwen3.5-9B-Base`，LoRA 微调。
- 硬件：每人 1 张 H200 141GB GPU，48 小时。以考试 PDF 为权威口径；仓库中 H100 相关说明只作为历史示例参考。
- 数据：`/data/datasets/qa_rl`，不得修改。
- 文档：`/data/docs`，作为检索资料来源。
- 评分：最佳 `validation/accuracy`。
- 交付：Fork 地址、最佳作业 ID、accuracy 截图、简要说明。

本方案的核心目标不是单纯“让模型会搜索”，而是让模型逐步学会：

1. 稳定遵循 `<search>...</search>` 与 `\boxed{...}` 协议。
2. 在需要时检索，并从检索结果中抽取有效证据。
3. 最终最大化验证集 `accuracy`，避免为了过程奖励而进行无意义检索。

由于正式考试前无法实际提交训练，本方案按“预案优先”组织：先把接口、判分、检索和调参分支设计到足够具体，明天拿到真实集群与数据后优先做最小闭环，再逐步打开复杂策略。

---

## 0. 开考前离线准备与开考后优先级

正式训练前不能访问 `/data/docs` 和完整 `/data/datasets/qa_rl` 时，仍然可以提前完成以下工作：

1. 写好 `Action Parser`、答案标准化、query 清洗、检索切段、reward 分解的纯 Python 单元测试。
2. 用 `datasets/qa_rl/examples.jsonl` 和 3 到 5 个 fake markdown 文档模拟 `/data/docs`，验证多轮状态机。
3. 准备 H200 的保守版与加速版两套配置片段，考试当天根据 OOM、吞吐和验证周期选择。
4. 准备首个最小可跑版本：无 `think`、无复杂课程切换、BM25/char n-gram 检索、Phase A/B 固定 reward。
5. 准备观察表：第一次作业只看是否能启动、是否能生成 `<search>`、是否能回灌检索结果、是否能输出 boxed。

开考后建议按下面顺序推进：

| 时间窗口 | 目标 | 判断标准 |
| --- | --- | --- |
| 第 0-2 小时 | 跑通最小闭环 | 作业启动、验证样本出现 search/result/boxed |
| 第 2-6 小时 | 修解析和检索 | `valid_search_rate`、`boxed_format_rate` 可用 |
| 第 6-12 小时 | 调 reward 和长度 | 训练 reward 与验证样本同步改善 |
| 第 12-24 小时 | 做题型拆分优化 | 客观题、填空、简答分别看错误 |
| 第 24-48 小时 | 固化最佳配置 | 少改代码，多提交稳态调参 run |

---

## 1. 总体架构

系统由六个模块组成：

| 模块 | 职责 |
| --- | --- |
| Prompt Builder | 构造系统提示词、题面、Few-shot 工具调用示例 |
| QA Dataset Loader | 读取 `/data/datasets/qa_rl/{train,val}.jsonl`，保留题型与标准答案 |
| QA Environment | NeMo-RL 多轮 Environment，负责状态机、工具反馈与奖励 |
| Action Parser | 解析 `<search>` 与 `\boxed{}`，处理不规范输出 |
| Search Engine | 对 `/data/docs` 建索引，执行检索、去重、截断和缓存 |
| Reward & Metrics | 课程式奖励、答案标准化、验证指标与调试样本输出 |

推荐目录组织。考试手册给出的创建命令是 `lab new grpo_qwen3.5-9b_qa-rl-agent_<你的名字> --from agent-grpo_qwen3.5-9b_sliding-puzzle_v1`，因此实验目录建议沿用该命名，避免提交说明和作业记录不一致。

```text
experiments/grpo_qwen3.5-9b_qa-rl-agent_<name>/
  config.yaml
  run.py

common/environments/
  qa_env.py
  search_utils.py
```

`run.py` 是实验入口。仓库的 `scripts/_run_experiment.sh` 会优先使用实验目录下的 `run.py`，因此不需要额外引入 `run_qa.py` 这种非标准入口。

### 1.1 `run.py` 必须承担的真实职责

多轮 QA Agent 不能只靠 `config.yaml` 里的 `ResponseDataset` 和 `env_name` 自动接入。`ResponseDataset` 适合单轮 GRPO；多轮工具调用必须自己构造 NeMo-RL 的 `DatumSpec`，并把自定义环境 actor 显式传入 `grpo_train`。

`run.py` 的最小职责：

| 步骤 | 实现要点 |
| --- | --- |
| 读取数据 | 从 `${QA_RL_DATA_DIR}/train.jsonl` 与 `val.jsonl` 读取 `query`、`expected_answer` |
| 构造 prompt | 将 system prompt、少量 few-shot、原始 `query` 拼成 user/chat message |
| 构造 `DatumSpec` | `message_log` 放 tokenized prompt；`extra_env_info` 放题目、标准答案、题型、样本 id、最大轮数 |
| 设置任务名 | `task_name = "qa_agent"`，必须与 `task_to_env` 和 `config.env.qa_agent` 对齐 |
| 设置停止符 | 建议至少包含 `</search>`；若启用答案标签，也可加 `}` 但需谨慎，避免截断正文 |
| 实例化环境 | `QAEnvironment.options(num_gpus=0).remote(cfg=dict(env_cfg))` |
| 启动训练 | 调用 `setup(...)` 后执行 `grpo_train(..., task_to_env, task_to_env, ...)` |

`extra_env_info` 建议结构：

```python
{
    "sample_id": idx,
    "query": raw_query,
    "expected_answer": expected_answer,
    "answer_type": answer_type,        # single / multiple / bool / fill / short
    "gold_answer": gold_answer,        # 去掉 [type] 前缀后的主体
    "num_turns": 0,
    "max_turns": env_cfg.get("max_turns", 6),
    "search_history": [],
}
```

注意两点：

- 不要把 `gold_answer`、reward、判分原因放入模型可见消息；这些只存在于 `extra_env_info` 和日志里。
- 验证集也需要走同一套环境和解析逻辑；否则训练 reward 与 `validation/accuracy` 会漂移。

---

## 2. Prompt 设计

Qwen Base 模型前期最难的是稳定遵循结构化协议，因此 Prompt 必须同时完成两件事：

1. 明确工具调用和最终答案格式。
2. 用短 Few-shot 教会模型“先搜、读结果、再 boxed”的节奏。

### 2.1 System Prompt 模板

```text
你是一个可以查阅内部技术文档的问答助手。请严格遵守以下协议：

1. 如果需要查询资料，只输出检索标签：
   <search>关键词</search>

2. 检索词可以是自然短语，也可以是多个关键词；多个检索意图建议用分号分隔：
   <search>自然短语</search>
   <search>关键词1;关键词2</search>

3. 系统返回检索结果后，你可以继续搜索，也可以作答。

4. 当你确定最终答案后，必须输出：
   \boxed{答案}

5. 一旦输出 \boxed{}，本题立即结束。不要在 \boxed{} 后继续输出内容。

6. 选择题只填选项字母；多选题按字母顺序用英文逗号分隔；填空和简答用分号分隔要点。
```

### 2.2 Few-shot 示例

Few-shot 只用于教授协议，不应直接泄漏训练集或验证集答案。示例应短，避免大量占用上下文。

```text
示例一：
题目：示例设备通过什么系统与示例区域连接？
助手：<search>示例设备与示例区域连接系统</search>
系统：
### 检索结果返回：
资料显示：示例设备通过 Alpha Link 系统与示例区域连接。
### 请基于上述资料继续作答：
助手：资料说明连接方式是 Alpha Link。 \boxed{Alpha Link}

示例二：
题目：示例流程必须先完成记录归档。A. 对 B. 错
助手：<search>示例流程;记录归档要求</search>
系统：
### 检索结果返回：
资料显示：示例流程开始前必须先完成记录归档。
### 请基于上述资料继续作答：
助手：资料明确说明该说法正确。 \boxed{A}
```

### 2.3 环境反馈格式

检索结果需要有边界，但不要堆太多结构化字段。建议只保留一个简短标题和自然语言资料内容。

```text

【检索结果】
{retrieved_context}

请根据这些资料继续思考；如果还不确定，可以换一个更具体的关键词继续检索。
```

格式错误反馈：

```text

上一轮格式不完整，我没有执行检索。
要查资料，请输出完整的 <search>关键词</search>；确定答案后，请输出 \boxed{答案}。
```

### 2.4 草稿与工具调用边界

不建议引入 `<think>` 这类额外标签。它会增加一个新格式，让 Base 模型先学标签而不是学任务本身。

更稳的规则是：

- 允许模型在 `<search>` 前写一句很短的自然语言草稿，用来组织检索词。
- 必须输出完整 `<search>关键词</search>` 才会执行检索。
- `</search>` 后不要继续预测检索结果或推演答案；这一轮应由 stop string 截断，交给环境返回资料。
- 纯草稿但没有 `<search>` 或 `\boxed{}`，按无有效动作处理。

示例：

```text
需要查流程开始前的归档要求。<search>示例流程;记录归档要求</search>
```

环境解析策略：

- 提取合法 `<search>...</search>`，忽略 search 前的短自然语言草稿。
- 若 `</search>` 后仍有可见文本，训练时轻微扣分，因为这通常是在脑补检索结果。
- 如果同轮同时出现 `\boxed{}`，按 boxed 优先结算；但 stop string 应尽量防止 search 后继续生成 boxed。

---

## 3. 环境状态机

环境 `step()` 必须是严格优先级状态机，避免同一轮既搜索又结算、或者把合法搜索误判成格式错误。

```text
Assistant 输出
  |
  |-- 1. 若存在 \boxed{...}
  |       取最后一个 boxed 作为最终答案
  |       忽略同轮 search
  |       标准化答案并判分
  |       done=True
  |
  |-- 2. 否则若存在合法 <search>...</search>
  |       提取并清洗 query
  |       最多保留前 K 个 query
  |       检索并返回环境消息
  |       done=False
  |
  |-- 3. 否则若存在未闭合 search、空 search 或无有效动作
  |       返回格式纠错提示
  |       done=False
  |
  |-- 4. 若达到 max_rollout_turns
          按未完成作答处理
          done=True
```

关键规则：

- `boxed` 优先级最高。只要模型输出了 `\boxed{}`，本题立即结算。
- 多个 `boxed` 时取最后一个，兼容 Base 模型自我修正。
- 合法 `search` 只有在没有 `boxed` 时才触发工具调用。
- 允许 search 前有短自然语言草稿；search 后仍继续写推演文本时训练轻微扣分。
- 每轮最多执行 `K=3` 个 query，单个 query 建议限制在 32 到 64 个字符。
- 检索词可以是自然短语；如果模型输出多个检索意图，推荐用英文分号 `;` 分隔。环境解析时要兼容中文分号、逗号、顿号、换行和多余空白。
- 重复 query 不重复检索，直接返回缓存或提示“该关键词已检索过”。
- `max_rollout_turns` 推荐 6；如果 Prompt/Few-shot 较长，先用 4 到 5 控制显存。

---

## 4. 检索与上下文控制

### 4.1 文档索引

`Search Engine` 在 Environment actor 初始化时扫描 `/data/docs`，不要每个 step 重新读盘。

推荐策略：

- 递归读取 Markdown、TXT 等文本文件。
- 先按双换行切段；段落过长时按固定 token 或字符窗口二次切分。
- 对每个 chunk 保存 `doc_path`、`heading`、`chunk_id`、`text`。
- 建立 BM25 或轻量倒排索引，同时保留 substring fallback。
- 查询结果做 LRU cache，避免相同 query 在同一批 rollout 中反复检索。

中文培训题通常是中文题干夹英文术语，不能只靠空格分词。推荐索引时同时保留四类特征：

| 特征 | 用途 |
| --- | --- |
| 原文 substring | 精确命中设备名、系统名、英文缩写 |
| 英文/数字 token | 命中 `SQL server`、`OFD`、`Sample`、`AMU` 等术语 |
| 中文 2-4gram | 解决中文无空格导致 BM25 漏召回 |
| heading/path token | 利用 markdown 标题、文件名提高主题相关性 |

轻量实现不必依赖重型向量库。首版可以使用 `rank_bm25` 或自写倒排分数；如果集群镜像里没有额外包，就用标准库实现：

1. 对每个 chunk 生成 feature list：英文 token + 中文 2gram/3gram/4gram + 标题 token。
2. 记录 `df` 和 chunk 内 `tf`，用简化 BM25 或 TF-IDF 打分。
3. substring 命中额外加分，标题命中再加分。
4. 同一文档连续 chunk 去重，避免 top-k 全来自同一段附近。

### 4.2 Query 标准化与鲁棒解析

Prompt 不应把检索过程限制成机械关键词列表。模型可以输出自然短语，例如 `<search>控制图分析 永久排除 Sample</search>`；也可以在确实有多个检索意图时使用分号，例如 `<search>控制图分析;永久排除 Sample;Exclude</search>`。检索层负责把这些输入转成稳健召回，而不是要求模型完全按某种分词方式思考。

解析策略：

- 先提取所有 `<search>...</search>` 内容。
- 对每段 `<search>` 内容，先保留原始自然短语作为 primary query。
- 如果内容里有英文分号 `;` 或中文分号 `；`，再切出多个 intent query。
- 如果没有分号，但有逗号、顿号、换行，可作为弱分隔符切出补充 query。
- 不要强行按空格拆碎英文短语；空格更多是短语内部结构。
- 可从长 query 中抽取 1 到 2 个较长名词片段或英文缩写作为 fallback query。
- 合并 primary query、intent query、fallback query，去重后最多保留前 `K=3` 个检索意图。

召回策略：

- 第一通道：用原始自然短语做 substring 召回，优先保留完整短语语义。
- 第二通道：用原始自然短语的中文 n-gram、英文 token 做 BM25/TF-IDF 召回。
- 第三通道：用切分出的关键词、设备名或英文缩写做补充召回，提高漏召回容错。
- 合并多路结果，按分数、标题命中、来源多样性和重复度重排。
- 奖励不评价“是否使用分号”，只评价检索是否有效、最终答案是否正确。

推荐的重排加权：

```text
score = bm25_score
      + 1.5 * exact_phrase_hit
      + 0.8 * heading_hit
      + 0.5 * acronym_hit
      - 0.3 * duplicate_doc_penalty
```

这不是最终算法约束，只是首版可解释、易调参的保守起点。

对每个 query 做轻量清洗：

- 去除 XML/Markdown 控制符。
- 中文逗号、顿号、分号统一处理。
- 英文大小写归一化。
- 全角字符转半角。
- 删除过短、纯标点、重复 query。

### 4.3 返回结果截断

上下文预算必须按 token 管理，而不是只按字符数。

优先级：

1. 永远保留原始题目。
2. 保留最近 1 到 2 次检索结果摘要。
3. 保留最新的环境纠错信息。
4. 丢弃或压缩更早的检索结果。

推荐配置：

```yaml
policy:
  max_total_sequence_length: 1250
  generation:
    max_new_tokens: 256

env:
  qa_agent:
    cfg:
      docs_dir: /data/docs
      max_turns: 6
      max_queries_per_turn: 3
      max_query_chars: 64
      search_top_k: 4
      max_retrieval_tokens_per_turn: 220
      max_total_retrieval_tokens: 480
      feedback_style: natural
      max_bad_output_chars: 160
```

如果实际 OOM，优先降低 `max_total_retrieval_tokens`、`search_top_k`、`max_turns`，不要首先砍掉 `generation.max_new_tokens`，否则 Base 模型可能没有足够空间完成推理与 boxed 输出。

### 4.4 返回给模型的自然语言上下文

环境返回不应堆叠太多管理字段。题型、轮数、命中数、query history 等信息适合写入日志和 metrics，不建议全部喂给模型。模型可见内容应尽量像一个自然的资料反馈，帮助它继续组织下一步检索或作答。

原则：

- 少用表格、键值对和“当前状态”类字段。
- 保留检索材料本身，必要时用一两句自然语言提示如何调整。
- 不把 reward、标准答案、轮数上限、命中数量等训练管理信息返回给模型。
- 错误纠正要短，不要形成一大段规则说明。

#### 成功检索反馈

```text

### 检索结果返回：
关于“{query}”，资料中有这些相关内容：

{short_chunk_1}

{short_chunk_2}

请根据这些资料继续思考；如果还不够确定，可以换一个更具体的关键词继续检索。
```

#### 空结果反馈

```text

### 检索结果返回：
没有找到和“{query}”直接相关的内容。

可以想一想题干里有没有更具体的设备名、系统名、英文缩写或工艺名，再换一个关键词检索。
```

#### 格式错误反馈

```text

你的上一轮输出格式不完整，我没有执行检索。
如果要查资料，请输出完整的 <search>关键词</search>。
如果已经确定答案，请输出 \boxed{答案}。
```

#### 答错后的反思样本

训练中终局答错后通常 `done=True`，不能在同一条 trajectory 里继续纠错；但可以把答错样本记录到验证日志或离线 replay buffer，用于后续 Prompt/Few-shot 或 SFT warmup。记录字段至少包括：

- 标准化后的预测答案与标准答案。
- 题型。
- 最后一次检索 query。
- 检索结果是否包含标准答案关键词。
- 错误类型：未检索到、检索到了但抽取错、格式错、题型规则错。

如果后续做二阶段训练，可以把这些失败样本转换成“错误 -> 反思 -> 正确动作”的格式数据，但不要把验证集答案写回训练 Prompt。

---

## 5. 答案标准化与判分

数据字段格式：

```json
{"query": "...", "expected_answer": "[single] A"}
```

`datasets/qa_rl` 的核心契约是：`query` 已经包含题型说明和 boxed 写法要求，`expected_answer` 只提供题型前缀与标准答案。评分逻辑必须以这个契约为准，不要把其他通用 QA 规则混进主评分。

需要先从 `expected_answer` 中解析题型前缀，再解析标准答案主体：

| 类型 | 标准答案示例 | 模型 boxed 示例 | 判分方式 |
| --- | --- | --- | --- |
| `[single]` | `A` | `\boxed{A}` | 只接受单个选项字母，和标准答案完全一致 |
| `[multiple]` | `A,B,C` | `\boxed{A,B,C}` | 只接受逗号分隔的选项字母序列，和标准答案顺序一致 |
| `[bool]` | `A` | `\boxed{A}` | 判断题在 QA 数据中也按 A/B 作答，A=对，B=错 |
| `[fill]` | `SQL server` | `\boxed{SQL server}` | 按空顺序比较；多空答案用分号分隔 |
| `[short]` | `离子源/Ion source ||| 分析磁场/AMU` | `\boxed{离子源; 分析磁场}` | 训练可用关键词覆盖率 proxy，验证尽量对齐平台 judge |

主评分标准化只做“格式噪声清理”，不能改变答案语义：

- 通用：提取最后一个 `\boxed{...}`，去掉首尾空白和末尾句号等轻微噪声。
- `[single]`：去空白、转大写；必须是单个 `A-L` 字母。不要接受选项正文。
- `[multiple]`：把中文逗号、顿号、全角逗号归一为英文逗号，去掉逗号两侧空白，转大写；必须与标准答案字符串一致。不要自动排序，因为题面要求按字母顺序输出，`C,A` 应视为格式不合规。
- `[bool]`：按数据集要求比较 `A` 或 `B`。不要在主评分中把“对/错”自动映射成 `A/B`，否则会奖励违反题面格式的输出。
- `[fill]`：把中文分号、全角分号归一为英文分号，去掉分隔符两侧空白，按空位顺序逐项比较。不要随意大小写折叠；英文大小写宽松匹配可以记录为辅助指标，但不应替代主 reward。
- `[short]`：将标准答案按 `|||` 拆成多个关键词槽位；每个槽位内部可再按 `/` 拆出同一要点的别名或中英文表达。模型 boxed 中的要点按分号拆分，命中某槽位任一别名即算该槽位覆盖。

简答题注意事项：

- `[short]` 的 `|||` 更接近“多个必需关键词槽位”，不是多个完整答案任选其一。
- 关键词覆盖率只适合作为训练 proxy reward，例如 `covered_slots / total_slots`。
- 考试说明提到简答题由平台内置 LLM 裁判打分；如果无法调用同一裁判，必须单独记录 `short_keyword_overlap`，不要把它等同于最终 `validation/accuracy`。
- 主 prompt 应要求模型在正文中完整作答，并在 boxed 中列出关键词；训练 proxy 可以主要看 boxed 关键词，但验证样本日志要同时保留正文，便于人工判断是否和平台 judge 一致。

### 5.1 简答题专用策略

简答题是最容易出现“本地 reward 高、平台 judge 不认可”的题型。建议把简答题拆成两个目标：

1. boxed 中列出足够多、足够准的关键词，方便 proxy reward 学习。
2. boxed 前的正文用自然语言完整解释，方便平台 LLM judge 判为覆盖要点。

简答题 prompt 可以额外加一句：

```text
简答题请先用一两句话完整说明，再把全部关键组成、步骤或要求放入 \boxed{}，关键词之间用分号分隔。
```

简答题 reward 建议：

| 项 | Reward |
| --- | --- |
| 覆盖每个关键词槽位 | `+ covered_slots / total_slots` |
| boxed 中要点数量明显少于槽位数 | `-0.1` |
| 正文长度过短，例如少于 12 个中文字且不是客观题 | `-0.1` |
| 检索结果包含未覆盖的 gold 别名 | 记录日志，不直接强扣 |
| 完全无 boxed | `-0.5` 到 `-1.0` |

不要强制模型只输出 boxed；对简答题，boxed 前的自然语言正文可能对平台 judge 很重要。Action Parser 只需要从全文提取最后一个 boxed 作为结构化关键词即可。

---

## 6. 课程式 Reward 设计

奖励必须随训练阶段变化。前期如果完全不给检索和格式过程奖励，Base 模型可能迟迟学不会结构化工具调用；后期如果继续给检索正奖励，又会诱导无意义搜索。

因此采用三阶段 curriculum。

在正式考试的 48 小时窗口里，建议先用固定 step 阈值启动，避免一开始就实现复杂动态调度：

| 阶段 | 默认 step 范围 | 主要目标 |
| --- | --- | --- |
| Phase A | `0-50` 或 `0-20%` | 学会协议和工具调用 |
| Phase B | `50-200` | 学会少搜、搜准、用证据 |
| Phase C | `200+` | 让最终正确率主导 |

如果实际 `valid_search_rate < 0.6` 或 `boxed_format_rate < 0.7`，延长 Phase A；如果搜索率过高且平均轮数上升但 accuracy 不涨，提前进入 Phase C。

### 6.1 Phase A：协议启动阶段

目标：让模型学会合法 `<search>`、读环境反馈、最终输出 `\boxed{}`。

适用条件：

- 训练前 10% 到 20% steps；或
- `valid_search_rate < 0.7`；或
- `boxed_format_rate < 0.8`。

建议奖励：

| 行为 | Reward |
| --- | --- |
| 合法 search | `+0.04` |
| search 命中非空结果 | 额外 `+0.02` |
| 合法 boxed 格式 | `+0.05` |
| 最终答案正确 | `+1.0` |
| 最终答案错误 | `0.0` 或 `-0.1` |
| 空 search / 未闭合标签 | `-0.05` |
| 无 search 且无 boxed | `-0.1` |
| 超过最大轮数未作答 | `-1.0` |

约束：单条 trajectory 的过程奖励上限建议 cap 到 `+0.2`，避免模型只靠搜索行为拿高分。

Phase A 不建议太长。只要验证样本里能稳定看到合法 `<search>` 和最终 `\boxed{}`，就应该降低过程奖励，否则模型会学成“为了拿分而搜索”。

### 6.2 Phase B：有效检索阶段

目标：让模型从“会搜”转向“搜得准、少重复、能利用证据”。

触发条件：

- `valid_search_rate >= 0.7` 且 `boxed_format_rate >= 0.8` 连续若干验证周期成立；或
- 达到预设 step 阈值。

建议奖励：

| 行为 | Reward |
| --- | --- |
| 合法 search | `+0.01` |
| 重复 query | `-0.02` |
| 空结果 query | `-0.02` |
| 最终答案正确 | `+1.0` |
| 最终答案错误 | `-0.2` |
| 格式错误 | `-0.1` |
| 超过最大轮数未作答 | `-1.0` |

### 6.3 Phase C：准确率对齐阶段

目标：最大化真实验证准确率，减少 reward hacking。

建议奖励：

| 行为 | Reward |
| --- | --- |
| search | `0.0` 或 `-0.005` |
| 最终答案正确 | `+1.0` |
| 最终答案错误 | `-0.2` |
| 格式错误导致无法结算 | `-0.5` 到 `-1.0` |
| 超过最大轮数未作答 | `-1.0` |

核心原则：

- 前期奖励结构化行为。
- 中期奖励有效行为。
- 后期只让最终任务结果主导梯度。

### 6.4 GRPO 方差与有效学习信号

GRPO 需要同一 prompt 的多条 generation 之间有 reward 差异。如果某批样本全错、全格式错或全靠过程奖励拿到相近分数，优势会很弱。

建议监控：

- `group_reward_std`：同一题 8 条采样的 reward 标准差。
- `all_wrong_group_rate`：同组全错比例。
- `format_only_win_rate`：靠格式/搜索过程奖励胜出但最终答案错的比例。

调参判断：

| 现象 | 处理 |
| --- | --- |
| 全部不会 boxed | 延长 Phase A，增加 few-shot，降低 prompt 复杂度 |
| 全部都能 search 但都答错 | 加强检索质量和证据截断，Phase B 不要继续奖励空泛 search |
| 同组 reward 几乎无方差 | 提高 `temperature` 到 1.0-1.1，保持 `num_generations_per_prompt=8` |
| 搜索很多但 accuracy 不涨 | 提前进入 Phase C，对重复/空 search 扣分 |

---

## 7. GRPO 训练配置建议

考试环境按 PDF 确认为单卡 H200 141GB，每个作业仍按 1 GPU 预算设计；`cluster/h200` profile 负责 pin 到 H200 节点，并保持 TP/PP/CP 为 1。9B Base + LoRA + colocated vLLM 在 H200 上有较大余量，但第一版仍建议先跑通多轮环境与格式指标，再逐步放大上下文和验证规模。

建议准备三档 H200 配置：

| 档位 | 用途 | 关键配置 |
| --- | --- | --- |
| `smoke` | 第一次提交，只验证链路 | `max_total_sequence_length=1250`，`max_rollout_turns=4`，`max_val_samples=32` |
| `main` | 主力训练配置 | `max_total_sequence_length=1536`，`max_rollout_turns=6`，`max_val_samples=128` |
| `stretch` | 链路稳定后冲分 | `max_total_sequence_length=2048`，`max_rollout_turns=6-8`，增大 retrieval budget |

首个正式 run 建议用 `smoke`，确认验证样本能看到完整“题目 -> search -> 检索反馈 -> boxed”轨迹后，再切到 `main`。

```yaml
defaults:
  - ../../configs/base/grpo_sliding_puzzle.yaml
  - ../../configs/models/qwen3.5-9b.yaml
  - ../../configs/base/grpo_megatron.yaml
  - ../../configs/base/grpo_lora.yaml

policy:
  model_name: "Qwen/Qwen3.5-9B-Base"
  tokenizer:
    name: "Qwen/Qwen3.5-9B-Base"
  max_total_sequence_length: 1536
  train_global_batch_size: 32
  generation_batch_size: 16
  generation:
    max_new_tokens: 256
    temperature: 1.0
    top_p: 0.95
    vllm_cfg:
      gpu_memory_utilization: 0.55

grpo:
  num_prompts_per_step: 4
  num_generations_per_prompt: 8
  max_rollout_turns: 6
  max_num_steps: 300
  val_period: 25
  val_at_start: true
  val_at_end: true
  max_val_samples: 128
  val_batch_size: 64

loss_fn:
  reference_policy_kl_penalty: 0.01

data:
  _override_: true
  max_input_seq_length: ${policy.max_total_sequence_length}
  shuffle: true
  num_workers: 1
  use_multiple_dataloader: false
  train:
    data_path: ${oc.env:QA_RL_DATA_DIR}/train.jsonl
  validation:
    data_path: ${oc.env:QA_RL_DATA_DIR}/val.jsonl
  # 多轮 Agent 数据由实验 run.py 读取上述 jsonl 并构造 DatumSpec。
  # 不依赖官方 ResponseDataset/env_name 自动接入。

env:
  qa_agent:
    cfg:
      docs_dir: /data/docs
      max_turns: 6
      reward_phase: curriculum
      feedback_style: natural
      search_top_k: 4
      max_queries_per_turn: 3
      max_query_chars: 64
      max_retrieval_tokens_per_turn: 240
      max_total_retrieval_tokens: 560
      max_bad_output_chars: 160
```

实验目录的 `cluster` 文件写 `h200`。如果作业 OOM，优先下调 `max_total_retrieval_tokens`、`search_top_k`、`max_rollout_turns` 和 `gpu_memory_utilization`；如果吞吐不足，再考虑降低 `num_generations_per_prompt` 或验证样本数。H200 上不要一开始就把 batch 和上下文都拉满，否则失败原因会很难定位。

注意：多轮自定义 Environment 通常不能只靠配置接入，实验目录必须提供 `run.py`，负责构造 dataset、实例化 `QAEnvironment`，并把 `task_to_env` 传给 `grpo_train`。

---

## 8. 监控指标

只看平均 reward 不够，必须记录行为指标和判分原因。

推荐指标：

| 指标 | 含义 |
| --- | --- |
| `validation/accuracy` | 主目标 |
| `train/final_correct_rate` | 训练样本最终答对率 |
| `train/boxed_format_rate` | 输出合法 boxed 的比例 |
| `train/valid_search_rate` | 合法 search 比例 |
| `train/search_nonempty_rate` | 检索非空比例 |
| `train/repeated_query_rate` | 重复检索比例 |
| `train/avg_turns_per_sample` | 平均轮数 |
| `train/truncation_rate` | 上下文或生成被截断比例 |
| `train/short_keyword_overlap` | 简答题 proxy 指标 |
| `train/reflection_fix_rate` | 格式错误后下一轮修正成功比例 |
| `train/extra_action_text_rate` | search 标签后继续输出推演文本的比例 |

验证样本日志至少应包含：

- 题型。
- 原始题目。
- 每轮 assistant 输出。
- 每轮 search query。
- 检索返回摘要。
- 环境自然语言反馈。
- 最终 boxed。
- search 前短草稿和 search 后多余文本统计。
- 标准答案。
- 标准化后的预测和标签。
- reward 分解和判分原因。

这些样本比单一曲线更能指导迭代。

---

## 9. 风险与防御

| 风险 | 表现 | 防御 |
| --- | --- | --- |
| 工具滥用 | 模型反复搜索但不作答 | 过程奖励 cap，Phase B/C 降低搜索奖励 |
| 格式学不会 | 大量无 search、无 boxed | Phase A 给格式奖励，Few-shot 强化协议 |
| 检索噪声 | 文档返回过长或无关 | BM25、query 清洗、top-k 限制、token 截断 |
| 上下文爆炸 | 多轮后 OOM 或截断题目 | token budget、只保留近期检索 |
| 反馈过度结构化 | 模型照抄状态字段、思考变僵硬 | 模型可见反馈用自然语言，管理字段只写日志 |
| 草稿膨胀 | 模型写长篇分析、不作答 | 只奖励有效 search/boxed；无动作输出扣分 |
| search 后脑补 | 检索标签后继续推演答案 | `</search>` stop string，search 后多余文本轻罚 |
| 简答 reward 偏差 | 关键词分高但平台 judge 不认可 | 单独记录 short 指标，尽量接入同一 judge |
| 同组奖励无方差 | GRPO 学不动 | 增大 `num_generations_per_prompt`，保留部分分奖励 |
| 数据泄漏 | 验证答案进入 prompt/cache | 搜索索引只读 `/data/docs`，不要索引 dataset 或日志 |

---

## 10. 实施顺序

建议按以下顺序落地，避免一次性实现过多变量：

1. 实现 `Action Parser`、答案标准化和纯规则判分单元测试。
2. 实现 `run.py` 读取 `QA_RL_DATA_DIR`，构造 train/val `DatumSpec`。
3. 实现无检索单轮 QA 环境，确认 boxed 判分链路可跑。
4. 接入 Search Engine，用少量 fake docs 测试检索和截断。
5. 实现多轮状态机、格式纠错和自然语言反馈模板。
6. 接入 Phase A/B reward 和 Few-shot，先提交 `smoke` 配置观察格式指标。
7. 调整检索质量、上下文预算和 short 题 proxy。
8. 链路稳定后再加入 Phase C 和更激进 H200 配置。

### 10.1 第一版必须通过的本地测试

即使不能真正训练，也应在本地用 fake 数据跑通这些测试：

| 测试 | 样例 | 期望 |
| --- | --- | --- |
| boxed 提取 | `abc \boxed{A}` | 提取 `A` |
| 多 boxed | `\boxed{B} ... \boxed{A}` | 取最后一个 `A` |
| search 提取 | `<search>控制图分析 Sample</search>` | 得到合法 query |
| search 前草稿 | `需要查 x。<search>x</search>` | 执行 search，不扣分 |
| search 后多余文本 | `<search>x</search>我猜答案是 A` | 执行 search，但训练轻微扣分 |
| single 判分 | gold `[single] A`，pred `a` | 正确 |
| multiple 判分 | gold `A,B,C`，pred `C,A,B` | 错误 |
| bool 判分 | gold `A`，pred `对` | 主评分错误，辅助指标可记格式近似 |
| fill 判分 | gold `SQL server`，pred `SQL Server` | 主评分按策略决定，需记录大小写宽松指标 |
| short proxy | gold `离子源/Ion source ||| 分析磁场/AMU` | 按槽位覆盖率给分 |
| 检索召回 | fake doc 含 `永久排除 Sample` | query 能召回对应 chunk |

### 10.2 明天首个作业验收标准

第一个 `smoke` 作业不追求高分，只判断系统链路是否健康：

- 作业能启动，不因 import、config、Ray actor、路径错误失败。
- 日志中能看到 `QA_RL_DATA_DIR` 和 `/data/docs` 被正确读取。
- 验证样本至少出现一条合法 `<search>...</search>`。
- 环境返回的检索结果没有把标准答案或 reward 泄漏给模型。
- 至少部分样本能以 `\boxed{...}` 终止。
- `validation/accuracy`、`boxed_format_rate`、`valid_search_rate` 能被记录。

如果首个作业失败，优先修运行链路；如果能跑但没有 search，优先修 prompt/Few-shot/Phase A；如果能 search 但全错，优先修检索召回和答案标准化。

最终成功标准：

- 训练早期 `valid_search_rate` 和 `boxed_format_rate` 快速上升。
- 中期 `avg_turns_per_sample` 不无意义增长。
- 后期 `validation/accuracy` 上升，而不是只有 `train/reward` 上升。
- 验证样本中能看到清晰的“检索证据 -> 推理 -> boxed 答案”轨迹。
