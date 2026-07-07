# grpo_qwen3.5-9b_qa-rl-agent_v1

QA-RL 考试实验：使用 GRPO + LoRA 训练 `Qwen/Qwen3.5-9B-Base`，在单卡 H200 141GB 上通过多轮 `<search>...</search>` 检索 `/data/docs`，最终输出 `\boxed{...}`。

## 目标

- 跑通自定义多轮 QA Environment。
- 让模型学习检索协议、答案格式和从资料中抽取证据。
- 最终答案 reward 直接调用官方 `common/rewards/qa_reward.py`，不维护另一套答案判分逻辑。
- 优化 `validation/accuracy`。

## 首次提交建议

```bash
lab validate grpo_qwen3.5-9b_qa-rl-agent_v1
lab submit grpo_qwen3.5-9b_qa-rl-agent_v1
```

第一个 run 只验收链路：能启动、能读 `QA_RL_DATA_DIR`、能索引 `/data/docs`、验证样本出现 search/result/boxed。

## 监控与回测

本实验开启 `require_docs: true`，如果 `/data/docs` 没挂载或没有 markdown/txt 文档，会在环境初始化时直接失败，避免空检索白跑。

本地检查协议和 DSL：

```bash
uv run python scripts/qa_backtest.py replay --limit 3
```

训练产生验证 JSONL 后，用 DSL 总结模型输出格式：

```bash
uv run python scripts/qa_backtest.py inspect <val_data_step*.jsonl> --max-examples 10
```

每 10 step 产生一个 validation 点（见 `grpo.val_period`）。提交后可轮询 accuracy 曲线：

```bash
uv run python scripts/monitor_accuracy.py <job_id> --interval 60
```
