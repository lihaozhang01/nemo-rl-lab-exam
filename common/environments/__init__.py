"""跨实验复用的自定义 GRPO 环境。"""

try:  # 集群运行时有 Ray/NeMo-RL，本地单测环境通常没有。
    from common.environments.qa_env import QARewardEnv  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - 本地无训练依赖时允许导入子模块。
    QARewardEnv = None  # type: ignore
