import yaml
from promptwright.topic_tree import TopicTree, TopicTreeArguments
from promptwright.dataset import Dataset

# 加载配置
with open("schema/autotherm.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ✅ 构造 TopicTree 参数（仅两个参数：root_prompt + model_name）
tree_args = TopicTreeArguments(
    root_prompt="Thermal Comfort Simulation",
    model_name=cfg.get("model_name", "ollama/mistral:latest")
)

# ✅ 构建并保存 topic tree
topic_tree = TopicTree(args=tree_args)
topic_tree.build(output_path="topic_tree.jsonl")

# ✅ 构建数据集并生成
dataset = Dataset(
    topic_tree_path="topic_tree.jsonl",
    output_path="dataset.jsonl",
    model_name=cfg.get("model_name", "ollama/mistral:latest"),
    max_tokens=cfg.get("max_tokens", 512),
    temperature=cfg.get("temperature", 0.7),
    num_samples=cfg.get("num_samples", 5),
    batch_size=cfg.get("batch_size", 1),
    output_format=cfg.get("output_format", "jsonl"),
    schema=cfg.get("fields", [])
)

dataset.generate()
