import yaml
from promptwright.engine.factory import EngineFactory

# 加载 autotherm.yaml 配置
with open("schema/autotherm.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 用 factory 构建引擎，识别 type: field
engine = EngineFactory.create(config["data_engine"])

# 运行生成
engine.run(fields=config["fields"])
