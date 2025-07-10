from .configuration_moe import MoEConfig
from .modeling_moe import MoEForCausalLM, MoEModel
# tools/hf_moe/__init__.py

# 注册到transformers的模型映射中
from transformers import AutoConfig, AutoModelForCausalLM

# 注册配置类
AutoConfig.register("moe", MoEConfig)

# 注册模型类
AutoModelForCausalLM.register(MoEConfig, MoEForCausalLM)


__all__ = ["MoEConfig", "MoEForCausalLM", "MoEModel"]