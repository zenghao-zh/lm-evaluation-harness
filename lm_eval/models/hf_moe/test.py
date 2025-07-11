import os, sys

from modeling_moe import MoEForCausalLM
from transformers import AutoTokenizer

# 模型路径
model_path = "/ssd_1234/haozeng/workspace/Megatron-LM/checkpoints/moe-0.6b-baseline-torch/torch/trfs_checkpoint_iter_0018750"

# 从本地加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 从本地加载自定义的大语言模型
model = MoEForCausalLM.from_pretrained(model_path, torch_dtype='auto')

# 测试推理示例
input_text = "Are you human?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("输入文本:", input_text)
print("生成结果:", decoded_output)