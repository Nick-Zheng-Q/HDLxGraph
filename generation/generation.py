import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from anthropic import Anthropic
from openai import OpenAI

qwen_key = ""
anthropic_key = "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"

checkpoint = "bigcode/starcoder2-15b"
device = "cpu" # for GPU usage or "cpu" for CPU usage

class MultiModelCoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.starcoder_model = None
        self.starcoder_tokenizer = None
        self.anthropic_client = None
        self.qwen_client = None

    def load_model(self, model_name):
        """动态加载指定模型"""
        if model_name == "starcoder2-7b":
            if not self.starcoder_model:
                self.starcoder_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                self.starcoder_model = AutoModelForCausalLM.from_pretrained(
                    "bigcode/starcoder2-7b",
                ).to(device)
                self.starcoder_tokenizer.pad_token = self.starcoder_tokenizer.eos_token
            return self.starcoder_model
        
        elif model_name == "claude":
            if not self.anthropic_client:
                self.anthropic_client = Anthropic(api_key=anthropic_key)
            return self.anthropic_client
        
        elif model_name == "qwen":
            if not self.qwen_client:
                self.qwen_client = OpenAI(
                    api_key=qwen_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            return self.qwen_client
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_code(self, model_name, system_prompt, full_prompt):
        """统一生成接口"""
        
        self.load_model(model_name)
        if model_name == "starcoder2-7b":
            client = self.starcoder_model
            tokenizer = self.starcoder_tokenizer
            prompt = system_prompt + '\n' + full_prompt
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192,
            ).to(device)
            outputs = client.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # <--- 添加注意力掩码
                pad_token_id=tokenizer.eos_token_id,    # <--- 明确pad_token
                max_new_tokens=1024,                    # <--- 使用max_new_tokens
            )
            return tokenizer.decode(outputs[0])
        
        elif model_name == "claude":
            client = self.anthropic_client
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ]
            )
            return response.content[0].text
        
        elif model_name == "qwen":
            client = self.qwen_client
            completion = client.chat.completions.create(
                model="qwen2.5-coder-7b-instruct", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_prompt}],
                )
            return completion
        else:
            raise ValueError("Invalid model selection")

def main_generation(model, system_prompt, full_prompt):
    coder = MultiModelCoder()
    result = coder.generate_code(
        model_name=model,
        system_prompt=system_prompt,
        full_prompt=full_prompt
    )
    
    return result
