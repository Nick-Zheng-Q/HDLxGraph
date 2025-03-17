import argparse
from anthropic import Anthropic
from openai import OpenAI
import ollama

qwen_key = ""
anthropic_key = ""

class MultiModelCoder:
    def __init__(self):
        self.starcoder_model = None
        self.starcoder_tokenizer = None
        self.anthropic_client = None
        self.qwen_client = None

    def load_model(self, model_name):
        """动态加载指定模型"""
        if model_name =="llama":
            pass
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
        if model_name =="llama":
            prompt = system_prompt + '\n' + full_prompt
            answer = ollama.generate(model='llama3.1', prompt=prompt)
            return answer['response']
            
        elif model_name == "claude":
            client = self.anthropic_client
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                system=system_prompt,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=1024
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
            return completion.choices[0].message.content
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
