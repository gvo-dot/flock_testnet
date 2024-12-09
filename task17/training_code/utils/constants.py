
from .validation.template import template_dict
from dataclasses import asdict

qwen_template = asdict(template_dict["qwen1.5"])

gemma_template = asdict(template_dict["gemma"])

llama2_template = asdict(template_dict["llama2"])

llama3_template = asdict(template_dict["llama3"])

llama3_template = asdict(template_dict["mixtral"])

mistral_template = asdict(template_dict["mistral"])

# yifan - add llama3-8B & mistral-7B template - 18/06/2024
model2template = {
    "Qwen/Qwen1.5-0.5B": qwen_template,
    "Qwen/Qwen1.5-1.8B": qwen_template,
    "Qwen/Qwen1.5-7B": qwen_template,
    "google/gemma-2b": gemma_template,
    "google/gemma-7b": gemma_template,
    "Qwen/Qwen2-72B": qwen_template,
    "Qwen/Qwen2-7B": qwen_template,
    "mistralai/Mistral-7B-v0.3": mistral_template,
    "meta-llama/Meta-Llama-3-8B": llama3_template,
    "mistralai/Mistral-7B-Instruct-v0.3": mistral_template,
    "Qwen/Qwen2-7B-Instruct": qwen_template,
    "Qwen/Qwen1.5-4B": qwen_template,
    "Qwen/Qwen1.5-4B-Chat": qwen_template,
    "Qwen/Qwen2.5-3B": qwen_template,
    "Qwen/Qwen2.5-1.5B": qwen_template,
    "Qwen/Qwen2.5-0.5B": qwen_template,
    "google/gemma-2-2b": gemma_template,
}


# yifan - add llama3-8B & mistral-7B template - 18/06/2024
model2size = {
    "Qwen/Qwen1.5-0.5B": 620_000_000,
    "Qwen/Qwen1.5-1.8B": 1_840_000_000,
    "Qwen/Qwen1.5-7B": 7_720_000_000,
    "google/gemma-2b": 2_510_000_000,
    "google/gemma-7b": 8_540_000_000,
    "Qwen/Qwen2-7B": 7_620_000_000,
    "meta-llama/Meta-Llama-3-8B": 8_030_000_000,
    "mistralai/Mistral-7B-v0.3": 7_250_000_000,
    "mistralai/Mistral-7B-Instruct-v0.3": 7_250_000_000,
    "Qwen/Qwen2-7B-Instruct": 7_620_000_000,
    "Qwen/Qwen1.5-4B": 3_950_000_000,
    "Qwen/Qwen1.5-4B-Chat": 3_950_000_000,
    "Qwen/Qwen2.5-3B": 3_090_000_000,
    "Qwen/Qwen2.5-1.5B": 1_540_000_000,
    "Qwen/Qwen2.5-0.5B": 494_000_000,
    "google/gemma-2-2b": 2_610_000_000,
}

model2base_model = {
    "Qwen/Qwen1.5-0.5B": "qwen1.5",
    "Qwen/Qwen1.5-1.8B": "qwen1.5",
    "Qwen/Qwen1.5-7B": "qwen1.5",
    "google/gemma-2b": "gemma",
    "google/gemma-7b": "gemma",
    "Qwen/Qwen2-72B": "qwen1.5",
    "Qwen/Qwen2-7B": "qwen1.5",
    "meta-llama/Meta-Llama-3-8B": "llama3",    
    "mistralai/Mistral-7B-v0.3": "mistral",
    "mistralai/Mistral-7B-Instruct-v0.3":  "mistral",
    "Qwen/Qwen2-7B-Instruct": "qwen1.5",
    "Qwen/Qwen1.5-4B": "qwen1.5",
    "Qwen/Qwen1.5-4B-Chat": "qwen1.5",
    "Qwen/Qwen2.5-3B": "qwen1.5",
    "Qwen/Qwen2.5-1.5B": "qwen1.5",
    "Qwen/Qwen2.5-0.5B": "qwen1.5",
    "google/gemma-2-2b": "gemma",
}