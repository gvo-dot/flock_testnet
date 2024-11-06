qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "system": "You are a helpful assistant.",
}

gemma_template = {
    "system_format": "<bos>",
    "user_format": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "assistant_format": "{content}<|eot_id|>",
    "system": None,
}


# yifan - add llama3-8B & mistral-7B template - 18/06/2024
llama3_template = {
    "system_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    "user_format": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "assistant_format": "{content}<|eot_id|>",
    "system": None,
   "stop_word": "<|eot_id|>",
}


mistral_template = {
    "system_format": "<s>",
    "user_format": "[INST]{content}[/INST]",
    "assistant_format": "{content}</s>",
    "system": "",
   "stop_word": "</s>",
}

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
    "mistralai-Mistral-7B-Instruct-v0.3": mistral_template,
    "Qwen/Qwen2-7B-Instruct": qwen_template,
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
    "mistralai-Mistral-7B-Instruct-v0.3": 7_250_000_000,
    "Qwen/Qwen2-7B-Instruct": 7_620_000_000,
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
    "mistralai-Mistral-7B-Instruct-v0.3":  "mistral",
    "Qwen/Qwen2-7B-Instruct": "qwen1.5",
}
