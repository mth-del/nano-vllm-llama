import os
import sys
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODELS = {
    "qwen":  ("Qwen3-0.6B",              "~/huggingface/Qwen3-0.6B/"),
    "llama": ("Llama-3.2-1B-Instruct",   "~/huggingface/Llama-3.2-1B-Instruct/"),
}


def run_example(name: str, path: str, prompts: list[str]):
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        print(f"[Skip] {name}: model path not found: {path}")
        return
    print(f"\n===== {name} =====")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    model_inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(model_inputs, sampling_params)
    llm.exit()

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


def main():
    prompts = [
        "请你做个自我介绍。",
        "请列出 100 以内所有质数。",
    ]
    # 用法: python3 example.py [qwen|llama]
    # 不加参数默认跑 qwen（单 GPU 显存不够同时跑两个大模型）
    keys = sys.argv[1:] or ["qwen"]
    for key in keys:
        if key not in MODELS:
            print(f"[Error] 未知模型: {key}，可选: {list(MODELS)}")
            continue
        name, path = MODELS[key]
        run_example(name, path, prompts)


if __name__ == "__main__":
    main()
