import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def run_example(name: str, path: str, prompts: list[str]):
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

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


def main():
    prompts = [
        "请你做个自我介绍。",
        "请列出 100 以内所有质数。",
    ]
    qwen_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llama_path = os.path.expanduser("~/huggingface/Llama-3.2-3B-Instruct/")
    run_example("Qwen3-0.6B", qwen_path, prompts)
    run_example("Llama-3.2-3B-Instruct", llama_path, prompts)


if __name__ == "__main__":
    main()
