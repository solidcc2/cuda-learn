import argparse
import os

from vllm import LLM, SamplingParams
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum

register_backend(
    AttentionBackendEnum.CUSTOM,
    "toy_flash_attn.ToyFlashAttentionBackend",
)

# Force offline resolution for locally cached models.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_CONFIGS = {
    "gpt2": {
        "model": "gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    },
    "qwen": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    },
}


def _attention_config() -> tuple[str, str, str]:
    impl = os.environ.get("TOY_FLASH_ATTN_USE", "bf16")
    if impl in {"offical", "official"}:
        return impl, "FLASH_ATTN", "auto"
    return impl, "CUSTOM", "bfloat16"


def _make_prompts(batch_size: int) -> list[str]:
    base_prompts = [
        "Please introduce yourself in one short paragraph.",
        "Explain what GPU attention does in one sentence.",
        "Write a short greeting to a new teammate.",
        "List three benefits of batching requests.",
        "Describe CUDA in simple terms.",
        "Give one tip for debugging numerical kernels.",
        "Summarize what a KV cache is.",
        "Explain why GQA reduces KV cache size.",
    ]
    return [base_prompts[i % len(base_prompts)] for i in range(batch_size)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="qwen",
        help="Model to run. Defaults to qwen.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to generate in non-interactive mode.",
    )
    parser.add_argument(
        "-t",
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum generated tokens per prompt.",
    )
    parser.add_argument("-i", action="store_true", dest="interactive")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


def main() -> None:
    args = _parse_args()
    model_config = MODEL_CONFIGS[args.model]
    model_revision = model_config["revision"]
    attention_impl, attention_backend, kv_cache_dtype = _attention_config()

    llm = LLM(
        model=model_config["model"],
        revision=model_revision,
        tokenizer_revision=model_revision,
        max_model_len=512,
        gpu_memory_utilization=0.8,
        max_num_seqs=args.batch_size,
        attention_backend=attention_backend,
        kv_cache_dtype=kv_cache_dtype,
    )
    engine = llm.llm_engine
    vconfig = engine.vllm_config

    print("model                  =", model_config["model"])
    print("revision               =", model_revision)
    print("TOY_FLASH_ATTN_USE     =", attention_impl)
    print("attention_backend      =", attention_backend)
    print("kv_cache_dtype arg     =", kv_cache_dtype)
    print("batch_size             =", args.batch_size)
    print("max_tokens             =", args.max_tokens)
    print("cache_config.cache_dtype =", vconfig.cache_config.cache_dtype)
    print("model_config.dtype      =", vconfig.model_config.dtype)
    print("resolved kv torch dtype =", get_kv_cache_torch_dtype(
        vconfig.cache_config.cache_dtype,
        vconfig.model_config.dtype,
    ))

    prompts = _make_prompts(args.batch_size)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    if args.interactive:
        print("Interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                prompt = input("prompt> ").strip()
            except EOFError:
                print()
                break
            if not prompt:
                continue
            if prompt in {"exit", "quit"}:
                break
            outputs = llm.generate([prompt], sampling_params)
            print()
            print("reply: ", outputs[0].outputs[0].text)
            print("-" * 40)
    else:
        outputs = llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            print(f"Prompt {i}: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")
            print("-" * 40)

    del engine
    del llm


if __name__ == "__main__":
    main()
