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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="qwen",
        help="Model to run. Defaults to qwen.",
    )
    parser.add_argument("-i", action="store_true", dest="interactive")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_config = MODEL_CONFIGS[args.model]
    model_revision = model_config["revision"]

    llm = LLM(
        model=model_config["model"],
        revision=model_revision,
        tokenizer_revision=model_revision,
        max_model_len=512,
        gpu_memory_utilization=0.8,
        max_num_seqs=1,
        # attention_backend="FLASH_ATTN",
        attention_backend="CUSTOM",
        kv_cache_dtype="bfloat16"
        # kv_cache_dtype="auto"
    )
    engine = llm.llm_engine
    vconfig = engine.vllm_config

    print("model                  =", model_config["model"])
    print("revision               =", model_revision)
    print("cache_config.cache_dtype =", vconfig.cache_config.cache_dtype)
    print("model_config.dtype      =", vconfig.model_config.dtype)
    print("resolved kv torch dtype =", get_kv_cache_torch_dtype(
        vconfig.cache_config.cache_dtype,
        vconfig.model_config.dtype,
    ))

    prompts = [
        "Please introduce yourself in one short paragraph.",
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=32,
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
