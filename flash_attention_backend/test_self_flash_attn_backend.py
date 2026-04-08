import os

from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum

register_backend(
    AttentionBackendEnum.CUSTOM,
    "toy_flash_attn.ToyFlashAttentionBackend",
)

# Pin GPT-2 to the locally cached revision and force offline resolution.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

model_name = "gpt2"
model_revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

llm = LLM(
    model=model_name,
    revision=model_revision,
    tokenizer_revision=model_revision,
    max_model_len=512,
    gpu_memory_utilization=0.8,
    max_num_seqs=1,
    attention_backend="CUSTOM",
    kv_cache_dtype="bfloat16"
)

prompts = [
    "Please introduce yourself in one short paragraph.",
]

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.9,
    max_tokens=32,
)

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt {i}: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 40)

del llm
