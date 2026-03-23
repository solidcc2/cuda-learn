from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum

register_backend(
    AttentionBackendEnum.FLASH_ATTN,
    "flash_attention_backend.MinimalFlashAttentionBackend",
)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

llm = LLM(
    model=model_name,
    max_model_len=512,
    gpu_memory_utilization=0.8,
    max_num_seqs=1,
    attention_backend="FLASH_ATTN",
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
