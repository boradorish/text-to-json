from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VllmModel:
    llm: object
    tokenizer: object
    lora_request: object | None = None


def is_lora_adapter(model_path: str | Path) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()


def get_base_model_id(adapter_path: str | Path) -> str:
    cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    return cfg["base_model_name_or_path"]


def load_vllm_model(
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
) -> VllmModel:
    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError(
            "vLLM을 import하지 못했습니다. vLLM이 없거나 현재 CUDA/PyTorch 환경과 "
            f"맞지 않는 wheel이 설치되어 있을 수 있습니다. 원인: {exc}"
        ) from exc

    model_path = str(model_path)
    kwargs = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len

    lora_request = None
    if is_lora_adapter(model_path):
        base_model = get_base_model_id(model_path)
        tokenizer_src = str(tokenizer_path) if tokenizer_path else base_model
        print(f"LoRA 어댑터 감지. vLLM 베이스 모델: {base_model}")
        llm = LLM(
            model=base_model,
            tokenizer=tokenizer_src,
            enable_lora=True,
            **kwargs,
        )
        lora_request = LoRARequest("adapter", 1, model_path)
    else:
        tokenizer_src = str(tokenizer_path) if tokenizer_path else model_path
        print(f"vLLM 모델 로드 중: {model_path}")
        llm = LLM(model=model_path, tokenizer=tokenizer_src, **kwargs)

    tokenizer = llm.get_tokenizer()
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("vLLM 모델 로드 완료.")
    return VllmModel(llm=llm, tokenizer=tokenizer, lora_request=lora_request)


def build_chat_prompts(tokenizer, system_prompt: str, user_texts: list[str]) -> list[str]:
    messages_list = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        for text in user_texts
    ]
    return [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]


def generate_texts(
    engine: VllmModel,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_tqdm: bool = False,
) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    outputs = engine.llm.generate(
        prompts,
        sampling_params,
        lora_request=engine.lora_request,
        use_tqdm=use_tqdm,
    )
    return [output.outputs[0].text for output in outputs]
