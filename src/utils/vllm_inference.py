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


def get_lora_rank(adapter_path: str | Path) -> int:
    cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    return int(cfg.get("r", 16))


def load_vllm_model(
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    enforce_eager: bool = False,
    guided_decoding_backend: str | None = None,
    tokenizer_mode: str = "auto",
) -> VllmModel:
    try:
        # transformers 5's TokenizersBackend dropped this compatibility
        # property, while vLLM 0.10.2 still reads it when creating its cached
        # tokenizer wrapper.  Its semantic replacement is the same list of
        # special tokens for these checkpoints.
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
        if not hasattr(TokenizersBackend, "all_special_tokens_extended"):
            TokenizersBackend.all_special_tokens_extended = property(  # type: ignore[attr-defined]
                lambda tokenizer: tokenizer.all_special_tokens
            )
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
        "enforce_eager": enforce_eager,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if guided_decoding_backend is not None:
        # vLLM 0.10 configures structured-output backends at engine scope.
        kwargs["guided_decoding_backend"] = guided_decoding_backend
    if tokenizer_mode != "auto":
        kwargs["tokenizer_mode"] = tokenizer_mode

    lora_request = None
    if is_lora_adapter(model_path):
        base_model = get_base_model_id(model_path)
        # vLLM defaults to rank 16, but the published Glaive baseline uses
        # rank 64.  Set the engine limit from the adapter metadata so valid
        # adapters do not fail only after expensive model initialization.
        kwargs["max_lora_rank"] = max(16, get_lora_rank(model_path))
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
    # Meta's base Llama-3.2 checkpoints intentionally ship without a chat
    # template.  Use the standard Llama-3 header form only for that missing-
    # template case; tokenizers with a declared template retain their native
    # formatting.
    tokenizer_id = str(getattr(tokenizer, "name_or_path", ""))
    if getattr(tokenizer, "chat_template", None) is None and "Llama" in tokenizer_id:
        return [
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            for user_text in user_texts
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
    seed: int | None = None,
    use_tqdm: bool = False,
) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    outputs = engine.llm.generate(
        prompts,
        sampling_params,
        lora_request=engine.lora_request,
        use_tqdm=use_tqdm,
    )
    return [output.outputs[0].text for output in outputs]
