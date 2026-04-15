"""
GRPO (Group Relative Policy Optimization) 학습 스크립트

JSONSchemaBench 스키마를 주면 유효한 JSON을 생성하도록
베이스 모델을 강화학습으로 훈련합니다.

보상 함수 (0 ~ 1):
    1.0  — 출력이 유효한 JSON이고 스키마 검증 통과
    0.3  — 유효한 JSON이지만 스키마 검증 실패
    0.0  — JSON 파싱 불가

사전 준비:
    python src/prepare_jsonschemabench.py

단일 GPU:
    python src/grpo_train.py --model Qwen/Qwen3-8B

멀티 GPU (accelerate):
    accelerate launch --config_file src/train/accelerate_zero2.yaml \\
        src/grpo_train.py --model Qwen/Qwen3-8B

DeepSpeed ZeRO-3:
    accelerate launch --config_file src/train/accelerate_zero3.yaml \\
        src/grpo_train.py --model Qwen/Qwen3-8B --per-device-batch-size 1

학습 완료 후 SFT:
    1. generate_rft_data.py --model <grpo_output_dir> 로 SFT 데이터 생성
    2. qwen3_8B_rft.yaml 의 model_name_or_path 를 grpo 출력 경로로 변경 후 SFT 학습
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jsonschema
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from utils.prompt_loader import find_project_root

PROJECT_ROOT = find_project_root()


# ---------------------------------------------------------------------------
# Reward 함수
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> tuple[bool, object]:
    """
    모델 출력에서 JSON 을 추출합니다.
    반환: (파싱 성공 여부, 파싱된 객체 또는 None)
    """
    text = text.strip()

    # ```json ... ``` 코드 블록 우선 시도
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if m:
        candidate = m.group(1).strip()
        try:
            return True, json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # { ... } 또는 [ ... ] 블록 추출 (가장 바깥 블록)
    for pattern in (r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"):
        m = re.search(pattern, text)
        if m:
            try:
                return True, json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # 전체 텍스트 파싱 시도
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, None


def schema_compliance_reward(
    completions: list[str],
    schema_str: list[str],
    **kwargs,
) -> list[float]:
    """
    GRPOTrainer 에 전달되는 reward 함수.

    Args:
        completions : 모델이 생성한 텍스트 목록
        schema_str  : 각 샘플의 JSON Schema 문자열 목록 (dataset 컬럼에서 자동 전달)

    Returns:
        각 샘플의 reward 값 (0.0 / 0.3 / 1.0)
    """
    rewards: list[float] = []

    for completion, schema_s in zip(completions, schema_str):
        ok, json_obj = _extract_json(completion)

        if not ok:
            rewards.append(0.0)
            continue

        try:
            schema_obj = json.loads(schema_s)
            jsonschema.validate(instance=json_obj, schema=schema_obj)
            rewards.append(1.0)
        except (jsonschema.ValidationError, jsonschema.SchemaError, json.JSONDecodeError):
            rewards.append(0.3)

    return rewards


# ---------------------------------------------------------------------------
# 데이터셋 로드
# ---------------------------------------------------------------------------

def load_grpo_dataset(data_path: Path) -> Dataset:
    records = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        raise ValueError(f"데이터셋이 비어 있습니다: {data_path}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str):
    print(f"모델 로드 중: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print("모델 로드 완료.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO 강화학습 — JSONSchemaBench 스키마 준수")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="베이스 모델 경로 또는 HF repo ID")
    parser.add_argument(
        "--data",
        default="data/grpo/jsonschemabench.jsonl",
        help="prepare_jsonschemabench.py 로 생성한 JSONL (기본: data/grpo/jsonschemabench.jsonl)",
    )
    parser.add_argument("--output-dir", default="saves/qwen3-8b/grpo/jsonschemabench")
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO 그룹 크기 G (기본: 4)")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.01, help="KL 페널티 계수 (기본: 0.01)")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--report-to", default="wandb", choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--deepspeed", default=None, help="DeepSpeed config 경로 (선택)")
    parser.add_argument("--max-samples", type=int, default=None, help="학습 샘플 수 제한 (디버그용)")
    args = parser.parse_args()

    # 경로 해석
    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        print(f"[오류] 데이터 파일이 없습니다: {data_path}")
        print("먼저 아래 명령을 실행하세요:")
        print("  python src/prepare_jsonschemabench.py")
        sys.exit(1)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋
    print(f"데이터셋 로드: {data_path}")
    dataset = load_grpo_dataset(data_path)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"총 {len(dataset)}개 샘플 사용")

    # 모델 / 토크나이저
    model, tokenizer = load_model_and_tokenizer(args.model)

    # GRPO 설정
    report_to = None if args.report_to == "none" else args.report_to
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        # GRPO 특수 파라미터
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        # 저장 / 로깅
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=report_to,
        # DeepSpeed
        deepspeed=args.deepspeed,
        # 기타
        remove_unused_columns=False,   # schema_str 컬럼을 reward 함수에 넘기기 위해 필수
        dataloader_num_workers=4,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[schema_compliance_reward],
        args=grpo_config,
        train_dataset=dataset,
    )

    print("\nGRPO 학습 시작...")
    trainer.train()

    print(f"\n학습 완료. 모델 저장: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\n[다음 단계] SFT 데이터 생성:")
    print(f"  python src/generate_rft_data.py --model {output_dir}")
    print("이후 qwen3_8B_rft.yaml 의 model_name_or_path 를 위 경로로 변경 후 SFT 학습")


if __name__ == "__main__":
    main()
