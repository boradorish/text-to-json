"""
Unsloth를 사용하여 언어 모델을 파인튜닝하는 스크립트.

이 스크립트는 4-bit 양자화와 LoRA를 사용하여 메모리 효율적인 파인튜닝을 수행합니다.
Hugging Face의 데이터셋을 로드하고, 모델의 채팅 템플릿에 맞게 자동으로 포맷팅한 후,
TRL의 SFTTrainer를 사용하여 학습을 진행합니다.

실행 전 필요한 라이브러리를 설치하세요:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "wandb" # (선택 사항) Weights & Biases 로깅을 원할 경우
pip install "ninja" "packaging"
pip install "datasets" "trl" "transformers" "accelerate"
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def create_and_train_model(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    max_seq_length: int = 2048,
    r: int = 16,
    lora_alpha: int = 16,
):
    """
    Unsloth를 사용하여 모델을 로드, 파인튜닝하고 저장합니다.

    Args:
        model_name: 파인튜닝할 Hugging Face 모델 이름 (Unsloth 지원 모델).
        dataset_name: (사용자 정의 데이터셋 사용 시 None) 학습에 사용할 Hugging Face 데이터셋 이름.
        output_dir: 학습 결과(체크포인트, 어댑터 등)가 저장될 디렉터리.
        max_seq_length: 모델이 처리할 최대 시퀀스 길이.
        r: LoRA의 rank.
        lora_alpha: LoRA의 alpha.
    """
    # 1. Unsloth를 사용하여 4-bit 양자화된 모델과 토크나이저 로드
    # FastLanguageModel은 메모리 사용량 감소와 속도 향상을 자동으로 처리합니다.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # 자동으로 BFloat16 또는 Float16 감지
        load_in_4bit=True,
    )

    # 2. LoRA 설정으로 모델 준비
    # PEFT(Parameter-Efficient Fine-Tuning)를 적용하여 전체 파라미터가 아닌
    # 일부(어댑터)만 학습하여 효율을 높입니다.
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # 3. 데이터셋 로드 및 프롬프트 포맷팅
    # 모델이 '리포트'와 'JSON 스키마'를 보고 'JSON 결과'를 생성하도록 학습시킵니다.
    custom_prompt = """You are an expert AI that converts a report and a JSON schema into a valid JSON object.

### REPORT:
{}

### JSON_SCHEMA:
{}

### JSON_OUTPUT:
{}"""

    def formatting_prompts_func(examples):
        reports = examples["report"]
        schemas = examples["json_schema"]
        outputs = examples["json_output"]
        texts = []
        for report, schema, output in zip(reports, schemas, outputs):
            # EOS(End-of-sequence) 토큰을 추가하여 모델이 응답의 끝을 학습하도록 합니다.
            text = custom_prompt.format(report, schema, output) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    # 프로젝트 루트 경로를 기준으로 커스텀 데이터셋 로드
    project_root = Path(__file__).resolve().parent.parent
    dataset = load_custom_dataset(project_root)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. 학습 인자(TrainingArguments) 설정
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # 예제를 위해 작은 값으로 설정, 실제로는 더 길게 설정
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="wandb", # (선택 사항) wandb 로깅 사용
    )

    # 5. SFTTrainer를 사용하여 파인튜닝 시작
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # 시퀀스 길이가 다양할 경우 False 권장
        args=training_args,
    )

    print("파인튜닝을 시작합니다...")
    trainer.train()
    print("파인튜닝이 완료되었습니다.")

    # 6. 학습된 LoRA 어댑터 저장
    model.save_pretrained(f"{output_dir}/lora_adapters")
    print(f"학습된 LoRA 어댑터를 '{output_dir}/lora_adapters'에 저장했습니다.")


if __name__ == "__main__":

    import wandb

    # Start a new wandb run to track this script.
    run =  wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="boradorish-seoul-national-university",
        # Set the wandb project where this run will be logged.
        project="my-awesome-project",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )

    # --- 설정 ---
    # Unsloth가 지원하는 Llama, Mistral 등의 4bit 모델 사용 가능
    # 예: "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    
    # 커스텀 데이터셋을 사용하므로 DATASET_NAME은 None으로 설정합니다.
    # 만약 공개 데이터셋으로 다시 테스트하고 싶다면 이 값을 설정하세요.
    DATASET_NAME = None
    
    OUTPUT_DIR = "outputs"

    create_and_train_model(
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME, # 현재는 사용되지 않음
        output_dir=OUTPUT_DIR,
    )

    run.finish()