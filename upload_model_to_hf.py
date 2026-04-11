"""
학습된 모델을 HuggingFace에 업로드합니다.

사용법:
    python upload_model_to_hf.py
    HF_TOKEN=hf_xxx python upload_model_to_hf.py
"""
import os
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "boradorish/qwen3-0.6b-finetuned"
LOCAL_DIR = Path("saves/qwen3-0.6b/full/sft")

token = os.environ.get("HF_TOKEN") or None

if not LOCAL_DIR.exists():
    print(f"[ERROR] 경로를 찾을 수 없습니다: {LOCAL_DIR}")
    exit(1)

print(f"업로드 중: {LOCAL_DIR} → {REPO_ID}")
api = HfApi(token=token)
api.upload_large_folder(
    repo_id=REPO_ID,
    folder_path=str(LOCAL_DIR),
    repo_type="model",
)
print(f"완료: https://huggingface.co/{REPO_ID}")
