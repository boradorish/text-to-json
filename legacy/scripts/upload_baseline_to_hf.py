"""
Baseline SFT 모델을 HuggingFace에 업로드합니다.

사용법:
    python upload_baseline_to_hf.py
    HF_TOKEN=hf_xxx python upload_baseline_to_hf.py
    python upload_baseline_to_hf.py --repo-id boradorish/qwen3-4b-baseline-sft
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "boradorish/qwen3-4b-baseline-sft"
DEFAULT_LOCAL_DIR = Path("saves/qwen3-4b/full/baseline_sft")


def main():
    parser = argparse.ArgumentParser(description="Baseline SFT 모델 HuggingFace 업로드")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"HF repo ID (기본: {DEFAULT_REPO_ID})")
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR), help=f"업로드할 로컬 경로 (기본: {DEFAULT_LOCAL_DIR})")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    token = os.environ.get("HF_TOKEN") or None

    if not local_dir.exists():
        print(f"[ERROR] 경로를 찾을 수 없습니다: {local_dir}")
        print(f"  학습 완료 후 {local_dir} 경로에 모델이 있는지 확인하세요.")
        exit(1)

    print(f"업로드 중: {local_dir} → {args.repo_id}")
    api = HfApi(token=token)
    api.upload_large_folder(
        repo_id=args.repo_id,
        folder_path=str(local_dir),
        repo_type="model",
    )
    print(f"완료: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()