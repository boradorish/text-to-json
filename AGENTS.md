# text-to-json (STAGE)

스프레드시트 기반 source-grounded text-to-JSON 학습 데이터 생성 + 벤치마크(STAGE-Eval) 프로젝트.

## 지금 최우선 작업

**`EXPERIMENTS.md`를 먼저 읽으세요.** NeurIPS 2026 SLM-Agents 워크숍 제출(마감 2026-09-07 13:00 UTC)을 위한
추가 실험 3개(xgrammar 비교 / BFCL / CORD·ExtractBench 실세계 평가)가 정의되어 있고,
이 실험들의 구현·실행·논문 반영이 현재 이 레포의 목표입니다.

## 레포 구조

- `benchmark/` — 추론(`inference.py`)·채점(`evaluate.py`)·에러분석 스크립트. 새 실험 코드도 여기에 추가
- `src/` — 데이터 생성 파이프라인, `src/utils/vllm_inference.py` (vLLM 래퍼)
- `overleaf-paper/` — 논문 (Overleaf git 연동, main 파일: `neurips2026.tex`). 메인 레포에는 커밋되지 않음(gitignore)
- `data/`, `outputs/` — 데이터·결과 (대용량, 커밋 금지)

## 규칙

- 추론 설정 기본값: vLLM, temperature 0.6, top-p 1.0, max_new_tokens 3100, max_model_len 8192, seed 42
- 평가 지표는 기존 `benchmark/evaluate.py`의 PFR/EMR/SCR/NR/VA를 재사용할 것 (새 지표 임의 추가 금지)
- 실험 결과는 `outputs/` 아래 실험명 디렉토리로 저장하고, 요약을 `EXPERIMENTS.md`에 추기
