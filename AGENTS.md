# text-to-json (STAGE)

스프레드시트 기반 source-grounded text-to-JSON 학습 데이터 생성 + 벤치마크(STAGE-Eval) 프로젝트.

## 지금 최우선 작업

**`EXPERIMENTS.md`를 먼저 읽으세요.** NeurIPS 2026 SLM-Agents 워크숍 제출(마감 2026-09-07 13:00 UTC)을 위한
추가 실험 3개(xgrammar 비교 / BFCL / CORD·ExtractBench 실세계 평가)가 정의되어 있고,
이 실험들의 구현·실행·논문 반영이 현재 이 레포의 목표입니다.

실험 1~7(BFCL 2b~2f, 4A, CORD·ExtractBench xgrammar, Llama Instruct 정정, SGD 파일럿, 실험 6 비용)은 완료됐습니다(`## 현재 실행 상태`, `## 실행 기록*`).
**지금 할 일은 실험 8 → 9 → (10)** 입니다. `EXPERIMENTS.md`의 `## 다음 실행 (5차)` runbook을 따르세요.
- 실험 8 (필수 정정): base Qwen3-4B가 thinking 모드로 실행돼 있었습니다(798개 전부 `<think>`, 29% 미종료 절단). `--no-thinking`으로 base만 재실행합니다.
  2026-09-03 05:08에 STAGE-Eval 851개 자유/xgrammar 두 run을 `outputs/nothink/`에서 시작해 두었습니다(로그 `outputs/nothink/*.log`). 끝나면 채점하고 CORD·ExtractBench·실험 6의 base도 같은 플래그로 재실행합니다.
- 실험 9: CORD 입력을 단어 좌표 기반 시각적 행 표로 렌더링해 STAGE SFT의 위치 기반 배정 오류를 줄이는 실험. 실험 10은 9가 격차를 못 닫을 때만.
- 실험 4B 장문맥 재학습은 진입 조건 미충족으로 실행 금지. 완료된 결과는 재실행하지 않습니다.
포지셔닝: STAGE-SFT는 tool router가 아니라 에이전트의 지각/상태 추출 계층. BFCL은 Appendix scope 근거로만 사용. 논문 반영 시 부정적 판정(CORD, SGD, Qwen3 ExtractBench VA)도 누락하지 않습니다.

## 레포 구조

- `benchmark/` — 추론(`inference.py`)·채점(`evaluate.py`)·에러분석 스크립트. 새 실험 코드도 여기에 추가
- `src/` — 데이터 생성 파이프라인, `src/utils/vllm_inference.py` (vLLM 래퍼)
- `overleaf-paper/` — 논문 (Overleaf git 연동, main 파일: `neurips2026.tex`). 메인 레포에는 커밋되지 않음(gitignore)
- `data/`, `outputs/` — 데이터·결과 (대용량, 커밋 금지)

## 규칙

- 추론 설정 기본값: vLLM, temperature 0.6, top-p 1.0, max_new_tokens 3100, max_model_len 8192, seed 42
- 평가 지표는 기존 `benchmark/evaluate.py`의 PFR/EMR/SCR/NR/VA를 재사용할 것 (새 지표 임의 추가 금지)
- 실험 결과는 `outputs/` 아래 실험명 디렉토리로 저장하고, 요약을 `EXPERIMENTS.md`에 추기
- `outputs/`는 `.gitignore` 대상이지만 실험 1~3 산출물은 이미 `git add -f`로 커밋돼 있음. 새로 커밋할 때는 요약·점수·진단 파일만 추가하고, 원출력(`result/`)이나 대용량 `.xlsx`는 강제 추가하지 말 것

## 인프라 (k8s 예약 pod)

- 실행 환경은 로컬이 아니라 mlxp 예약 pod. 레포 사본: `~/work/sunghee/text-to-json`, Python: `~/work/sunghee/venv/bin/python` (vLLM, xgrammar, bfcl_eval 설치됨)
- pod는 12시간마다 재생성되고 이름이 바뀜. `~/work`(공유 영구 볼륨)만 남고 `/root`는 초기화됨. 최신 pod는 라벨
  `lab.snupi/mlxp-username=interns,lab.snupi/workload-type=reservation-shell`로 찾는다
- `~/work`는 여러 사람이 공유하는 볼륨이므로 **개인 인증 정보(codex/HF 토큰 등)를 저장하지 말 것**
- GPU는 `nvidia-smi`로 빈 카드를 확인하고 `CUDA_VISIBLE_DEVICES`로 지정. 다른 사람 작업이 있는 카드는 쓰지 않음
- 작업 시작 전 `git pull`, 끝나면 커밋·푸시. 로컬 사본과 pod 사본이 갈라지지 않게 한다
