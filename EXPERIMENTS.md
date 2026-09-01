# 워크숍 제출용 추가 실험 계획 (SLM-Agents @ NeurIPS 2026)

> **이 문서가 지금 이 레포의 최우선 작업입니다.** 아래 실험 3개를 돌려서 논문에 추가해야 합니다.
> 에이전트(codex/claude)는 새 작업을 시작하기 전에 이 문서를 읽고, 여기 정의된 실험 구현을 우선하세요.

## 배경

- 논문: **STAGE** — 스프레드시트 기반 source-grounded text-to-JSON 학습 데이터 생성 (`overleaf-paper/neurips2026.tex`)
- 제출처: [SLM-Agents workshop @ NeurIPS 2026](https://slmw2026.github.io/) — 주제: "작은 언어모델(SLM)로 에이전트 시스템을 만들 수 있는가"
- **마감: 2026-09-07 (일) 13:00 UTC = 한국시간 22:00** (OpenReview 기준 확인됨)
- 분량: long paper 6페이지에 맞춰 제출
- **프레이밍**: 우리가 만든 JSON 추출기를 "agentic system의 tool calling 모델로 쓸 수 있다"로 포지셔닝.
  스키마 준수 + 값 정확한 JSON 생성은 툴콜/API 인자 구성의 핵심 전제인데 1B~4B SLM이 이걸 못 함
  → STAGE 데이터로 SFT하면 고쳐진다는 스토리.
- 기존 결과: Qwen3-4B가 STAGE SFT 후 STAGE-Eval EMR 31.37→74.27, VA 45.46→90.69.
  학습된 체크포인트(Qwen3-4B, Qwen2.5-3B, Llama-3.2-1B/3B + STAGE SFT)로 **추론만** 하면 되는 실험들임. 추가 학습 없음.

## 실험 1 — Constrained decoding (xgrammar) 비교

**목적**: "그냥 constrained decoding 쓰면 되지 왜 학습해?"라는 리뷰어 질문에 대한 답.
SLM에서 JSON 뽑을 때 가장 흔한 방법이라 비교 언급이 필수.

- **설정**: STAGE-Eval 851개 테스트셋에서 2×2 비교
  - {Qwen3-4B base, Qwen3-4B + STAGE SFT} × {자유 디코딩, xgrammar로 스키마 강제}
  - vLLM은 guided decoding backend로 xgrammar를 지원함 (`guided_json` 파라미터에 예제별 스키마 전달)
- **지표**: 기존 5개 그대로 (PFR / EMR / SCR / NR / VA) — `benchmark/evaluate.py` 재사용
- **기대 결과**: xgrammar는 구조 지표(PFR/SCR)만 올리고 값 지표(EMR/VA)는 못 올림 → 데이터 학습의 기여 입증.
  SFT+xgrammar 조합이 최고점이면 "상호보완" 논지 추가.
- **주의**: 예제마다 스키마가 달라서 문법 컴파일이 실패하는 케이스가 있을 수 있음.
  **먼저 30~50개로 파일럿** 돌려서 호환성 확인 후 전체 실행. 컴파일 실패 예제는 개수를 기록하고 제외 사유 명시.

## 실험 2 — BFCL (Berkeley Function Calling Leaderboard) 평가

**목적**: 워크숍 주제가 "SLM으로 에이전트 가능?"이므로 tool calling 성능 증거가 필요.
JSON 추출 학습이 함수 호출 능력으로 전이되는지(최소한 해치지 않는지) 확인.

- **설정**: BFCL v4의 **오프라인 카테고리만** (single / multiple / parallel function). 라이브 API·멀티턴 카테고리는 제외.
  - 도구: `pip install bfcl-eval` + vLLM 서버 (`--skip-server-setup`으로 기존 서버 연결 가능)
  - 비교: base vs STAGE-SFT — 우선 Qwen3-4B, 여유 되면 Llama-3.2-1B/3B도
- **지표**: AST 정확도. **반드시 분리 리포트**: (a) 함수 선택 정확도, (b) 인자 이름/스키마 유효성, (c) 인자 값 정확도.
  STAGE가 가르치는 건 (b)(c)이므로 합산 점수만 내면 효과가 희석됨.
- **리스크**: 함수 선택이 오히려 떨어질 수 있음 → 그 경우 "인자 구성 능력 개선"으로 범위 한정해서 보고.

## 실험 3 — 실세계 데이터셋 zero-shot 평가

**목적**: 합성 데이터 밖의 실제 문서에서도 작동함을 시연. 온디바이스 에이전트 시나리오(영수증→경비처리 JSON)로 워크숍 청중에 직관적.

- **데이터셋 1: CORD-v2** (우선)
  - HF `naver-clova-ix/cord-v2` (CC-BY-4.0), test 100개
  - 입력: `valid_line`의 OCR 텍스트를 읽기 순서로 이어붙여 report 구성 (OCR 엔진 불필요)
  - 정답: `ground_truth.gt_parse` (menu 배열 + sub_total/total 중첩 JSON)
  - gt_parse 구조에서 JSON 스키마를 유도해서 프롬프트에 제공 (기존 STAGE-Eval 입력 포맷과 동일하게)
- **데이터셋 2: ExtractBench** (여유 되면 / 카메라레디)
  - HF `llamaindex/ExtractBench` (Apache-2.0), 370문서 (short 252 / medium 98 / long 20)
  - `pdf` 필드에서 텍스트 추출 필요 (digital PDF는 pdftotext, 스캔본은 제외 가능) + `data_schema`, `expected_output` 제공됨
- **비교**: base vs STAGE-SFT (4개 모델 전부, zero-shot)
- **지표**: **VA 중심** (+ SCR/PFR). 스키마가 생소해서 EMR은 0 근처일 수 있으므로 EMR을 헤드라인으로 쓰지 말 것.
- **주의**: CORD 50~100개는 수동 검수 (OCR 읽기 순서/어노테이션 관례가 채점을 깨뜨릴 수 있음).

## 실행 순서 & 논문 반영

1. 실험 1 파일럿 (반나절) → 전체 (1일)
2. 실험 3 CORD (1일)
3. 실험 2 BFCL (1~1.5일, 하네스 설치가 변수)
4. 결과를 `overleaf-paper/neurips2026.tex`에 반영 (본문 6페이지 유지, 상세는 Appendix로)
5. 성능 보면서 방향 조정 — 결과가 약한 실험은 Appendix로 강등하거나 제외

## 인프라 메모

- 추론: vLLM, 1× H200 (설정은 논문 Appendix C 참조: temp 0.6, top-p 1.0, max_new 3100, max_len 8192, seed 42)
- 기존 코드: `benchmark/inference.py` (추론), `benchmark/evaluate.py` (채점), `src/utils/vllm_inference.py`
- 체크포인트 위치는 실험 시작 전 확인 필요
