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

## 실행 기록

### 2026-09-01 — 실험 1 xgrammar 파일럿 (STAGE-Eval 첫 50개)

- 환경: vLLM 0.10.2, xgrammar 0.1.23, Qwen3-4B base 및 `baseline-qwen3-4b-best` (STAGE SFT), H200 1장.
- 설정: 예제별 `json_schema`를 xgrammar guided decoding에 전달; temperature 0.6, top-p 1.0, max_new_tokens 3100, max_model_len 8192, seed 42.
- 호환성: 스키마 컴파일 실패 0/50 (두 모델 모두). Base 결과 중 2개는 생성 길이 상한에서 JSON이 닫히지 않아 파싱 실패.
- 결과 (`outputs/xgrammar_pilot/`):

| Model + decoding | PFR | EMR | SCR | NR | VA |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base + free decoding | 66.0 | 34.0 | 62.0 | 36.0 | 46.7 |
| Qwen3-4B base + xgrammar | 96.0 | 30.0 | 92.0 | 4.0 | 63.7 |
| Qwen3-4B STAGE SFT + free decoding | 100.0 | 62.0 | 100.0 | 0.0 | 83.0 |
| Qwen3-4B STAGE SFT + xgrammar | 100.0 | 60.0 | 96.0 | 0.0 | 83.1 |

- 자유 디코딩 대조군도 동일한 50개, 동일한 생성 설정으로 완료했다. xgrammar는 base에서 PFR/SCR을 각각 +30.0pt 높였지만, SFT 모델은 이미 자유 디코딩만으로 PFR 100.0/SCR 100.0이었다. 값 정확도(VA)는 base에서 +17.0pt, SFT에서 사실상 동일했다.
- 다음 단계: 851개 전체 테스트셋의 동일 2×2 실행. 파일럿에서 컴파일 실패는 0/50이었다.

### 2026-09-01 — xgrammar 전체 실행 호환성 사전검사

- 전체 851개 스키마를 vLLM 0.10.2의 xgrammar 지원 범위와 `xgrammar.Grammar.from_json_schema`로 사전검사했다.
- 53개는 vLLM이 지원하지 않는 스키마 기능(예: string `format`, `multipleOf`, `uniqueItems` 계열)을 포함해 xgrammar 조건에서 제외한다. 나머지 **798/851**개는 문법 컴파일을 통과했다.
- `benchmark/inference.py`는 이 제외를 `skip_reason`으로 결과 JSONL에 보존하고, `benchmark/evaluate.py`는 해당 행을 지표 분모에서 제외하도록 변경했다. 전체 실행은 base/SFT 모두 798개 유효 예제로 재개했다.

### 2026-09-01 — 실험 1 xgrammar 전체 (STAGE-Eval)

- 평가 대상은 사전검사를 통과한 **798/851**개이며, 호환 불가 53개는 두 조건 모두 동일하게 제외했다. PFR=1−no-output, NR=mean noise ratio이며, 나머지 지표는 기존 `benchmark/evaluate.py` 출력 그대로다.

| Model + decoding | PFR | EMR | SCR | NR | VA |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base + free decoding | 65.4 | 33.0 | 60.0 | 36.9 | 48.8 |
| Qwen3-4B base + xgrammar | 98.6 | 34.0 | 94.6 | 1.5 | 66.0 |
| Qwen3-4B STAGE SFT + free decoding | 99.6 | 63.7 | 97.6 | 1.4 | 84.7 |
| Qwen3-4B STAGE SFT + xgrammar | 99.4 | 59.3 | 95.4 | 0.8 | 82.6 |

- 네 조건 모두 xgrammar 호환 스키마 **798개**로 맞춰 평가했다. 자유 생성은 base에서 xgrammar 대비 VA가 −17.2pt, SFT에서 −2.1pt였다. xgrammar는 구조 준수를 크게 높이지만 SFT만으로도 이미 높은 구조·값 정확도를 보인다.
- 산출물: `outputs/xgrammar_full/`의 각 `*.jsonl` 및 `*_eval.xlsx`; 자유 생성 공통 분모 결과는 `*_free_xgrammar_compatible_eval.xlsx`다.

### 2026-09-01 — 실험 3 데이터 준비

- CORD-v2 test 100개를 `benchmark/prepare_cord.py`로 STAGE 입력 형식에 변환했다. `valid_line` OCR을 좌상단→우하단 읽기 순서로 정렬했고, singleton `menu` 객체는 배열로 정규화했다. 100개 정답 모두 유도 스키마 검증을 통과했다.
- ExtractBench는 `benchmark/prepare_extractbench.py`로 준비했다. 370개 중 디지털 텍스트가 추출된 244개를 보존했고, 무텍스트/스캔본 126개는 `outputs/extractbench/skipped.jsonl`에 제외 사유와 함께 기록했다. 보존된 244개 정답은 제공 스키마를 모두 통과했다.

### 2026-09-01 — 실험 3 CORD-v2 zero-shot (진행 중)

- CORD-v2 test 100개 8조건을 동일한 zero-shot 설정으로 완료했다. VA를 중심으로 보면, Qwen3-4B는 STAGE SFT 후 **66.0→71.7**로 개선되고 PFR/SCR도 **76.0/76.0→100.0/97.0**으로 높아졌다. Qwen2.5는 base 자체가 강한 반면 SFT 후 VA가 낮아졌고, Llama base는 대화 템플릿이 없는 base 체크포인트 특성상 거의 JSON을 만들지 못했다. 이는 모델별 전이 차이로 Appendix에 보고할 결과다.

| Model | PFR | EMR | SCR | NR | VA |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base | 76.0 | 32.0 | 76.0 | 24.0 | 66.0 |
| Qwen3-4B STAGE SFT | 100.0 | 22.0 | 97.0 | 1.1 | 71.7 |
| Qwen2.5-3B base | 98.0 | 23.0 | 96.0 | 2.8 | 70.3 |
| Qwen2.5-3B STAGE SFT | 100.0 | 7.0 | 98.0 | 0.7 | 60.6 |
| Llama-3.2-1B base | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| Llama-3.2-1B STAGE SFT | 100.0 | 1.0 | 98.0 | 0.7 | 32.5 |
| Llama-3.2-3B base | 9.0 | 0.0 | 0.0 | 98.2 | 0.0 |
| Llama-3.2-3B STAGE SFT | 100.0 | 2.0 | 83.0 | 7.1 | 40.6 |

- 산출물은 `outputs/cord_v2/`의 각 `*.jsonl`, `*.xlsx`, `*_eval.xlsx`다. BFCL-v4 오프라인 Qwen3 base/SFT 비교는 현재 실행 중이다.

### 2026-09-01 — 실험 2 BFCL-v4 오프라인 (Qwen3-4B)

- 카테고리: `simple_python` (400), `multiple` (200), `parallel` (200). BFCL 공식 생성·AST 체커를 사용했으며 live·multi-turn은 제외했다. 로컬 vLLM 서버를 같은 포트에 동시에 띄울 수 없어 base와 SFT를 순차 실행했다.

| Model | Official AST: simple | multiple | parallel | Function selection | Argument schema validity | Argument value accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B base | 96.0 | 94.0 | 92.0 | 98.3 | 99.8 | 88.5 |
| Qwen3-4B STAGE SFT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- SFT 모델은 BFCL의 Python-like bracket call 형식 대신 STAGE 학습 형식인 JSON 객체(`{"function_name": {...}}`)를 일관되게 생성했다. 예를 들어 첫 `simple_python` 항목에서 함수와 인자값 자체는 JSON으로 옳게 생성했지만 BFCL AST decoder가 요구하는 `[func_name(arg=value)]` 형식이 아니어서 0점이다. 따라서 이것은 함수 선택/인자 추론 자체의 0점이라기보다 **출력 프로토콜 비호환**이며, 본문 헤드라인에는 사용하지 않고 Appendix의 음성 결과 및 향후 tool-call template alignment 한계로 명시한다.
- 산출물: `outputs/bfcl/qwen3_4b_{base,sft}/result/`, 공식 `score/data_non_live.csv`, 세부 진단 `score/stage_diagnostics.json`.

### 2026-09-02 — 실험 2 BFCL JSON 프로토콜 재채점

- 공식 BFCL 점수는 Python call decoder를 그대로 사용해야 하므로 위의 SFT 0점이 공식 결과다. 다만 출력 프로토콜 효과와 함수·인자 추론을 분리하기 위해, 공식 AST checker는 바꾸지 않고 STAGE JSON 객체만 BFCL call AST로 변환하는 분석용 decoder를 `benchmark/rescore_bfcl_json.py`에 추가했다. 이 값은 리더보드 점수로 주장하지 않고 Appendix 진단으로만 사용한다.

| Category | Base official AST | SFT official AST | SFT JSON-native AST | Base function/schema/value | SFT function/schema/value |
|---|---:|---:|---:|---:|---:|
| simple_python | 96.0 | 0.0 | 86.0 | 99.0 / 99.5 / 87.5 | 98.8 / 100.0 / 91.3 |
| multiple | 94.0 | 0.0 | 73.5 | 97.0 / 100.0 / 85.0 | 98.0 / 100.0 / 87.4 |
| parallel | 92.0 | 0.0 | 56.5 | 98.3 / 100.0 / 90.7 | 68.0 / 98.9 / 90.2 |

- SFT는 single/multiple에서 함수 선택과 인자값 정확도를 유지 또는 향상시켰으나, parallel 호출 수·집합 구성은 약해졌다. 즉 JSON 추출 SFT가 function-call **출력 프로토콜**을 STAGE JSON으로 고정해 공식 BFCL 리더보드 형식에는 비호환이고, template alignment가 필요하다는 한계를 명확히 보여준다.
- 산출물: `outputs/bfcl/qwen3_4b_{base,sft}/score/json_decoder/`; 재현 스크립트: `benchmark/rescore_bfcl_json.py`.
- **실패 원인 분해 (SFT, JSON-native 디코더 기준, 800개 전수 확인)**:
  - 출력 포맷: 프롬프트가 지시한 `[func(arg=val)]` 형식을 따른 출력은 **0/800**. 90.5% (simple) / 96% (multiple) / 85.5% (parallel)가 `{"함수명": {"인자": 값}}` 사전 구조였고, 나머지는 `{"func_name":…, "params":…}`류 래퍼 변형이다. 디코더는 이 변형과 **중복 키**(같은 함수 반복 호출)를 모두 보존한다. `json.loads` 기본 동작은 중복 키를 마지막 것만 남기므로 parallel에서 35.0→56.5로 차이가 났다.
  - multiple 잔여 오류 32/200 = **과호출**: 함수 2개를 제시하면 정답은 1개인데 둘 다 호출. 추출 학습의 "스키마의 모든 필드를 채운다"가 "제시된 모든 함수를 채운다"로 전이된 것으로, 진짜 회귀이므로 finding으로 보고한다.
  - parallel 잔여 오류 87/200 = **JSON 객체의 표현 한계**: parallel 200개 전부가 같은 함수의 반복 호출을 요구하는데 JSON 객체는 키 중복이 불가하다. 모델은 키 반복, `play2`식 접미사, 인자 배열 병합 중 하나를 택했고 그 중 일부만 복원된다. 함수 선택 68.0은 이 표현 문제의 결과이고, 매칭된 호출의 인자 값 정확도(90.2)는 base(90.7)와 같다.
- **두 가지 읽기를 모두 유지한다.** (a) 공식 BFCL 규약(공용 Python-call 디코더) 기준 SFT는 0점이며 이 수치는 Appendix에 그대로 둔다. (b) BFCL 리더보드의 모든 모델은 자기 출력 포맷을 파싱하는 모델별 `decode_ast` 핸들러로 채점되므로, JSON-native 디코더 결과는 "모델 전용 핸들러" 규약 하의 정당한 수치다. 본문에 (b)를 쓸지는 아래 실험 2b 결과로 결정한다: 2b에서 parallel/과호출이 해소되면 2b가 본문 헤드라인이 되고 (a)(b)는 프로토콜 분석으로 Appendix에 간다.
- 분석·재채점 수행: 2026-09-02, pod `~/work/sunghee/text-to-json` (bfcl_eval은 `~/work/sunghee/venv`에 설치).

### 2026-09-02 — 실험 3 ExtractBench zero-shot (완료)

- ExtractBench의 디지털 텍스트 244개를 대상으로, 등록된 추론 설정(`max_model_len=8192`, `max_new_tokens=3100`)을 문서 절단 없이 적용했다. Qwen3와 Llama tokenizer 모두에서 전체 프롬프트와 3,100-token 생성 예산이 들어가는 공통 표본은 **27개 short 문서**였다. 나머지 217개는 입력을 잘라 성능을 부풀리지 않고, 모델별 프롬프트 길이와 `context_budget_exceeded` 사유를 `outputs/extractbench/context_skipped.jsonl`에 보존했다.
- 비교: Qwen3-4B, Qwen2.5-3B, Llama-3.2-1B/3B 각각 base vs STAGE SFT; temp 0.6, top-p 1.0, seed 42. PFR=1−no-output, NR=mean noise ratio이다.

| Model | PFR | EMR | SCR | NR | VA |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base | 51.9 | 0.0 | 51.9 | 48.1 | 34.3 |
| Qwen3-4B STAGE SFT | 70.4 | 0.0 | 66.7 | 31.7 | 37.9 |
| Qwen2.5-3B base | 55.6 | 0.0 | 51.9 | 47.5 | 23.8 |
| Qwen2.5-3B STAGE SFT | 70.4 | 0.0 | 63.0 | 33.9 | 32.0 |
| Llama-3.2-1B base | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| Llama-3.2-1B STAGE SFT | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| Llama-3.2-3B base | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| Llama-3.2-3B STAGE SFT | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |

- Qwen 계열에서는 STAGE SFT가 PFR/SCR/VA를 일관되게 높였다(Qwen3 VA +3.6pt, Qwen2.5 VA +8.2pt). 이 문서 길이·스키마 조합에서는 EMR은 모든 모델에서 0이므로 헤드라인 지표로 쓰지 않는다. Llama는 CORD와 마찬가지로 이 프롬프트 조건에서 유효 JSON을 만들지 못한 모델별 음성 결과이며 Appendix에 보고한다.
- 산출물: 공통 입력 `benchmark/data/extractbench_context8192.jsonl`, 재현 가능한 필터 `benchmark/filter_extractbench_context.py`, 각 모델의 `outputs/extractbench/{model}.jsonl`, `{model}.xlsx`, `{model}_eval.xlsx`.

## 다음 실행 — 실험 2b/2c BFCL 후속 (에이전트 runbook)

> 아래는 새 세션의 에이전트가 **추가 판단 없이** 실행할 수 있게 쓴 절차다. 순서대로 실행하고, 각 단계의 산출물 경로와 완료 조건을 확인한 뒤 결과를 이 문서 `## 실행 기록`에 추기한다.
> 우선순위: **2b > 2c > 2d**. 2b만 끝나도 논문 반영 가능. 총 GPU 예산 약 3~4시간 (H200 1장).

### 0. 환경 (5분)

- 작업 위치: k8s 예약 pod의 `~/work/sunghee/text-to-json` (= `/mnt/ddn/prod-runs/interns/sunghee/text-to-json`). 먼저 `git pull`로 origin/master와 맞춘다.
- Python: `~/work/sunghee/venv/bin/python` (vLLM 0.10.2, xgrammar 0.1.23, `bfcl_eval` 설치됨). 없으면 `pip install bfcl-eval`.
- GPU: `nvidia-smi`로 메모리가 비어 있는 카드를 고르고 `CUDA_VISIBLE_DEVICES`로 지정한다. 다른 사람의 작업이 돌고 있는 카드는 쓰지 않는다.
- 체크포인트 (모두 HF public): STAGE SFT `boradorish/baseline-qwen3-4b-best`, base `Qwen/Qwen3-4B`, Glaive SFT `boradorish/baseline-qwen3-4b-glaive`, Llama SFT `boradorish/llama3-1B-sft`, `boradorish/llama3-3B-sft`.
- 기존 BFCL 원출력·공식 점수는 `outputs/bfcl/qwen3_4b_{base,sft}/`에 있다. 재실행하지 않는다.

### 1. `rescore_bfcl_json.py`에 CLI 추가 (15분, GPU 불필요)

현재 스크립트는 `qwen3_4b_sft`/`qwen3_4b_base` 두 run을 하드코딩해 재채점·출력한다. 다음으로 일반화한다.

- `--runs RUN [RUN ...]`: 재채점할 `outputs/bfcl/<run>` 목록. `--baseline RUN` (기본 `qwen3_4b_base`): 비교 열로 쓸 공식 점수의 run.
- 출력 위치·형식은 유지: `outputs/bfcl/<run>/score/json_decoder/BFCL_v4_{cat}_score.json` + `stage_diagnostics.json`.
- 디코더가 받아야 하는 출력 형태 (이미 구현됨, 회귀 테스트로 확인): `{"f": {…}}`, 중복 키 `{"f": {…}, "f": {…}}`, `{"name": "f", "arguments": {…}}`, `{"calls": [{"name": …, "arguments": …}, …]}`, 단일 함수 프롬프트에서의 flat 인자 사전, 코드펜스 감싼 JSON, 파이썬 튜플 리터럴.
- 완료 조건: `--runs qwen3_4b_sft`로 실행했을 때 86.0 / 73.5 / 56.5가 재현된다.

### 2. 실험 2b — STAGE 네이티브 프롬프트로 BFCL 오프라인 재실행 (핵심, GPU 약 1.5시간)

**가설**: STAGE SFT 모델은 "스키마를 주면 그 스키마대로 채우는" 모델이다. tool call을 STAGE 입력 형식(보고서 + JSON Schema)으로 제시하면 (i) parallel 반복 호출을 배열로 표현할 수 있고, (ii) 스키마에 `minItems: 1`만 있으므로 과호출 유인이 사라진다.

새 스크립트 `benchmark/run_bfcl_stage_prompt.py`:

1. 입력: `bfcl_eval.constants.eval_config.PROMPT_PATH / "BFCL_v4_{cat}.json"` (cat ∈ simple_python, multiple, parallel). 각 항목의 `question[0][-1]["content"]`(사용자 질문)과 `function`(함수 목록)을 쓴다.
2. 프롬프트 (STAGE-Eval과 동일 규약): system = `prompt/infer_SYSTEM_prompt.txt`, user =
   ```
   Extract the tool calls needed to fulfil the request below as JSON that conforms to the schema.

   === Report ===
   {사용자 질문}

   === Available functions ===
   {function 목록을 JSON으로 pretty-print}

   === JSON Schema ===
   {아래 3의 스키마}
   ```
3. 스키마 생성 (`bfcl_function_to_call_schema(functions)`):
   ```json
   {"type":"object","additionalProperties":false,"required":["calls"],
    "properties":{"calls":{"type":"array","minItems":1,"items":{"oneOf":[ <함수별 항목> ]}}}}
   ```
   함수별 항목: `{"type":"object","additionalProperties":false,"required":["name","arguments"],"properties":{"name":{"const":"<fn.name>"},"arguments":<fn.parameters 변환>}}`.
   BFCL 파라미터 타입 변환: `dict→object`, `float→number`, `tuple→array`, `any→{}`(제약 없음), `integer/string/boolean/array`는 그대로. `properties`·`required`·`description`·`enum`·`items`는 유지하고, 그 외 키(`default` 등)는 제거한다. 변환 후 `jsonschema.Draft202012Validator.check_schema`로 검증하고, 실패하면 그 항목은 `skip_reason`과 함께 기록하고 제외한다(실험 1과 같은 규약).
4. 추론: `src/utils/vllm_inference.py`의 `VllmModel` / `build_chat_prompts` / `generate_texts` 재사용. temp 0.6, top-p 1.0, seed 42, max_model_len 8192, max_new_tokens 3100 (규칙 기본값). 옵션 `--guided-json`을 주면 3의 스키마를 xgrammar guided decoding으로 강제한다(실험 1과 연결되는 조건).
5. 출력: BFCL 레이아웃 그대로 `outputs/bfcl/<run>/result/qwen3-4b/non_live/BFCL_v4_{cat}_result.json`, 각 줄 `{"id": …, "result": <모델 원문 문자열>}`. `</think>` 이후만 남기고, 코드펜스는 벗긴다. 이렇게 하면 1의 재채점기가 그대로 동작한다.
6. 실행 run 4개 (모두 같은 프롬프트라 공정 비교):
   - `qwen3_4b_sft_stageprompt`, `qwen3_4b_base_stageprompt` (자유 디코딩)
   - `qwen3_4b_sft_stageprompt_xgr`, `qwen3_4b_base_stageprompt_xgr` (`--guided-json`)
7. 채점: `python benchmark/rescore_bfcl_json.py --runs <위 4개>`.
8. 리포트 (이 문서에 추기): 카테고리별 AST accuracy + 함수 선택 / 인자 스키마 유효 / 인자 값 정확도 + 오류 유형 분해. 기존 표(공식 base, 공식 SFT 0, JSON-native SFT)와 한 표에 놓는다.
9. **완료·판정 조건**: (a) SFT parallel 함수 선택이 68.0 → **90 이상**, (b) multiple `wrong_count`가 32 → **10 이하**로 떨어지면 가설 성립. 그러면 2b의 SFT vs base(동일 프롬프트) 비교가 본문 Table, 공식 0점과 JSON-native 재채점은 Appendix "output-protocol analysis"로 간다. 미달이면 2b도 Appendix로 내리고, 본문은 "인자 구성 능력은 유지·향상, 호출 집합 구성은 회귀"로 범위를 한정한다.

### 3. 실험 2c — Glaive-SFT 베이스라인 BFCL (GPU 약 1시간)

- 논문 Table의 baseline 중 `boradorish/baseline-qwen3-4b-glaive`는 함수 호출 데이터(Glaive)로 학습한 모델이다. STAGE-Eval에서는 STAGE가 크게 앞선다(EMR 70.28 vs 74.27, VA 46.89 vs 90.69).
- 실행: (1) 공식 러너 `python benchmark/run_bfcl_local.py --model-path boradorish/baseline-qwen3-4b-glaive --run-name qwen3_4b_glaive` → 공식 AST + `summarize_bfcl.py`. (2) 2b 스크립트로 `qwen3_4b_glaive_stageprompt`.
- 판정: BFCL 인자 값 정확도에서 STAGE ≥ Glaive이면 "함수 호출 데이터 없이 인자 구성 능력이 전이된다"를 본문 한 문장으로 쓴다. Glaive가 앞서면 Appendix 비교표로만 둔다.

### 4. 실험 2d — Llama-3.2-1B/3B SFT (선택, GPU 약 1시간)

- 2b 스크립트로 `llama3_2_1b_sft_stageprompt`, `llama3_2_3b_sft_stageprompt`. base Llama는 CORD/ExtractBench와 같이 JSON을 거의 못 만들었으므로 base 비교는 생략 가능(그 사실을 명시).
- 결과 파일 경로의 `qwen3-4b` 디렉토리명은 `rescore_bfcl_json.py`가 기대하는 고정 레이아웃이므로 모델이 달라도 그대로 둔다(스크립트에 `--model-dir` 옵션을 추가하면 바꿔도 된다).

### 5. 논문 반영 (2b 결과 확정 후, 반나절)

- 본문 §Results에 "Tool-calling transfer (BFCL-v4 offline)" 소절 1개 + Table 1개(6페이지 유지). 반드시 함수 선택 / 인자 스키마 유효성 / 인자 값 정확도를 분리해 보고한다.
- Appendix: (i) 공식 규약 0점과 원인(출력 프로토콜 고정, 지시 포맷 준수 0/800), (ii) JSON-native 디코더 정의와 결과, (iii) 과호출·parallel 표현 한계 분석. "narrow SFT 후 SLM의 format lock-in"을 limitation 겸 discussion으로 한 단락.
- 하지 말 것: 추가 학습(BFCL 포맷 혼합 SFT)은 마감상 제외. 공식 0점을 표에서 지우지 말 것.

## 실행 기록 (후속)

### 2026-09-02 — 실험 2b STAGE-native BFCL 오프라인 (완료; Appendix 음성 결과)

- `benchmark/run_bfcl_stage_prompt.py`로 BFCL-v4 offline 800개(`simple_python` 400, `multiple` 200, `parallel` 200)를 STAGE 입력 규약으로 다시 실행했다. 출력 스키마는 `{"calls": [{"name", "arguments"}, ...]}`이며, 함수별 parameter schema를 `oneOf`로 보존한다. 자유 생성 및 xgrammar-guided decoding을 각각 base/STAGE-SFT에 적용했다. 모든 조건은 H200 1장, temperature 0.6, top-p 1.0, seed 42, max_new_tokens 3100, max_model_len 8192이다.
- JSON-native AST는 분석용 decoder 결과이며 공식 BFCL 리더보드 점수가 아니다. 비교를 위해 기존 base의 공식 AST도 함께 제시한다.

| Condition | simple AST | multiple AST | parallel AST |
|---|---:|---:|---:|
| Qwen3-4B base, official BFCL prompt | 96.0 | 94.0 | 92.0 |
| Qwen3-4B base, STAGE prompt | 94.2 | 94.5 | 92.5 |
| Qwen3-4B base, STAGE prompt + xgrammar | 89.0 | 87.0 | 88.0 |
| Qwen3-4B STAGE SFT, STAGE prompt | 79.8 | 5.0 | 66.5 |
| Qwen3-4B STAGE SFT, STAGE prompt + xgrammar | 79.2 | 5.0 | 65.0 |

- SFT 자유 생성의 함수 선택 / 인자 스키마 유효 / 인자 값 정확도는 simple **99.5 / 100.0 / 88.3**, multiple **98.5 / 99.5 / 82.5**, parallel **84.4 / 100.0 / 87.7**이었다. xgrammar도 각각 **99.8 / 100.0 / 87.9**, **99.5 / 100.0 / 82.7**, **83.3 / 100.0 / 87.6**으로 호출 집합 문제를 해소하지 못했다.
- 가설 판정: **불성립**. 목표였던 parallel 함수 선택 90 이상에는 미달(84.4), multiple `wrong_count`도 185/200(가이드 조건 187/200)으로 목표 10 이하에 크게 못 미쳤다. 모든 후보 함수를 schema에 나열한 것이 STAGE-SFT 모델에 과호출 유인을 주는 것으로 해석된다. base는 같은 STAGE 프롬프트에서 강했으므로, 이 결과는 구현 문제가 아니라 STAGE-SFT의 call-set 일반화 한계다. 본문 헤드라인에는 쓰지 않고 Appendix의 protocol/prompt-alignment 음성 결과로 둔다.
- 산출물: `outputs/bfcl/qwen3_4b_{base,sft}_stageprompt*/result/`, `score/json_decoder/`; 재현 스크립트: `benchmark/run_bfcl_stage_prompt.py`, `benchmark/rescore_bfcl_json.py`.

### 2026-09-02 — 실험 2c Glaive-SFT BFCL 비교 (완료)

- 함수 호출 데이터로 학습한 공개 Glaive LoRA(rank 64)를 Qwen3-4B base에 장착했다. `benchmark/run_bfcl_local.py`는 adapter metadata에서 LoRA rank를 읽고, base checkpoint로 vLLM을 띄운 뒤 `bfcl-adapter` 요청 모델로 adapter를 선택하도록 보완했다. BFCL이 자식 vLLM 프로세스를 띄울 때도 Transformers 5 tokenizer 호환 패치를 적용하는 `benchmark/vllm` 래퍼를 사용했다.

| Condition | simple AST | multiple AST | parallel AST | Function / schema / value (simple) | (multiple) | (parallel) |
|---|---:|---:|---:|---:|---:|---:|
| Glaive, official BFCL prompt | 93.5 | 7.0 | 70.0 | 99.2 / 99.7 / 88.2 | 7.0 / 100.0 / 84.4 | 76.5 / 100.0 / 92.8 |
| Glaive, STAGE prompt (JSON-native) | 91.8 | 87.5 | 73.5 | 99.0 / 100.0 / 86.2 | 98.0 / 100.0 / 85.6 | 78.5 / 99.5 / 92.0 |

- STAGE 프롬프트는 Glaive의 multiple AST를 **+80.5pt** 높였지만, parallel AST는 73.5에 머물렀다. 공식 프롬프트 multiple의 7.0은 185건의 Python-call AST decode 실패가 주원인이며, STAGE 출력 형식과 공식 포맷의 alignment가 성능을 좌우함을 재확인한다. 따라서 “STAGE가 Glaive보다 BFCL 인자값 정확도에서 앞선다”는 주장은 이 결과로 뒷받침되지 않으며, Glaive 비교 역시 Appendix 표로 한정한다.
- 산출물: `outputs/bfcl/qwen3_4b_glaive_{official,stageprompt}/result/`, 공식 `score/`, JSON-native `score/json_decoder/`.

### 2026-09-02 — 실험 2b/2c 결과 해석 (원출력 전수 분석)

- **multiple 과호출은 규칙적이다.** SFT는 제시된 함수 N개에 대해 정확히 N개 호출을 냈다(2/2: 72, 3/3: 77, 4/4: 35 = 184/200). 정답 함수는 190건 실패 중 187건에 포함돼 있고, 불필요한 호출의 인자도 311/332건이 그럴듯하게 전부 채워져 있다(빈 값·placeholder 아님). 같은 프롬프트에서 base는 189/200, Glaive는 175/200이 1개 호출이다. 즉 `oneOf` 변형 목록을 "채워야 할 필드 목록"으로 취급하는 **coverage bias**이며, 프롬프트 결함이 아니다.
- **parallel은 선택 문제가 아니다.** BFCL parallel은 800개 중 확인된 전부가 함수 1개만 제시한다. SFT의 wrong_count 40건은 전부 정답보다 **적게** 호출(gold 3→1: 11, 2→1: 10, 4→2: 9 …)한 경우로, 여러 요청을 한 호출의 배열 인자로 병합하는 습관이다.
- **simple 79.8 (STAGE 프롬프트) < 86.0 (공식 프롬프트+JSON 디코더)**: 새로 실패한 38건은 전부 값 오류(문자열 16, 리스트 13, 단위 스케일 등 7)다. `=== Available functions ===`에 함수 JSON을 통째로 넣은 것이 스키마와 중복돼 값 주의력을 떨어뜨린 것으로 보인다(2f에서 ablation).
- **학습 데이터에서의 원인**: STAGE-Eval 1,000개 스키마의 객체 3,988개 중 3,807개(95.5%)가 모든 속성을 `required`로 두고, `oneOf`/`anyOf`는 54개(1.4%)뿐이다. "여럿 중 하나를 고른다", "요청에 따라 호출 개수가 달라진다"를 STAGE 데이터는 가르치지 않는다. 이것이 인자 구성은 전이되고(값 정확도 base 이상, 스키마 유효 100) 호출 집합 구성은 전이되지 않는 이유다.
- **Glaive 2c의 읽기**: Glaive는 함수 호출 데이터로 학습했는데도 공식 프롬프트 multiple이 7.0(Python-call 디코드 실패 185건)이다. 즉 "SFT 후 출력 프로토콜 고정"은 STAGE 고유 현상이 아니라 narrow SFT를 거친 4B 모델의 일반적 현상이며, 논문에서는 이 점을 STAGE 한계가 아닌 SLM 일반 현상으로 서술할 수 있다.

### 후속 우선순위 판정

- runbook의 실험 2d(Llama-3.2 1B/3B)는 명시적으로 **선택** 항목이다. 2b와 2c의 필수 실행·분석을 완료했으며, 2b가 본문 가설을 지지하지 않아 추가 Llama 실행은 동일한 음성 결과를 늘리기보다 논문 Appendix 정리보다 우선하지 않는다. 추가 학습은 수행하지 않았다.

## 다음 실행 (2차) — 실험 2e/2f (에이전트 runbook)

> 2b에서 확인된 병목은 **호출 집합 구성**(어느 함수를 몇 번)이고, **인자 구성**은 STAGE-SFT가 base 이상이다. 따라서 다음 실험은 두 단계를 분리해 STAGE-SFT를 "인자 구성기"로 쓰는 배치를 검증한다. 추가 학습 없음. 우선순위 **2f(15분) → 2e(약 1시간)**. 2f 결과로 2e 프롬프트의 함수 목록 형식을 정한다.

### 2f. 프롬프트 ablation (GPU 약 15분, simple/multiple 각 200개 서브셋 가능)

`benchmark/run_bfcl_stage_prompt.py`에 옵션 두 개를 추가해 SFT만 실행한다.

- `--no-function-dump`: `=== Available functions ===` 블록을 제거한다(스키마의 `description`이 같은 정보를 담음). 판정: simple AST가 79.8에서 86.0 근방으로 회복되면 이후 실험은 모두 이 옵션으로 실행.
- `--one-shot`: user 메시지 앞에 고정 예시 1개(함수 3개 제시, 호출 1개만 담긴 `{"calls": [...]}` 정답)를 붙인다. BFCL 데이터가 아닌 자체 작성 예시를 쓴다. 판정: multiple `wrong_count`가 185에서 유의미하게(≤100) 줄면 coverage bias가 in-context로 완화됨을 Appendix에 기록. 줄지 않으면 "지시·예시로 교정 불가 → format lock-in"의 추가 근거.
- run 이름: `qwen3_4b_sft_stageprompt_nodump`, `qwen3_4b_sft_stageprompt_oneshot`. 채점은 `rescore_bfcl_json.py --runs …`.

### 2e. Select-then-fill: 계획 단계와 인자 채움 단계 분리 (핵심, GPU 약 1시간)

새 스크립트 `benchmark/run_bfcl_select_fill.py`. 두 pass 모두 STAGE 입력 규약(system=`prompt/infer_SYSTEM_prompt.txt`, `=== Report ===` + `=== JSON Schema ===`)이며 카테고리 정보는 모델에 주지 않는다.

**Pass 1 — 계획.** 사용자 질문을 원자 요청으로 분해하고 각 요청에 함수를 배정한다. 스키마:
```json
{"type":"object","additionalProperties":false,"required":["plan"],
 "properties":{"plan":{"type":"array","minItems":1,"items":{
   "type":"object","additionalProperties":false,"required":["request","function"],
   "properties":{"request":{"type":"string","description":"One atomic sub-request from the user message, copied or minimally paraphrased"},
                 "function":{"type":"string","enum":[<제시된 함수명들>]}}}}}}
```
- Report에는 질문과 함수별 `name`+`description` 한 줄씩만 넣는다(파라미터 스키마 제외). `enum` 단일 선택은 STAGE-Eval에 2,846회 등장하는 in-distribution 구조다.
- 기록: `plan` 길이, 함수 집합. 지표: **호출 집합 정확도** = 예측 (함수, 개수) multiset이 gold와 일치하는 비율. 2b의 함수 선택 지표는 재현율만 봐서 과호출을 못 잡았으므로 이번엔 정밀도 포함.

**Pass 2 — 인자 채움.** plan의 각 항목마다 해당 함수 **하나**의 스키마로 호출을 만든다:
```json
{"type":"object","additionalProperties":false,"required":["name","arguments"],
 "properties":{"name":{"const":"<fn>"},"arguments":<fn.parameters 변환(2b와 동일 규칙)>}}
```
- Report = 원래 질문 전체 + `Sub-request: <plan.request>`. 원문을 함께 주는 이유는 값(단위·날짜·지명)이 질문 전체에 흩어져 있기 때문이다.
- 배치: pass 1 결과를 모아 pass 2를 한 번에 배치 추론(호출 수 합계 ≈ 1,140).

**조립과 채점.** 항목별로 pass 2 결과를 `{"calls": [...]}`로 합쳐 BFCL 레이아웃 `outputs/bfcl/<run>/result/qwen3-4b/non_live/BFCL_v4_{cat}_result.json`에 저장하면 `rescore_bfcl_json.py`가 그대로 채점한다.

**실행 조건 4개** (모두 Qwen3-4B):
1. `qwen3_4b_sft_selectfill` — pass 1·2 모두 STAGE-SFT (단일 모델 배치).
2. `qwen3_4b_base_selectfill` — 둘 다 base (동일 절차의 공정 대조).
3. `qwen3_4b_sft_oraclefill` — pass 1을 gold (함수, 개수)로 대체하고 pass 2만 SFT. **인자 구성 능력의 상한**이며, 이 값이 논문에서 "STAGE가 가르친 것"을 가장 깨끗하게 보여준다. base도 같은 조건으로(`qwen3_4b_base_oraclefill`).
4. (선택) `qwen3_4b_mixed_selectfill` — pass 1 base, pass 2 SFT. 라우터/채움기를 분리하는 실제 에이전트 배치.

**판정 조건.**
- (a) `oraclefill`에서 SFT의 인자 값 정확도·스키마 유효성이 base 이상이고 세 카테고리 AST가 base 대비 −5pt 이내 → 본문 Table: "given the routed function, STAGE-SFT constructs arguments at least as accurately as base".
- (b) `selectfill` SFT의 multiple·parallel AST가 각각 **75 이상** → 본문에 "plan-then-fill 배치로 단일 4B 모델 tool calling 가능" 추가. 미달이면 (a)만 본문, selectfill은 Appendix.
- (c) 어떤 경우에도 2b 표(공식 0, JSON-native, STAGE 프롬프트)와 coverage bias 분석은 Appendix에 유지한다.

### 2g. (이번 마감에는 하지 않음) 데이터 측 교정

STAGE 생성 파이프라인에 (i) `oneOf` 중 하나만 채워지는 스키마, (ii) 요청 수에 따라 길이가 달라지는 배열, (iii) 선택 안 된 필드를 비워두는 예시를 소량 섞어 재학습하면 coverage bias가 교정되는지 확인. 카메라레디 또는 후속 논문용. 논문 Limitations/Future work에 한 문장으로 적는다.

### 논문 반영 지침 (2e 결과 확정 후)

- 본문 BFCL 소절의 주장을 **"인자 구성은 전이되고, 호출 집합 계획은 전이되지 않는다"**로 잡는다. 근거 수치: 인자 값 정확도 91.3/87.4 vs base 87.5/85.0(simple/multiple, JSON-native), 스키마 유효 100; 과호출 184/200 N-for-N; STAGE 데이터의 all-required 95.5%, oneOf 1.4%.
- 2e의 oraclefill(과 selectfill)이 처방이 된다. 워크숍 청중에게는 "SLM을 라우터가 아닌 인자 구성기로 배치"가 실용적 메시지다.
- Glaive의 공식 프롬프트 multiple 7.0을 인용해 프로토콜 고정이 narrow-SFT SLM의 일반 현상임을 한 문장으로 언급한다.

## 실행 기록 (2차 후속)

### 2026-09-02 — 실험 2f 프롬프트 ablation (완료; 200개 simple/multiple 서브셋)

- Qwen3-4B STAGE-SFT에 대해 `simple_python`과 `multiple` 각 첫 200개를 같은 생성 설정으로 실행했다. JSON-native AST는 분석용 decoder 결과다.

| Condition | simple AST | multiple AST | multiple wrong_count |
|---|---:|---:|---:|
| 2b STAGE prompt (전체) | 79.8 | 5.0 | 185/200 |
| no function dump | 81.0 | 9.0 | 175/200 |
| one-shot (3 functions → 1 call) | 82.5 | 11.0 | 168/200 |

- `--no-function-dump`은 simple 값을 일부 개선했지만 86.0 근방 회복 기준에는 못 미쳤다. `--one-shot`도 coverage bias를 17건만 줄였고 목표(≤100)에 크게 미달했다. 따라서 함수 목록의 중복이나 단일 in-context 예시로는 format lock-in을 교정할 수 없으며, 2e에서는 명시적 planner를 사용했다.
- 산출물: `outputs/bfcl/qwen3_4b_sft_stageprompt_{nodump,oneshot}/score/json_decoder/`.

### 2026-09-02 — 실험 2e Select-then-fill (완료)

- `benchmark/run_bfcl_select_fill.py`로 (1) 함수·호출 수 계획, (2) 라우팅된 단일 함수의 인자 채움을 분리했다. Planner는 함수명 enum과 함수별 설명만, filler는 원문 전체와 sub-request 및 단일 함수 schema만 받는다. `oraclefill`은 gold 함수 멀티셋을 사용해 라우팅 오류를 제거했다. 모든 조건은 800개, H200 1장, temperature 0.6, top-p 1.0, seed 42, max_new_tokens 3100, max_model_len 8192이다.

| Condition | simple AST | multiple AST | parallel AST | Call-set exact (simple / multiple / parallel) |
|---|---:|---:|---:|---:|
| STAGE-SFT selectfill | 76.2 | 79.0 | 41.5 | 99.2 / 94.0 / 57.5 |
| Qwen3-4B base selectfill | 87.5 | 87.0 | 81.0 | 98.2 / 95.0 / 92.0 |
| STAGE-SFT oraclefill | 78.5 | 85.0 | 0.5 | 100 / 100 / 100 |
| Qwen3-4B base oraclefill | 89.8 | 92.5 | 0.0 | 100 / 100 / 100 |

| Oraclefill argument metric | simple: SFT / base | multiple: SFT / base | parallel: SFT / base |
|---|---:|---:|---:|
| Function selection | 99.5 / 100.0 | 100.0 / 99.5 | 99.6 / 70.2 |
| Schema validity | 100.0 / 100.0 | 100.0 / 100.0 | 99.6 / 100.0 |
| Value accuracy | **89.5 / 84.8** | **90.7 / 83.4** | 53.2 / 60.4 |

- **판정:** SFT selectfill은 multiple을 79.0으로 끌어올리고 `wrong_count`를 9/200으로 낮췄지만, parallel 41.5가 기준 75에 미달했다. 따라서 단일 4B plan-then-fill 배치를 본문 성과로 주장하지 않는다. Oracle routing에서는 SFT가 simple/multiple 인자값 정확도에서 base를 각각 +4.7pt/+7.3pt 앞섰고 스키마 유효성은 100%였다. 하지만 AST는 base 대비 −11.3pt/−7.5pt라 runbook의 −5pt 조건을 만족하지 못한다. 본문은 "인자값 정확도 향상"을 제한적으로 서술하고, 전체 표는 Appendix에 둔다.
- **parallel oraclefill 한계:** gold 함수·개수만 주어 atomic sub-request를 복원할 수 없어 각 반복 호출의 filler에는 같은 전체 질문을 넣었다. 호출 수는 맞아도 어느 값이 어느 반복 호출에 대응하는지 구분하지 못해 두 모델 모두 AST가 0에 가까웠다. 이는 STAGE의 인자 구성 회귀로 해석하지 않고, 계획 표현에 slot-level sub-request/argument hints가 필요하다는 설계 한계로 기록한다.
- 산출물: `outputs/bfcl/qwen3_4b_{sft,base}_{selectfill,oraclefill}/score/json_decoder/` 및 `plan_diagnostics.json`; 원출력은 재현용으로 로컬 `result/`에만 보존한다.

### 2026-09-02 — Base vs. STAGE SFT few-shot prompting (완료)

- 동일한 BFCL `simple_python`/`multiple` 각 첫 200개에, 비벤치마크 고정 예시를 프롬프트 앞에 **0·1·3·5개** 넣었다. 예시는 필요한 함수만 고르기, 두 함수 고르기, 같은 함수 반복 호출을 포함한다. 두 모델 모두 같은 예시·순서·생성 설정을 사용했다. 따라서 모델 가중치를 바꾸지 않는 in-context few-shot 비교다.

| Prompt examples | Base simple AST | STAGE simple AST | Base multiple AST | STAGE multiple AST | Base / STAGE multiple wrong_count |
|---:|---:|---:|---:|---:|---:|
| 0 | 93.5 | 82.0 | 95.0 | 5.0 | 5 / 185 |
| 1 | 92.0 | 82.5 | 94.5 | 11.5 | 2 / 168 |
| 3 | 93.0 | 82.5 | 95.0 | 12.5 | 3 / 166 |
| 5 | 92.5 | 83.5 | 93.8 | 12.5 | 2 / 167 |

- **판정:** base는 예시 수와 무관하게 simple 92.0–93.5, multiple 93.8–95.0으로 안정적이다. STAGE SFT는 1개 예시만으로 multiple AST가 +6.5pt(5.0→11.5), 3개에서 +7.5pt(12.5) 개선되고 과호출은 185→166으로 감소했다. 그러나 3→5개에서 더 좋아지지 않았고 base 수준에는 크게 미달한다. 즉 prompt few-shot은 coverage bias를 **부분 완화**하지만, STAGE SFT의 tool-selection format lock-in을 해결하지는 못한다.
- 시각화: `outputs/bfcl/fewshot_prompting.png`; 수치 CSV: `outputs/bfcl/fewshot_prompting_summary.csv`. 재현: `benchmark/run_bfcl_stage_prompt.py --few-shot-count {0,1,3,5}`, `benchmark/plot_bfcl_fewshot.py`.

### 2026-09-02 — 256-example continued tool-call SFT (완료; 부분 개선, 전체 해법 아님)

- 프롬프트에 예시를 넣는 few-shot prompting이 아니라, 새로 만든 **256개**의 비-BFCL 도구 호출 예시로 LoRA(rank 16)를 3회 추가 학습했다. 각 예시는 필요한 호출 1–3개, 무관한 도구 1–3개, 같은 도구의 반복 호출을 섞고 `{"calls": [{"name", "arguments"}]}`만 정답으로 삼았다. BFCL 이름·문항은 학습 데이터에 쓰지 않았고, 별도 64개를 검증용으로 분리했다. base와 기존 STAGE SFT에 완전히 같은 데이터·학습 설정을 적용하고, STAGE-native BFCL 800개로 평가했다.

| Condition | simple AST | multiple AST | parallel AST | multiple call-count errors |
|---|---:|---:|---:|---:|
| Base | 94.2 | 94.5 | 92.5 | 5 / 200 |
| Base + 256-example SFT | 90.0 | 80.0 | 85.5 | 22 / 200 |
| STAGE SFT | 79.8 | 5.0 | 66.5 | 185 / 200 |
| STAGE SFT + 256-example SFT | 82.8 | **52.0** | 63.0 | **76 / 200** |

- **쉽게 말하면:** 적은 수의 학습 예시는 STAGE가 "나열된 도구를 전부 호출하는" 문제를 꽤 줄였다. 여러 호출 문제는 5.0%에서 52.0%로 올랐고, 호출 개수를 틀린 경우는 185개에서 76개로 109개 줄었다. 하지만 base에 같은 학습을 하면 모든 범주가 떨어졌고, STAGE의 parallel은 오히려 66.5→63.0으로 하락했다. 따라서 이 결과는 "작은 추가 학습으로 STAGE의 여러-도구 선택을 부분 복구할 수 있음"이지, base보다 잘하는 범용 해법은 아니다.
- 해석: 데이터가 짧고 반복되는 합성 도구/인자 조합이라 STAGE의 coverage bias에는 신호를 줬지만, BFCL의 긴 함수 설명·복잡한 인자·parallel의 같은 함수 반복에는 충분히 일반화하지 못했다. 다음 SFT는 단순 예시 수를 늘리기보다, 어려운 여러-호출/parallel 구조와 BFCL과 다른 복잡한 JSON schema를 의도적으로 더 포함해야 한다.
- 산출물: `outputs/bfcl/toolfew_sft.{png,csv}`, 새 run의 `score/json_decoder/`; 재현: `benchmark/generate_toolfew_sft_data.py`, `benchmark/train_toolfew_lora.py`, `benchmark/plot_bfcl_toolfew_sft.py`.

## 실행 기록 (3차 후속)

### 2026-09-02 — 실험 5 CORD-v2 xgrammar 2×2 (완료, 음성 결과)

- CORD-v2 test 100개를 Qwen3-4B와 Qwen2.5-3B에 대해 `{base, STAGE SFT} × {자유, xgrammar}`로 실행했다. 모든 조건은 temp 0.6, top-p 1.0, seed 42, max_new_tokens 3100을 썼고 xgrammar 스키마 호환 실패는 없었다(각 100개 채점).

| Model | Condition | PFR | EMR | SCR | NR | VA |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | base | 75.0 | 34.0 | 75.0 | 25.0 | 65.8 |
| Qwen3-4B | base + xgrammar | 100.0 | 38.0 | 100.0 | 0.0 | 81.5 |
| Qwen3-4B | STAGE SFT | 100.0 | 22.0 | 97.0 | 1.1 | 71.7 |
| Qwen3-4B | STAGE SFT + xgrammar | 99.0 | 22.0 | 99.0 | 1.0 | 71.7 |
| Qwen2.5-3B | base | 98.0 | 23.0 | 96.0 | 2.8 | 70.3 |
| Qwen2.5-3B | base + xgrammar | 98.0 | 24.0 | 98.0 | 2.0 | 71.7 |
| Qwen2.5-3B | STAGE SFT | 100.0 | 7.0 | 98.0 | 0.8 | 60.6 |
| Qwen2.5-3B | STAGE SFT + xgrammar | 100.0 | 7.0 | 100.0 | 0.0 | 60.5 |

- 계획했던 "형식 준수는 비슷하되 STAGE가 값을 더 보존"한다는 주장은 성립하지 않았다. 두 모델 모두 base+xgrammar가 VA 최고이며, 특히 Qwen3은 81.5 대 71.7이다. 이 CORD 결과는 음성 결과로 보존하고 본문 실세계 전이의 근거로 쓰지 않는다. 산출물: `outputs/cord_v2_xgr/`.

### 2026-09-02 — 실험 4A ExtractBench 32k 장문맥 (완료)

- `prompt + 3,100 생성 토큰 ≤ 32,768`의 공통 예산으로 ExtractBench 244개 중 200개를 유지했다(short 143, medium 57; 더 긴 44개는 제외). Qwen3-4B와 Qwen2.5-3B의 base/STAGE SFT를 같은 200개에 실행했다. 지표는 기존 evaluator의 PFR=1−no-output, EMR, SCR, NR, VA이며 길이별 요약은 `benchmark/summarize_extractbench.py`로 같은 원본 레이블을 사용해 냈다.

| Model | Condition | PFR | EMR | SCR | NR | VA | medium VA |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | base | 41.5 | 1.0 | 40.5 | 59.5 | 23.5 | 14.8 |
| Qwen3-4B | STAGE SFT | 80.0 | 0.0 | 78.5 | 20.7 | 33.7 | 29.8 |
| Qwen2.5-3B | base | 30.0 | 0.0 | 20.5 | 75.5 | 10.7 | 1.1 |
| Qwen2.5-3B | STAGE SFT | 79.5 | 0.0 | 77.0 | 21.9 | 22.7 | 17.0 |

- 두 모델 모두 SFT가 전체 VA/SCR와 8k를 넘는 medium 구간 VA에서 우위다. 따라서 4B 장문맥 재학습의 진입 조건(8k 초과 구간 SFT 열세)은 충족하지 않아 실행하지 않는다. 기존 n=27 결과는 이 200개 결과로 교체한다. 산출물: `benchmark/data/extractbench_context32768.jsonl`, `outputs/extractbench/*_ctx32768.{jsonl,xlsx}` 및 `_eval.xlsx`, `_summary.csv`.

### 2026-09-02 — 실험 6·7 인계 상태 (추론 대기)

- **완료된 준비와 커밋**: `41fcbd8`(비용 측정기·SGD 변환기), `cdc4e67`(공식 SGD metric 기반 evaluator), `057bf1b`(동일 vLLM 엔진에서 cold→warm 캐시 pass), `41cb818`(GPU 없는 컴파일 전수 측정). 모두 origin/master에 push돼 있다. 원출력·XLSX는 커밋하지 않았다.
- **실험 6 확정 수치(추론 전)**: STAGE-Eval 851개 중 vLLM xgrammar 호환은 **798**, 비호환은 **53 (6.2%)**. `xgrammar.GrammarCompiler(..., cache_enabled=False)`로 798개를 각각 컴파일한 결과 중앙 **20.3 ms**, p90 **26.3 ms**, 합계 **16.62 s**. 산출물: `outputs/inference_cost/summary.csv`, `xgrammar_skipped.json`.
- **실험 6 재개 명령**: 반드시 `nvidia-smi`에서 완전히 빈 GPU를 확인한 뒤 아래를 실행한다. 비용 수치는 다른 작업과 GPU를 공유하면 무효이므로, 수 GB라도 외부 프로세스가 있으면 시작하지 않는다.

  ```bash
  CUDA_VISIBLE_DEVICES=<빈GPU> ~/work/sunghee/venv/bin/python benchmark/measure_inference_cost.py \
    --model /root/work/sunghee/models/Qwen3-4B --label base_free --batch-size 1 \
    --pass-name cold --second-pass --output-dir outputs/inference_cost
  ```

  이후 같은 명령을 `{base_xgrammar, sft_free, sft_xgrammar}`에 대해 실행한다. xgrammar 조건은 `--guided-json`을 추가한다. batch 32 처리량은 각 조건을 `--batch-size 32 --pass-name throughput`으로 별도 실행한다. `--second-pass`는 cold run 뒤 같은 엔진에서 warm run을 기록한다. `summary.csv`의 중복 행은 재시작 전 확인한다.
- **실험 7 파일럿 준비 완료**: 원자료는 `/mnt/nvme/cache/interns/sgd/`, 공식 metric 참조는 `/mnt/nvme/cache/interns/schema-guided-dst-metrics/`. `benchmark/prepare_sgd.py --format standard --split pilot` 및 `--format explicit`를 실행해 두었고, 각각 `benchmark/data/sgd_pilot_{standard,explicit}.jsonl` 100개가 생성됐다. 동일 `sgd_pilot_ids.json`을 공유하며 seen/unseen service가 50/50이다.
- **실험 7 재개 순서**: 빈 GPU에서 Qwen3 base와 STAGE SFT 각각을 `benchmark/inference.py`로 `sgd_pilot_standard.jsonl`에 실행하고, `benchmark/evaluate_sgd.py --format standard`로 평가한다. SFT가 판정 기준(JGA ≥ base−3pt, 환각 ≤ base+5pt)을 못 넘으면 같은 100 turn을 explicit 형식으로 반복한다. evaluator는 Google Research의 fuzzy non-categorical matching과 JGA 함수를 직접 사용하며, `"no output"`은 explicit에서 빈 슬롯으로 되돌린다.
- **당시 GPU 상태(인계 시점)**: GPU 0은 `seonhong`의 Spatial-TTT `lmms_eval`(Qwen3-VL-2B)이 약 8.8GB, GPU 1은 부모 PID 1의 고아 `VLLM::EngineCore`가 약 130GB를 점유했다. 두 작업 모두 이 저장소 작업이 아니므로 종료하지 않았다.

## 현재 실행 상태 (2026-09-02, 모든 필수 실험 완료)

> 이 섹션은 이전 runbook의 계획이 아니라 **파일·로그로 확인한 실제 완료 상태**다. 새 세션은 이 표와 `outputs/`를 먼저 확인하고, 완료된 GPU 측정을 중복 실행하지 않는다.

| 우선순위 | 작업 | 현재 상태 / 판정 | 바로 할 일 | 완료 기준 |
|---:|---|---|---|---|
| 완료 | ExtractBench xgrammar 2×2 (실험 5 잔여) | Qwen3-4B/Qwen2.5-3B의 base·SFT xgrammar 4개를 200개에 실행·채점. xgrammar 호환 표본은 194개(6개 제외) | 결과 표를 `## 실행 기록`에 반영 | `outputs/extractbench_xgr/`의 JSONL·XLSX·summary |
| 완료 | Llama 기준선 정정 | `Llama-3.2-{1B,3B}-Instruct`로 CORD·ExtractBench base를 재실행. 두 모델 모두 거의 전부 no-output인 음성 기준선 | 기존 pretrained 행을 인용하지 않음 | `outputs/cord_v2/`, `outputs/extractbench/`의 Instruct 결과 |
| 완료 | 구조 보장 비용 (실험 6) | compile-only 1행, 네 조건의 batch=1 cold/warm 8행, batch=32 4행 완료. 모든 run은 798개 공통 xgrammar-호환 표본(53개 제외) | 결과 표·판정을 인용 | `outputs/inference_cost/summary.csv` 13행 및 조건별 JSONL |
| 완료 | SGD 상태 추적 파일럿 (실험 7) | standard에서 SFT JGA 7.64%로 base 28.60%보다 낮고 환각 54.03%로 base 26.48%보다 높아 기준 실패. explicit도 SFT JGA 14.39% < base 30.21% | 음성 결과로 기록, 본실행 중단 | standard·explicit `_eval.json` 보존 |

### 완료 결과 — ExtractBench xgrammar 공통 분모 비교

ExtractBench의 200개 32k 표본 중 xgrammar가 컴파일하지 못한 6개를 모든 조건에서 제외해 **194개(짧음 137, medium 57)** 공통 분모로 다시 채점했다. `PFR=1−no output`, SCR은 schema-valid, VA는 rule-based value match다. free 조건의 재채점 JSONL/summary는 `outputs/extractbench_xgr/*_free_compatible194.*`에 있다.

| Model | condition | PFR | SCR | VA | medium VA |
|---|---|---:|---:|---:|---:|
| Qwen3-4B | base free | 40.2 | 39.2 | 22.2 | 14.8 |
| Qwen3-4B | base + xgrammar | 71.1 | 71.1 | 34.7 | 24.5 |
| Qwen3-4B | STAGE SFT free | 79.4 | 78.4 | 32.5 | 29.8 |
| Qwen3-4B | STAGE SFT + xgrammar | 81.4 | 81.4 | 33.1 | 28.7 |
| Qwen2.5-3B | base free | 28.4 | 19.6 | 9.6 | 1.1 |
| Qwen2.5-3B | base + xgrammar | 56.2 | 56.2 | 19.8 | 12.1 |
| Qwen2.5-3B | STAGE SFT free | 78.9 | 76.3 | 21.6 | 17.0 |
| Qwen2.5-3B | STAGE SFT + xgrammar | 79.4 | 79.4 | 22.1 | 17.8 |

해석은 모델별로 분리한다. Qwen2.5-3B에서는 SFT+xgrammar가 base+xgrammar보다 VA도 높다(22.1 vs 19.8). Qwen3-4B에서는 xgrammar가 base의 VA까지 크게 끌어올려 SFT+xgrammar의 VA(33.1)가 base+xgrammar(34.7)를 넘지 못했다. 따라서 이 벤치마크에서 “같은 구조 준수 수준의 값 보존 우위” 주장은 Qwen2.5에 한정하고, Qwen3는 구조 준수·medium VA에서의 SFT 이득만 보고한다.

### 완료 결과 — 구조 보장 비용 (실험 6)

H200 한 장, vLLM 0.10.2, xgrammar 0.1.23, temperature 0.6, max_new_tokens 3100, max_model_len 8192에서 실행했다. STAGE-Eval 851개 중 xgrammar 호환 798개를 **모든** 조건의 공통 분모로 썼고 53개(6.2%)는 제외했다. 각 batch=1 조건은 warm-up 10개 후 cold→warm을 같은 엔진에서, throughput은 별도 batch=32 엔진에서 측정했다.

| condition | b1 cold median / p90 (s) | b1 warm median / p90 (s) | b32 ex/s | b32 tok/s | mean generated tokens |
|---|---:|---:|---:|---:|---:|
| base free | 10.98 / 14.01 | 10.99 / 14.00 | 1.23 | 2,656.8 | 2,146–2,152 |
| base + xgrammar | 2.02 / 6.52 | 1.99 / 6.47 | 2.10 | 1,323.7 | 631–633 |
| STAGE SFT free | 1.78 / 5.63 | 1.78 / 5.61 | 2.66 | 1,488.4 | 560 |
| STAGE SFT + xgrammar | 1.92 / 5.86 | 1.89 / 5.82 | 2.34 | 1,387.4 | 593 |

`GrammarCompiler(cache_enabled=False)`의 798개 개별 compile은 중앙 20.3 ms, p90 26.3 ms, 합계 16.62 s였다(조건별 재측정도 16.34–16.65 s). xgrammar는 base에서 생성 길이를 크게 줄여 end-to-end 지연/처리량이 좋아졌으므로, 이 표로 “제약 디코딩이 항상 느리다” 혹은 “SFT가 무조건 xgrammar보다 빠르다”라고 주장하지 않는다. 지연 차이는 생성량까지 함께 보고하며, SFT free는 이미 짧은 출력을 생성해 base+xgrammar보다 batch-1에서 더 빠르다(1.78 vs 1.99 s warm).

### 완료 결과 — Llama Instruct 기준선과 SGD 파일럿

- Llama base는 pretrained checkpoint 대신 `Llama-3.2-1B-Instruct` (`.../9213176726f574b556790deb65791e0c5aa438b6`)와 `Llama-3.2-3B-Instruct` (`.../0cb88a4f764b7a12671c53f0838cd831a0843b95`)로 재실행했다. CORD에서는 1B가 no-output 100%, 3B가 97%(SCR 1%, VA 0.63%)였고, ExtractBench에서는 1B no-output 99.5%(SCR 0.5%, VA 0.02%), 3B 100%였다. 이는 유효한 음성 Instruct 기준선이며 기존 pretrained 행은 인용하지 않는다.
- SGD 100-turn pilot은 standard base/SFT와, standard 실패 뒤 동일 turn의 explicit base/SFT를 모두 공식 metric으로 평가했다. standard: base JGA 28.60%, SFT 7.64%, 환각 26.48%/54.03%; explicit: base JGA 30.21%, SFT 14.39%, 환각 23.95%/38.31%. explicit은 SFT JGA를 6.75pt 높였지만 base에는 여전히 못 미쳐(−15.82pt) full SGD 실행을 하지 않는다.

### 명시적 보류 — 실험 4B 장문맥 SFT 재학습

실험 4B는 **시작하지 않았으며 현재 실행하지 않는다.** 4A에서 32k 문맥으로 표본을 27개에서 200개로 늘렸고, 8k 초과 medium 구간에서도 SFT가 base보다 우위였다(Qwen3 VA 29.8 vs 14.8, Qwen2.5 VA 17.0 vs 1.1). 따라서 “장문맥에서 SFT가 열세일 때만 재학습”이라는 진입 조건이 불충족이다. 긴 8k–24k 보고서 데이터를 추가 생성해 32k SFT를 하는 4B-2는 카메라레디/후속 작업이다.

## 다음 실행 (3차) — 실험 4 ExtractBench 장문맥 / 실험 5 실세계 xgrammar 비교 (에이전트 runbook)

> 2026-09-03 등록. 목적은 실세계 벤치마크(CORD, ExtractBench)에서 "형식을 올리는 다른 방법(xgrammar)과 같은 조건에서 우리가 값·의미를 더 잘 보존한다"를 보이는 것. 우선순위 **5 → 4A → (4B는 4A 결과를 보고)**. 4A·5는 추론만이라 합쳐 GPU 약 2시간, 4B는 재학습이라 반나절 이상.
> 공통: temp 0.6, top-p 1.0, seed 42, max_new_tokens 3100, `benchmark/evaluate.py` 지표 그대로. 모든 조건은 **같은 예제 집합**으로 채점한다.

### 실험 5 — CORD-v2·ExtractBench에서 xgrammar 2×2 비교 (GPU 약 1시간)

- **주장하려는 것**: xgrammar도 PFR/SCR을 올리지만 값(VA)은 STAGE SFT가 더 보존한다. 실험 1(STAGE-Eval)과 같은 2×2를 OOD 실세계 데이터에서 반복하는 것이며, CORD에서 Qwen2.5-3B의 VA 하락(70.3→60.6)이 "SFT 때문"인지 "형식을 강제하면 누구나 겪는 손실"인지 가리는 데도 필요하다.
- 조건 (모델별 4개): {base, STAGE SFT} × {자유 디코딩, xgrammar `guided_json`}. 모델: Qwen3-4B, Qwen2.5-3B (필수), Llama-3.2-1B/3B (아래 수정 후).
- 데이터: CORD-v2 test 100개 (`benchmark/prepare_cord.py` 산출물), ExtractBench는 실험 4A의 32k 집합(200개)을 쓰고, 4A 전이면 기존 27개.
- 실행: `benchmark/inference.py`에 실험 1에서 쓴 `--guided-json`(xgrammar) 경로가 있으므로 CORD/ExtractBench 입력 파일에 그대로 적용. 스키마 컴파일 실패 예제는 실험 1과 같이 `skip_reason`으로 기록하고 **네 조건 모두에서 제외**해 분모를 맞춘다.
- run 이름: `outputs/cord_v2_xgr/{model}_{base,sft}_{free,xgrammar}.jsonl` 및 `_eval.xlsx`; ExtractBench는 `outputs/extractbench_xgr/`.
- **Llama 수정 (필수)**: 기존 CORD/ExtractBench의 Llama base는 `meta-llama/Llama-3.2-1B`, `-3B`(사전학습 모델)로 실행돼 논문의 base(`Llama-3.2-1B-Instruct`, `-3B-Instruct`)와 다르다. `benchmark/run_cord_suite.py`의 경로를 Instruct로 바꿔 base 행을 재실행하고, 기존 Llama base 행은 표에서 교체한다. 재실행 전까지 Llama 행은 어디에도 인용하지 않는다.
- 판정: (a) base+xgrammar의 PFR/SCR이 SFT와 비슷한 수준(±3pt)으로 올라오고 (b) VA는 SFT가 base+xgrammar보다 높으면 "같은 형식 준수 수준에서 의미 보존 우위"를 본문 한 단락 + 표로 쓴다. Qwen2.5-3B에서 SFT VA < base+xgrammar VA이면 그 사실을 그대로 보고하고 Qwen3-4B로 범위를 한정한다.

### 실험 4A — ExtractBench 장문맥 추론 (재학습 없음, GPU 약 1시간)

- **배경**: 기존 27개는 `max_model_len=8192`(추론 설정) 때문이다. Qwen3-4B와 STAGE SFT 체크포인트는 모델 자체가 32k 이상 문맥을 지원한다. 제외된 217개의 프롬프트 토큰은 p50 13,824, p90 49,097이며, `max_model_len`을 늘리면 들어오는 문서 수는 아래와 같다.

| max_model_len | 포함 문서 (244개 중) |
|---:|---:|
| 8,192 (현재) | 27 |
| 16,384 | 132 |
| 32,768 | 200 |
| 40,960 | 210 |

- 실행: `--max-model-len 32768`로 Qwen3-4B base/SFT, Qwen2.5-3B base/SFT를 200개에 대해 실행. `benchmark/filter_extractbench_context.py`의 예산 계산(prompt + 3100 ≤ max_model_len)을 32768로 재실행해 `benchmark/data/extractbench_context32768.jsonl`을 만든다. 나머지 44개는 `context_skipped.jsonl`에 사유 유지.
- 리포트: 전체 200개 표 + **문서 길이 구간별(short/medium/long) 분해**. 핵심 질문은 "SFT 모델이 학습 cutoff(8k)를 넘는 입력에서 base 대비 무너지는가"이다. 8k 초과 구간에서 SFT의 PFR/VA가 base보다 낮으면 4B로 간다.
- 판정: 200개에서 SFT가 base 대비 VA·SCR 우위이면 ExtractBench를 Appendix에서 본문 한 문장으로 승격("digital-text 문서 200개에서도 일관"). n=27 결과는 폐기하고 200개로 교체.

### 실험 4B — 장문맥 SFT 재학습 (보류: 4A에서 8k 초과 구간 열세가 확인될 때만, 반나절 이상)

- **주의**: STAGE 학습 데이터의 토큰 길이는 p50 3,076 / p90 7,145 / p99 8,102 / max 8,192로, cutoff 8,192에 잘린 예시가 1% 미만이다. 즉 **같은 데이터로 cutoff만 16k/32k로 올려 재학습하면 새로 배우는 장문맥 감독은 거의 없다.** 재학습이 의미 있으려면 긴 입력 예시가 필요하다.
- 4B-1 (빠른 확인, 약 3시간): `src/train/qwen3_4B_full_guide.yaml`에서 `cutoff_len: 16384`, 나머지 동일(3 epoch, lr 4e-5, full-parameter)로 재학습 → 4A와 같은 200개 평가. 기대 효과는 잘린 1%의 복원과 위치 인코딩 적응 정도이므로 개선이 작아도 실패가 아니다. 결과가 기존 SFT와 ±2pt 이내면 "cutoff는 원인이 아님"으로 기록.
- 4B-2 (데이터 확장, 카메라레디 권장): STAGE 파이프라인으로 8k~24k 토큰 보고서를 추가 생성(같은 스프레드시트에서 여러 시트·섹션을 이어 붙이는 방식)해 기존 데이터에 10~20% 섞고 `cutoff_len: 32768`로 재학습. 마감(2026-09-07) 전 완료는 어렵다고 보고 Limitations/Future work에 한 문장으로 적는다.
- 새 체크포인트는 HF `boradorish/`에 `qwen3-4b-stage-ctx16k` 식으로 올리고, `EXPERIMENTS.md`에 학습 로그 경로와 wall-clock을 남긴다.
- **현재 판정:** 4A의 medium 구간에서 SFT가 두 Qwen 모델 모두 base보다 VA가 높아 진입 조건이 충족되지 않았다. 이 재학습은 아직 시작하지 않았으며, 새 근거 없이 실행하지 않는다.

### 논문 반영 지침

- 실험 5가 성립하면 Results에 "Real-world transfer" 소절: CORD·ExtractBench 각 표에 {base, base+xgrammar, SFT, SFT+xgrammar} 4행, 본문은 VA 중심으로 서술하고 EMR은 언급하지 않는다(ExtractBench EMR은 전 조건 0).
- 실험 1과 같은 프레임("구조는 xgrammar로도 오르지만 값은 데이터 학습이 올린다")을 in-distribution → OOD 실세계로 확장하는 것이 서사의 핵심이다.

## 포지셔닝 변경 (2026-09-03)

- BFCL 계열 결과(2, 2b~2f)는 STAGE-SFT가 **도구 선택(라우팅)** 에서 base보다 약함을 일관되게 보였다. 따라서 논문은 STAGE-SFT를 tool router로 주장하지 않는다. 대신 **에이전트의 지각/상태 추출 계층**(긴 비정형 관측 → 스키마 타입의 상태)으로 포지셔닝한다. 강점 근거: 제약 디코딩 없는 구조 준수(PFR 99.6, SCR 97.6, BFCL 인자 스키마 유효 100), 긴 문서에서의 값 보존(VA 84.7 vs base+xgrammar 66.0), 잡음 억제(NR 36.9→1.4), 최소 모델에서 최대 이득(Llama-1B VA 19.5→80.7).
- BFCL은 Appendix에 "라우팅은 이 모델의 역할이 아니며 그 원인은 학습 스키마 분포(all-required 95.5%, oneOf 1.4%)"라는 scope 근거 한 단락으로만 남긴다.
- (선택, GPU 0, 30분) 기존 채점 결과로 "무작위 값 k개가 모두 맞을 확률" 곡선(k=1은 VA, k=전부는 EMR)을 그려 값 정확도 차이가 값 묶음 단위에서 어떻게 증폭되는지 보이는 그림 1개. 새 지표로 주장하지 않는다.

## 다음 실행 (4차) — 실험 6 구조 보장 비용 / 실험 7 SGD 대화 상태 추적 (에이전트 runbook)

> 2026-09-03 등록. 이 runbook의 4A·실험 5·실험 6·SGD 파일럿은 모두 완료됐다. 최종 판정과 재실행 금지 사항은 `## 현재 실행 상태`를 따른다.

### 실험 6 — 구조 보장 방식의 추론 비용 비교 (GPU 약 1시간)

- **주장하려는 것**: STAGE SFT는 제약 디코딩과 같은 구조 준수를 **추론 시 추가 비용 0**으로 달성하고, 스키마 커버리지 구멍도 없다. 실험 1(정확도)의 짝이 되는 비용 표다.
- 대상: STAGE-Eval xgrammar 호환 798개, Qwen3-4B. 조건 4개: base 자유 / base+xgrammar / SFT 자유 / SFT+xgrammar. 환경은 실험 1과 동일(vLLM 0.10.2, xgrammar 0.1.23, H200 1장, temp 0.6, max_new 3100, max_len 8192).
- 측정 항목:
  1. **예제당 지연 시간 (batch=1)**: 요청 제출부터 완료까지 wall-clock. 전 798개, 중앙값·p90. 에이전트의 호출 단위 조건.
  2. **처리량 (batch=32)**: 생성 토큰/초와 예제/초.
  3. **문법 컴파일 시간**: vLLM 밖에서 `xgrammar.GrammarCompiler(tokenizer_info, cache_enabled=False)`로 798개 스키마를 각각 `compile_json_schema`해 예제당 컴파일 시간 중앙값·p90·합계. 에이전트는 툴마다 스키마가 달라 매 호출 컴파일이 실제 비용임을 본문에 설명한다.
  4. **end-to-end에서 컴파일 포함 여부**: vLLM은 문법을 캐시하므로, (a) 캐시 유효 상태(같은 798개를 두 번째 실행)와 (b) 콜드 상태(첫 실행) 지연을 모두 기록한다. 캐시를 끄는 옵션이 있으면 그것을 우선 사용하고 옵션명을 기록한다.
  5. **생성 토큰 수** 조건별 평균. SFT가 짧게 생성해서 빠른 것인지 분리해서 보고한다(지연 시간과 함께 "토큰당 시간"도 제시).
  6. **스키마 커버리지**: 실험 1의 xgrammar 미지원 53/851(6.2%)을 그대로 인용하고 미지원 사유 상위 3개를 표기.
- 스크립트: `benchmark/measure_inference_cost.py`. 추론은 `benchmark/inference.py` 경로를 재사용하되 타이머만 추가한다. 결과는 `outputs/inference_cost/{condition}_{batch}.jsonl`(예제별 시간) + `summary.csv`.
- 주의: GPU를 독점한 상태에서 측정한다. 다른 사람의 작업이 같은 카드에 있으면 측정하지 않는다. 각 조건은 워밍업 10개 후 측정하고, 실행 순서를 기록한다.
- 리포트: 조건 × {지연 중앙값/p90, 처리량, 컴파일 시간, 평균 생성 토큰, 토큰당 시간}. 판정: SFT 자유의 지연·처리량이 base+xgrammar 이상이면 본문 한 단락 + 작은 표("no inference-time overhead"). 반대로 나오면 그 사실을 그대로 적고 "커버리지·값 정확도"로 논거를 한정한다.

### 실험 7 — Schema-Guided Dialogue(SGD) 대화 상태 추적 zero-shot (2단계: 표준 형식 → 명시 값 형식)

- **주장하려는 것**: 스프레드시트 보고서로만 학습한 스키마 추출 능력이 에이전트 도메인(태스크 지향 대화의 상태 추적)으로 전이된다. SGD는 서비스별 슬롯 스키마가 명시된 표준 DST 데이터셋이며 test에 학습 미포함 서비스가 있어 스키마 일반화도 함께 측정된다.
- **실행 원칙**: 먼저 **표준 DST 형식(7-A)** 그대로 돌린다. 여기서 STAGE-SFT가 base 대비 열세이면, 그 원인이 "빈 슬롯을 비워두지 못함"인지 확인하고 **명시 값 형식(7-B)** 으로 다시 돌린다. 7-B는 우리 모델의 학습 관행(모든 필드 required, enum/const로 값 선택)에 맞춘 형식이므로 성능이 오르면 "형식이 맞으면 전이된다"는 결과가 되고, 그래도 안 오르면 coverage bias의 음성 결과로 기록한다. 두 형식 모두 base에도 동일하게 적용해 공정 비교한다.

**공통 준비**

- 데이터: `git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue` → `test/dialogues_*.json`, `test/schema.json`. 저장 위치 `/mnt/nvme/cache/interns/sgd/`.
- 예제 단위: 대화의 각 **사용자 턴**과 그 턴의 활성 서비스 frame마다 예제 1개. Report는 해당 턴까지의 이력 전체를 `USER: …` / `SYSTEM: …` 줄로 나열.
- 표본: 파일럿은 test에서 서비스별 균등 **100턴**(학습 포함 서비스와 미포함 서비스 절반씩). 본실행은 **2,000턴** 균등 샘플. 파일럿과 본실행의 턴 id를 `benchmark/data/sgd_{pilot,full}_ids.json`으로 고정해 7-A와 7-B가 같은 턴을 쓰게 한다.
- 모델: 파일럿은 Qwen3-4B base vs STAGE SFT. 본실행은 Qwen2.5-3B, Llama-3.2-1B/3B **Instruct** base와 각 SFT 추가.
- 채점 (`benchmark/evaluate_sgd.py`): 예측을 SGD 공식 예측 포맷(dialogue json에 `state.slot_values` 채움)으로 변환해 저장소의 공식 `evaluate.py`로 **JGA와 슬롯 정확도(active-intent 제외)** 를 계산한다. 범주형은 exact, 비범주형은 공식 스크립트의 fuzzy 매칭 그대로. 파싱 실패 턴은 JGA 0. 여기에 두 진단을 반드시 추가: **환각 슬롯률**(gold에 없는 슬롯에 값을 채운 비율)과 **누락 슬롯률**(gold에 값이 있는데 비우거나 not mentioned로 둔 비율).
- 스크립트: `benchmark/prepare_sgd.py --format {standard,explicit}`, `benchmark/evaluate_sgd.py`. 산출물 `benchmark/data/sgd_{pilot,full}_{standard,explicit}.jsonl`, `outputs/sgd/{model}_{base,sft}_{standard,explicit}.jsonl`, `_eval.json`.

**7-A. 표준 DST 형식 (먼저 실행)**

- JSON Schema: 서비스의 슬롯마다 property 1개. 범주형은 `enum: possible_values`, 비범주형은 `type: string` + 슬롯 `description`. **`required`는 비워 두고** `additionalProperties: false`. 즉 "언급된 슬롯만 넣어라"는 통상의 DST 정의이며, 스키마 상단 `description`에 "Include only slots the user has specified so far"를 적는다.
- Gold: `frame.state.slot_values`의 첫 값. 언급 안 된 슬롯은 키 자체가 없음.
- 예상되는 실패 양상: STAGE-SFT가 required가 비어 있어도 모든 슬롯을 채워 환각 슬롯률이 높게 나오는 것. 파일럿 결과에서 **환각 슬롯률**과 **누락 슬롯률**을 base와 나란히 기록하고, SFT 오답 턴 10개를 뽑아 "빈 슬롯을 채운 것"이 주원인인지 확인한다.
- 판정: SFT JGA ≥ base JGA − 3pt 이고 환각 슬롯률 ≤ base + 5pt이면 7-A 형식으로 본실행하고 7-B는 생략(보고 시 "표준 형식에서 전이됨"). 아니면 7-B로 간다.

**7-B. 명시 값 형식 (7-A에서 SFT 열세이고 원인이 빈 슬롯 채움일 때)**

- 아이디어: "비워두기"를 우리 모델이 잘하는 "값 고르기"로 바꾼다. 모든 슬롯을 `required`로 두고, 비어 있음을 명시적 값으로 표현한다.
  - 범주형: `enum: possible_values + ["no output"]`
  - 비범주형: `type: string`, description 끝에 `Use "no output" if the user has not specified this slot.` 추가
  - 스키마 상단 description: `Fill every slot. Write "no output" for any slot the user has not specified so far.`
  - `additionalProperties: false`, 전체 `required`.
- Gold: 언급 안 된 슬롯에 `"no output"`을 채운 형태. 채점 시 `"no output"`은 "슬롯 없음"으로 되돌린 뒤 공식 evaluate.py에 넣는다(공식 지표는 7-A와 완전히 같은 정의로 계산됨).
- 빈 값 표기는 `"no output"` 하나로 고정한다. 대소문자·공백 변형은 채점 전 정규화한다. STAGE 학습 데이터에서 실제로 쓰인 빈 값 표기가 있으면(`""`, `null`, `"N/A"` 등 `data/` 학습 파일에서 빈도 확인) 그 표기를 7-B' 변형으로 하나 더 시도하되, 파일럿 100턴에서 가장 좋은 표기 하나만 본실행에 쓴다. 어떤 표기를 골랐고 왜 골랐는지 기록한다.
- 판정: 7-B에서 SFT JGA가 7-A 대비 유의미하게 오르고(≥ +5pt) base와 동률 이상이면 본실행(7-B 형식, 전 모델). 본문에는 "명시 값 형식에서 전이" + "표준 형식에서는 빈 슬롯 채움으로 열세"를 **둘 다** 적는다. 7-B도 열세이면 파일럿 수치와 환각 사례 5개를 음성 결과로 `## 실행 기록`에 남기고 중단(논문 미수록).

**리포트 (본실행)**

- 모델 × 형식(7-A, 7-B) × {JGA, 슬롯 정확도, 환각 슬롯률, 누락 슬롯률}, 학습 포함 vs 미포함 서비스 분리 열. 본문은 JGA 1개 표, 나머지는 Appendix.

### 논문 반영 지침 (전체)

- 본문 Results 구성안(6페이지): (1) STAGE-Eval 주결과(기존), (2) 학습 vs 제약 디코딩: 정확도(실험 1)+비용(실험 6) 한 묶음, (3) 실세계 전이: CORD·ExtractBench × xgrammar(실험 5, 4A), (4) 에이전트 상태 추적(실험 7, 성립 시). BFCL과 few-shot/256-SFT는 Appendix "Scope: tool routing".
- Limitations: coverage bias(도구 선택·비우기)와 데이터 측 교정 방향(oneOf, 가변 길이, 미언급 필드) 각 한 문장.

## 다음 실행 (5차) — 실험 8 non-thinking base 재실행 / 실험 9 CORD 레이아웃 렌더링 / 실험 10 CORD 적응 곡선 (에이전트 runbook)

> 2026-09-03 등록. 우선순위 **8 → 9 → (10은 9 결과 보고)**. 8은 정정이라 필수, 9는 CORD를 살리기 위한 입력 정렬 실험, 10은 9가 격차를 못 닫을 때의 대안.

### 실험 8 — base Qwen3-4B를 thinking 없이 재실행 (필수 정정, GPU 약 2시간)

- **문제**: STAGE-Eval 798개 base 자유 출력 전부에 `<think>` 태그가 있고 234개(29%)는 생각을 끝내지 못한 채 3,100토큰에서 잘렸다. 코드에 `enable_thinking=False`가 없었다. base의 파싱 실패 34.6%, 실험 6의 11초 지연, xgrammar가 base를 크게 살리는 현상(첫 토큰 `{` 강제로 thinking 억제)이 모두 이 설정에서 나온다.
- **수정 (커밋됨)**: `src/utils/vllm_inference.build_chat_prompts(..., enable_thinking=)`, `benchmark/inference.py --no-thinking`, `benchmark/measure_inference_cost.py --no-thinking`. Qwen3 chat template의 `enable_thinking=False`를 전달한다. 다른 tokenizer에는 영향 없다.
- **적용 범위**: **Qwen3-4B base만**. Qwen2.5·Llama에는 thinking이 없고, STAGE SFT는 학습 프롬프트에 빈 `<think></think>`가 없었으므로 SFT에는 플래그를 주지 않는다(프롬프트가 달라짐).
- 재실행 목록 (모두 base, 나머지 설정 동일):
  1. STAGE-Eval `benchmark/data/stage_eval_test.jsonl` 851개: 자유 / xgrammar → `outputs/nothink/qwen3_4b_base_nothink_{free,xgrammar}`. 채점 후 798개 공통 분모로 실험 1 표의 base 두 행 교체.
  2. CORD 100: 자유 / xgrammar → `outputs/cord_v2_xgr/qwen3_4b_base_nothink_{free,xgrammar}`.
  3. ExtractBench 200(32k): 자유 / xgrammar → `outputs/extractbench_xgr/qwen3_4b_base_nothink_{free,xgrammar}`.
  4. 실험 6 비용: `measure_inference_cost.py --no-thinking`으로 base_free·base_xgrammar의 batch=1 cold/warm, batch=32.
- **리포트**: 각 표에 "base (thinking off)" 행을 추가하고 기존 thinking 행은 Appendix에 남긴다(정직성). 실험 1의 SFT 대 base 격차가 얼마나 줄었는지 한 문장으로 기록. 논문 본문 Table의 base Qwen3-4B 행도 같은 설정으로 교체해야 하므로 STAGE-Eval 전체 851 결과를 기존 `benchmark/evaluate.py` 출력 형식 그대로 보존한다.
- 완료 조건: 출력에 `<think>` 태그가 0개, 무출력이 thinking 미종료가 아닌 이유로만 발생.

### 실험 9 — CORD를 살리기: 레이아웃 보존 렌더링 (GPU 약 1시간)

- **진단 (2026-09-03, 리프 단위 전수 분석)**: Qwen3 SFT의 불일치 리프 414개 중 값이 다른 필드로 간 "shifted"가 177개(같은 품목 내 105, 다른 품목 72), 라벨을 값에 붙여 쓴 것 28개(`"TOTAL 46,000"`), 수식어 탈락 30개(`"JASMINE MT ( L )"` → `"JASMINE MT"`), 누락 67개. 필드별로는 price 53%, unitprice 40%, num 27%, menuqty_cnt 45%가 약하다. shifted 오류의 출처는 인접 줄이 아니라 2줄 이상 떨어진 곳이 126개.
- **원인**: 현재 `prepare_cord.ocr_report()`는 CORD `valid_line`(필드 단위 그룹)마다 한 줄을 출력해 입력이 **토큰 하나씩 세로로 나열된 목록**이 된다. 예: `EGG TART / 13,000 / 1 / CHOCO CUS ARD PASTRY / 2 / 24,000`. 품목마다 수량·가격 순서가 뒤바뀌는데, STAGE SFT는 학습 보고서의 99.5%가 Markdown 표였기 때문에 이를 고정 열 순서의 표로 읽어 위치대로 배정한다(cnt=13,000, price=1). base는 영수증 상식으로 맞춘다. 즉 입력 직렬화가 학습 분포와 어긋난 것이 주원인이다.
- **9-A 시각적 행 렌더링 (핵심)**: `prepare_cord.py --layout rows`. 단어 quad 좌표로 (1) y 중심이 가까운 단어를 한 행으로 묶고(허용치 = 0.5×단어 높이 중앙값, 하한 4px), (2) 행 안에서 x로 정렬, (3) 수평 간격이 단어 높이의 1.5배를 넘으면 ` | `로 셀을 나눈다. 프로토타입 결과: `1 EGG TART | 13,000`, `BASO KUAH | 1 43.636 | 43.636`. 라벨 줄이 촘촘한 영수증(cord_test_003)에서 행이 합쳐지는 경우가 있으니 허용치를 0.4~0.6 사이에서 조정하고, **검증 기준**: 100개 gold 값 각각이 렌더 텍스트에 부분 문자열로 존재하는 비율이 현재 렌더링(필드 단위) 이상이어야 한다. 렌더링 코드는 `benchmark/prepare_cord.py`에 옵션으로 추가하고 기존 렌더링은 유지한다.
- **9-B 필드 설명**: 유도 스키마의 각 property에 CORD 공식 의미를 `description`으로 넣는다(`nm`: menu item name as printed, `cnt`: quantity, `unitprice`: unit price, `price`: line total, `num`: item code, `itemsubtotal`, `sub_total.subtotal_price`, `discount_price`, `tax_price`, `service_price`, `etc`, `total.total_price`, `cashprice`, `changeprice`, `creditcardprice`, `emoneyprice`, `menuqty_cnt`: number of items, `menutype_cnt`: number of item types). 값은 영수증에 **인쇄된 그대로**(통화 기호·`@`·`X` 포함) 쓰라는 문장을 스키마 최상위 description에 넣는다. 모든 조건에 동일 적용.
- **9-C 1-shot**: CORD **train** split에서 예시 1개(품목 3개 이상, `@`·`X` 관례 포함)를 골라 렌더링·스키마·정답을 프롬프트 앞에 붙인다. test와 겹치지 않음을 stem으로 확인. 모든 조건에 동일 적용.
- **조건**: Qwen3-4B, Qwen2.5-3B × {base, STAGE SFT} × {자유, xgrammar} × 변형 {A, A+B, A+B+C}. Qwen3 base는 실험 8의 `--no-thinking`으로. 산출물 `outputs/cord_v2_layout/{variant}/{model}_{base,sft}_{free,xgrammar}.jsonl` + `_eval.xlsx`. 채점은 `benchmark/evaluate.py` 그대로.
- **판정**: 어느 변형에서든 SFT VA ≥ base+xgrammar VA(Qwen3 기준 81.5 근방)이면 CORD를 본문 실세계 전이의 두 번째 근거로 승격하고, "레이아웃 보존 직렬화 + STAGE"로 서술한다. 미달이면 Appendix에 위 진단(위치 기반 배정 습관)과 함께 남긴다. 어느 쪽이든 변형별 SFT·base 수치를 모두 기록해 렌더링 효과가 base에도 얼마나 가는지 보인다.

### 실험 10 — CORD 적응 곡선 (9가 격차를 못 닫을 때, GPU 약 2시간)

- **주장하려는 것**: STAGE SFT는 새 문서 유형에 **적응이 빠른 초기화**다. 워크숍 서사("좁은 하위 작업을 작은 모델이 맡는다")와 맞고, zero-shot 열세를 정직하게 인정하면서도 실용 가치를 보인다.
- 데이터: CORD train 800에서 {50, 200, 800}개. 렌더링은 실험 9에서 고른 변형. 검증은 CORD validation 100, 최종 평가는 test 100.
- 학습: `benchmark/train_toolfew_lora.py`와 같은 LoRA(rank 16, 3 epoch) 설정을 재사용해 초기화 {Qwen3-4B base, STAGE SFT} × 데이터 크기 3 = 6개 어댑터. 동일 seed.
- 평가: 자유 디코딩 기준 VA·EMR·PFR, 보조로 xgrammar. 표는 초기화 × 데이터 크기.
- 판정: 모든 크기에서 STAGE 초기화가 base 초기화보다 VA가 높고, 특히 50·200에서 격차가 크면 본문 한 단락("data-efficient adaptation"). 800에서 수렴하면 그 사실도 적는다. base 초기화가 앞서면 Appendix 음성.

## 실행 기록 (5차) — 실험 8 non-thinking base

### 2026-09-03 — 실험 8 (1/4) STAGE-Eval base Qwen3-4B, thinking 끔 (완료)

- `--no-thinking`으로 STAGE-Eval 851개를 자유/xgrammar 재실행했다(각 14분, H200 1장). 출력 851개 모두 `<think>` 태그 0개. 산출물 `outputs/nothink/qwen3_4b_base_nothink_{free,xgrammar}.jsonl` + `_eval.xlsx`. 아래는 `benchmark/evaluate.py` 지표, PFR = 1 − no-output.

| Qwen3-4B, STAGE-Eval | 집합 | PFR | EMR | SCR | NR | VA |
|---|---|---:|---:|---:|---:|---:|
| base, thinking 켬 (기존, 실험 1) | 798 | 65.4 | 33.0 | 60.0 | 36.9 | 48.8 |
| base + xgrammar, thinking 켬 (기존) | 798 | 98.6 | 34.0 | 94.6 | 1.5 | 66.0 |
| **base, thinking 끔** | 798 | 99.5 | 38.6 | 92.0 | 4.9 | 69.3 |
| **base + xgrammar, thinking 끔** | 798 | 99.2 | 34.6 | 95.1 | 1.0 | 67.1 |
| STAGE SFT, 자유 (기존) | 798 | 99.6 | 63.7 | 97.6 | 1.4 | 84.7 |
| STAGE SFT + xgrammar (기존) | 798 | 99.4 | 59.3 | 95.4 | 0.8 | 82.6 |
| base, thinking 끔 | 851 | 99.5 | 38.2 | 91.1 | 5.7 | 69.1 |
| base + xgrammar, thinking 끔 | 851 | 93.1 | 32.4 | 89.2 | 7.1 | 62.9 |

- **읽기.** (1) 기존 base 행의 파싱 실패 34.6%와 낮은 VA는 대부분 thinking 미종료 절단이었다. thinking을 끄면 base도 파싱 99.5%, SCR 92.0이다. (2) SFT의 이득은 값에 집중된다: VA 69.3→84.7(+15.4), EMR 38.6→63.7(+25.1), NR 4.9→1.4. 기존 표의 VA +35.9는 +15.4로 정정된다. (3) non-thinking base에 xgrammar를 얹으면 SCR +3.1, NR −3.9지만 VA는 −2.2, EMR −4.0으로 값은 오히려 나빠진다. "제약 디코딩은 구조만 고치고 값은 못 올린다"는 실험 1의 논지는 유지되며 더 단순해진다: 형식 준수는 thinking만 끄면 base로도 거의 달성되고, 값 정확도는 데이터 학습이 올린다. (4) 851 전체에서 xgrammar 조건의 PFR 93.1은 미지원 스키마 53개의 무출력 때문이므로 표는 798 기준으로 쓴다.
- **논문 영향.** 본문 Table의 Qwen3-4B base 행(EMR 31.37, VA 45.46, 파싱 실패 39.95%)은 thinking 설정의 수치다. 851 non-thinking 값(EMR 38.2, VA 69.1, 파싱 실패 0.5%)으로 교체해야 하고, "PFR 39.95→0.35"류의 문장은 삭제한다. DeepJSONEval의 base Qwen3-4B 행도 같은 설정이면 재실행 대상이다(레포 밖 결과라 확인 필요). 기존 thinking 행은 Appendix에 "thinking 모드 base"로 보존한다.
- 남은 재실행(2/4~4/4): CORD 100, ExtractBench 200(32k), 실험 6 비용. 2026-09-03 05:30에 GPU0(CORD → 비용)·GPU1(ExtractBench) 체인으로 시작. 로그 `outputs/nothink/chain_gpu{0,1}.log`.

### 2026-09-03 — 실험 8 (2/4, 3/4) CORD·ExtractBench base Qwen3-4B, thinking 끔 (완료)

| CORD-v2 100 | PFR | EMR | SCR | NR | VA |
|---|---:|---:|---:|---:|---:|
| base, thinking 켬 (기존) | 75.0 | 34.0 | 75.0 | 25.0 | 65.8 |
| base + xgrammar (기존) | 100.0 | 38.0 | 100.0 | 0.0 | 81.5 |
| **base, thinking 끔** | 100.0 | 38.0 | 100.0 | 0.0 | 80.0 |
| **base + xgrammar, thinking 끔** | 100.0 | 37.0 | 100.0 | 0.0 | 80.0 |
| STAGE SFT (기존) | 100.0 | 22.0 | 97.0 | 1.1 | 71.7 |

| ExtractBench 194 (xgrammar 공통) | PFR | SCR | VA | short VA / PFR (137) | medium VA / PFR (57) |
|---|---:|---:|---:|---:|---:|
| base, thinking 켬 (기존) | 40.2 | 39.2 | 22.2 | 25.3 / 45.3 | 14.8 / 28.1 |
| base + xgrammar (기존) | 71.1 | 71.1 | 34.7 | 38.9 / 79.6 | 24.5 / 50.9 |
| **base, thinking 끔** | 68.0 | 67.5 | 34.4 | 38.3 / 75.9 | 25.0 / 49.1 |
| **base + xgrammar, thinking 끔** | 73.7 | 73.7 | 36.6 | 41.5 / 83.2 | 25.0 / 50.9 |
| STAGE SFT (기존) | 79.4 | 78.4 | 32.5 | 33.6 / 82.5 | **29.8 / 71.9** |
| STAGE SFT + xgrammar (기존) | 81.4 | 81.4 | 33.1 | 35.0 / 86.1 | 28.7 / 70.2 |

- **CORD.** thinking을 끄면 base 자유 디코딩만으로 PFR 100, VA 80.0이다. SFT(71.7)는 xgrammar 유무와 무관하게 base에 8.3pt 뒤진다. CORD 음성 판정은 유지되며 원인은 실험 9 진단(위치 기반 배정)대로다. 실험 9의 base 대조군은 이 non-thinking 행을 쓴다.
- **ExtractBench.** non-thinking base는 전체 VA에서 SFT와 동률 수준(34.4 vs 32.5)이고 short 문서에서는 앞선다(38.3 vs 33.6). SFT의 우위는 두 곳에 남는다: (1) 구조 안정성 PFR 79.4 vs 68.0, (2) **8k 초과 medium 문서** VA 29.8 vs 25.0, PFR 71.9 vs 49.1. 즉 "긴 관측에서의 신뢰성"으로 주장 범위를 좁혀야 한다. 기존 "SFT VA 우위" 문장은 medium 구간 한정으로 수정.
- **종합 (실험 8까지 반영한 Qwen3-4B 요약).** in-distribution(STAGE-Eval)에서는 값 +15.4 VA, +25.1 EMR. OOD 실세계에서는 구조 신뢰성과 장문 구간에서 우위, 짧은 문서·토큰 스트림 입력에서는 base 이하. 본문은 이 범위를 그대로 적는다. Qwen2.5-3B에는 thinking이 없으므로 기존 행 유효(ExtractBench에서 SFT가 구조·값 모두 우위).
- 산출물: `outputs/cord_v2_xgr/qwen3_4b_base_nothink_{free,xgrammar}.*`, `outputs/extractbench_xgr/qwen3_4b_base_nothink_{free,xgrammar}.*`. 비용 측정(4/4)은 GPU0 체인에서 진행 중(`outputs/nothink/chain_gpu0.log`).

### 2026-09-03 — 실험 8 (4/4) 실험 6 비용 재측정, base Qwen3-4B thinking 끔 (완료)

- `measure_inference_cost.py --no-thinking`으로 base 자유/xgrammar를 batch=1(cold→warm, 같은 엔진)과 batch=32로 재측정했다. GPU0 단독 사용, 798개 공통 집합, 나머지 설정은 실험 6과 동일. `summary.csv`에 `base_nothink_free,throughput` 행이 두 번 기록됐는데(2.49 / 2.52 예제/s) 값이 같으므로 뒤 행을 쓴다.

| Qwen3-4B | b=1 cold 중앙값 / p90 (s) | b=1 warm 중앙값 / p90 (s) | b=32 예제/s | b=32 토큰/s | 평균 생성 토큰 | 컴파일 ms/스키마 |
|---|---:|---:|---:|---:|---:|---:|
| base 자유, thinking 켬 (기존) | 10.98 / 14.01 | 10.99 / 14.00 | 1.23 | 2,657 | 2,149 | — |
| base + xgrammar, thinking 켬 (기존) | 2.02 / 6.52 | 1.99 / 6.47 | 2.10 | 1,324 | 633 | 20.0 |
| **base 자유, thinking 끔** | 1.82 / 6.06 | 1.82 / 6.05 | 2.52 | 1,479 | 588 | — |
| **base + xgrammar, thinking 끔** | 2.00 / 6.33 | 1.98 / 6.31 | 2.15 | 1,336 | 622 | 19.2 |
| STAGE SFT 자유 (기존) | 1.78 / 5.63 | 1.78 / 5.61 | 2.66 | 1,488 | 560 | — |
| STAGE SFT + xgrammar (기존) | 1.92 / 5.86 | 1.89 / 5.82 | 2.34 | 1,387 | 593 | 19.8 |

- **읽기.** thinking을 끄면 base 자유의 지연·처리량은 SFT와 사실상 같다(1.82 vs 1.78 s, 2.52 vs 2.66 예제/s). 즉 실험 6의 "SFT가 base보다 빠르다"는 thinking 아티팩트였고 표에서 제거한다. 남는 사실은 두 가지다: (1) xgrammar는 같은 모델에서 batch=1 지연 +0.2 s(약 +10%), batch=32 처리량 −15%, 스키마당 컴파일 19~20 ms의 비용이 있고, 미지원 스키마 6.2%가 있다. (2) SFT는 이 비용 없이 xgrammar 이상의 구조 준수(SCR 97.6 vs 95.1)와 더 높은 값 정확도(VA 84.7 vs 67.1)를 낸다. 논문 비용 단락은 "SFT는 추론 시 추가 비용이 0이고 커버리지 구멍이 없다"로 한정하고, 절대 지연 우위는 주장하지 않는다.
- 산출물: `outputs/inference_cost/base_nothink_{free,xgrammar}_{cold,warm}_b1.jsonl`, `_throughput_b32.jsonl`, `summary.csv`.
- **실험 8 완료.** 4개 재실행 모두 끝났다. 논문 수정 목록: (1) 본문 Table base Qwen3-4B 행을 851 non-thinking 값(EMR 38.2, VA 69.1, 파싱 실패 0.5%)으로 교체, (2) thinking 행에 기댄 문장("PFR 39.95→0.35", "VA 45.46→90.69", "SFT가 base보다 빠르다") 정정, (3) DeepJSONEval base 행 설정 확인, (4) thinking 행은 Appendix "thinking-mode baseline"으로 보존.
### 2026-09-03 — 실험 9 CORD 레이아웃 렌더링 (완료; zero-shot 음성 결과)

- `A=좌표 기반 행 렌더링`, `A+B=필드 설명`, `A+B+C=고정 train 1-shot`의 세 입력 변형을 만들고, Qwen3-4B/Qwen2.5-3B의 base/STAGE SFT × 자유/xgrammar를 각각 CORD test 100개에서 실행했다(24조건). Qwen3 base에는 `--no-thinking`을 적용했다.
- Qwen3 VA(%)는 A에서 base free/xgrammar **82.23/81.81**, SFT free/xgrammar **70.29/67.59**; A+B에서 **80.35/80.27**, **71.67/69.45**; A+B+C에서 **75.29/75.83**, **72.84/72.69**였다. Qwen2.5도 A의 base free/xgrammar **60.55/61.04**가 SFT **59.28/59.29**보다 높았고, A+B+C는 one-shot이 자유 생성에 특히 불리했다(base 12.81, SFT 43.52).
- 따라서 최고 SFT VA 72.84는 Qwen3 base+xgrammar 기준 약 81.5에 못 미쳤다. 행 렌더링은 base에는 유용했으나 STAGE의 CORD 위치 기반 배정 오류를 닫지 못했으므로, CORD zero-shot은 Appendix 음성 결과로 남기고 실험 10 적응 곡선으로 진행했다. 산출물: `outputs/cord_v2_layout/`.

### 2026-09-03 — 실험 10 CORD 적응 곡선 (완료; STAGE 초기화 음성 결과)

- 최고 zero-shot 입력(A+B+C)을 고정하고, CORD train을 seed 42로 섞어 50/200/800개 접두 집합으로 만들었다. Qwen3 base와 STAGE SFT 각각에서 LoRA rank 16, 3 epoch를 학습했다. 8k 입력에서 batch 4는 H200 메모리를 넘어서 batch 1 / gradient accumulation 16으로 바꿔 유효 배치 크기 16을 유지했다. validation/test는 각 100개이며 자유 디코딩으로 평가했다.

| Init | Train n | validation VA / EMR / PFR | test VA / EMR / PFR |
|---|---:|---:|---:|
| base | 50 | 75.15 / 16.0 / 95.0 | 74.72 / 18.0 / 100.0 |
| STAGE SFT | 50 | 74.32 / 13.0 / 100.0 | 74.08 / 18.0 / 100.0 |
| base | 200 | 88.14 / 48.0 / 100.0 | 84.72 / 40.0 / 99.0 |
| STAGE SFT | 200 | 74.67 / 31.0 / 91.0 | 73.46 / 26.0 / 93.0 |
| base | 800 | 92.52 / 60.0 / 100.0 | 90.08 / 56.0 / 100.0 |
| STAGE SFT | 800 | 89.79 / 59.0 / 98.0 | 86.00 / 50.0 / 98.0 |

- 판정: STAGE 초기화가 모든 크기에서 base를 이기지 못했다(50 test −0.64pt, 200 −11.26pt, 800 −4.08pt VA). 따라서 data-efficient adaptation 주장은 성립하지 않으며 Appendix 음성 결과로 남긴다. 산출물: `outputs/cord_adaptation/`; 재현 데이터 생성: `benchmark/prepare_cord_adaptation.py`.

### 2026-09-03 — 실험 10b CORD long-context filtered slice (완료; 범위가 한정된 양성)

- **공정한 학습 대조.** `train_50`의 처음 50개를 4회 반복하고, 원래 STAGE 합성 형식 replay 200개를 더한 400개 고정 SFT 파일을 만들었다. Qwen3-4B base와 STAGE-Qwen3-4B-SFT 모두 LoRA r16/alpha 32/all-linear, 3 epoch, lr 2e-5, batch 1 × accumulation 16, max length 8192, seed 42로 학습했다. 초기 체크포인트만 다르다. `benchmark/build_cord_stage_replay_mix.py`가 학습 파일을 재생성한다.
- **동일한 추론 대조.** 양쪽에 CORD A+B+C test/validation 100개, `--no-thinking`, seed 42, xgrammar JSON 제약을 동일하게 적용했다. 자유 생성의 탐욕적 JSON 추출은 유효 JSON 뒤의 여분 `}`를 무효로 만들었으므로, `inference.py`를 첫 balanced JSON object를 읽도록 고쳤다. 이 수정은 두 arm에 동일하게 적용했다.
- **전체 결과는 음성이다.** test-100 전체 VA/EMR은 base **85.76/48.0**, STAGE **84.00/39.0**이다. 따라서 CORD 전체 성능 우위나 일반적인 adaptation 우위를 주장하지 않는다.
- **사전 관측 가능한 long-context slice에서는 양성이다.** validation-100의 `user_prompt` 문자 길이 75백분위(6,826)를 label을 읽지 않고 고정했다. validation에서 길이 `>6,826`인 24개는 base 83.51, STAGE 83.95 VA(+0.44pt)였다. 독립 test-100의 동일 규칙 24개에서 base **69.01 VA / 4.17 EMR**, STAGE **72.47 VA / 12.50 EMR**로, STAGE가 **+3.46pt VA, +8.33pt EMR** 앞섰다. 양쪽 PFR/SCR은 모두 100.0이고 NR은 0.0이다.
- **판정과 한계.** 이 결과는 긴 OCR 문맥이라는 명시적 CORD 하위 분포에서만 성립하는 양성이고, n=24의 작은 slice다. 전체 CORD 음성 결과를 덮어쓰지 않는다. 필터 집계 재현: `benchmark/evaluate_cord_long_context_filter.py --validation-reference outputs/cord_stage_replay/50r4_s200/base_xgrammar_validation.jsonl --base outputs/cord_stage_replay/50r4_s200/base_xgrammar_test.jsonl --stage outputs/cord_stage_replay/50r4_s200/stage_xgrammar_test.jsonl`.

## 실험 11 — STAGE 파이프라인의 source-grounded dialogue-state extension (2026-09-03 시작, 진행 중)

**목표.** 실험 7에서 STAGE-SFT가 SGD 대화 상태 추적에 진 원인(언급 안 된 슬롯을 채우는 coverage bias)을 **STAGE 방법론으로 만든 추가 데이터**로 고쳐 base를 넘긴다. 인프라: 이 계정에서 블랙웰 노드는 보이지 않아(노드 목록 권한 없음, 보이는 노드는 `h200-03-w-50a0` 하나) 이전 예약 pod의 H200 2장을 쓴다.

**음성 원인 재확인.** 실험 7 파일럿에서 SFT 환각 슬롯률 54.0%(base 26.5%). STAGE 학습 스키마의 객체 95.5%가 all-required이고 oneOf/anyOf 1.4%라 "소스에 없는 필드는 비워둔다"를 데이터가 가르치지 않는다. 또 base의 SGD 점수는 thinking 유무에 따라 다르다: thinking 켬 standard 28.6 / explicit 30.2, **thinking 끔 19.1 / 26.5**(2026-09-03 재실행). 양성 판정은 둘 중 높은 쪽(explicit 30.2)을 기준으로 한다.

**데이터 생성 (STAGE 원칙 그대로).**
1. 소스: STAGE 보고서 20,713개 안의 Markdown 표에서 (헤더, 행) 쌍을 평평한 레코드로 추출 (`benchmark/stage_dialog/extract_records.py`). 영어 헤더·셀, 열 4~10개, 셀 길이 2~60자. 표 39,172개 중 5,898개 적합, 레코드 6,000개 샘플.
2. 생성: Qwen3-4B-Instruct-2507(vLLM)이 레코드마다 USER/SYSTEM 교대 대화(8~14줄)를 쓴다. 열의 50~80%를 "반드시 말할 열"로 무작위 지정하고, 사용자는 그 값을 **철자 그대로** 세 턴 이상에 나눠 말하며, 나머지 열의 값은 어느 화자도 말하지 않는다 (`generate_dialogs.py`).
3. 검증(필터): 지정 값 각각이 USER 턴에 verbatim 존재, 미지정 값은 대화 전체에 부재(토큰 경계 일치), 화자 교대, 첫 언급 턴이 2개 이상. 실패는 수정 없이 폐기. 결과 **6,000 → 3,525 대화 통과(58.8%)**, 탈락은 대부분 미지정 값 누출.
4. 예제화: 대화마다 사용자 턴 절단점 최대 3개에서 상태(그 시점까지 언급된 열)를 gold로 하고, SGD 평가와 같은 두 형식(standard: 언급 슬롯만 / explicit: 전체 required + `"no output"`)으로 프롬프트를 만든다. 18,096개 생성 → 대화당 절단점 2개, 형식 균형으로 4,000개 사용.
5. 믹스: 원본 STAGE train(HF `boradorish/text-to-json-benchmark`) 3,000개 + source-grounded dialogue-state 예제 4,000개 = 7,000개 (`build_mix.py`, 6,000토큰 초과 STAGE 예제 1,420개 제외).

**학습.** STAGE-Qwen3-4B-SFT에서 LoRA r16 α32 all-linear, lr 1e-4, 2 epoch, batch 1 × 누적 16, max_len 6144, gradient checkpointing (`train_toolfew_lora.py --gradient-checkpointing` 추가). 어댑터: `outputs/stage_dialog/lora_stage_sft_mix`. 학습 데이터에 SGD는 전혀 쓰지 않았다.

**평가 계획.** (1) SGD 파일럿 100턴 standard/explicit → base(thinking 켬/끔), STAGE-SFT와 비교. (2) 양성이면 `sgd_full_{standard,explicit}.jsonl` 2,000턴 본실행. (3) STAGE-Eval 200개로 회귀 확인(STAGE-SFT 대비). (4) 가능하면 base + 같은 믹스 LoRA(초기화 대조). 판정: explicit JGA > 30.2이고 환각 슬롯률 < base이면 양성.

**파일럿 결과 (2026-09-03 12:04, SGD 100턴, standard 형식, 전체 대화 이력 입력) — 양성.**

| Qwen3-4B | JGA | 슬롯 정확도 | 환각 슬롯률 | 누락 슬롯률 | seen / unseen JGA |
|---|---:|---:|---:|---:|---:|
| base, thinking 켬 | 28.6 | 82.9 | 26.5 | 7.8 | 28.2 / 29.0 |
| base, thinking 끔 | 19.1 | 84.5 | 38.5 | 5.4 | 20.2 / 18.1 |
| STAGE SFT | 7.6 | 84.8 | 54.0 | 0.6 | 6.7 / 8.6 |
| **STAGE SFT + dialogue-state extension LoRA** | **38.6** | 81.5 | **22.8** | 9.4 | 36.4 / 40.8 |

- JGA가 base(thinking) 대비 +10.0pt, STAGE-SFT 대비 +31.0pt이고 환각 슬롯률은 base보다 낮다(22.8 vs 26.5). 학습에 SGD를 쓰지 않았고, unseen 서비스에서 40.8로 seen(36.4)보다 높아 스키마 일반화가 유지된다. 이 경로는 **원래 정의의 DST**(전체 이력, carry-over 포함)에서 얻은 결과라 아래 single-turn 부분집합 결과와는 과제 정의가 다르다.
- explicit 파일럿과 STAGE-Eval 회귀 검사는 같은 GPU에 다른 학습이 올라와 vLLM 메모리 비율 0.9 요구에 걸려 첫 시도가 실패했고, 비율 0.45로 2,000턴 본실행(source-grounded dialogue-state extension, base thinking/non-thinking, STAGE-SFT × standard/explicit)과 함께 재실행 중이다(`logs/eval_full.log`). base 초기화 대조군(Qwen3-4B base + 같은 믹스 LoRA, `outputs/stage_dialog/lora_base_mix`)도 GPU0에서 학습 중.

**본실행 결과 (2026-09-03 14:13, SGD test 2,000턴, seen/unseen 서비스 50:50, 전체 대화 이력 입력, 공식 SGD metric).** LoRA 학습 데이터에 SGD는 포함되지 않았다.

| Qwen3-4B | standard JGA / 환각 | explicit JGA / 환각 |
|---|---:|---:|
| base, thinking 끔 (논문 기준 base) | 19.2 / 40.4 | 32.8 / 24.9 |
| base, thinking 켬 | 36.4 / 23.5 | 37.5 / 22.5 |
| STAGE SFT (논문 모델) | 7.3 / 55.0 | 17.9 / 34.9 |
| **STAGE SFT + STAGE-Dialog LoRA** | 34.9 / 24.0 | 36.2 / 16.7 |
| **base + STAGE-Dialog LoRA** (초기화 대조) | **39.8 / 15.0** | **38.8 / 17.5** |

- **양성 판정.** (1) 논문 기준 base(thinking 끔) 대비 STAGE-SFT+STAGE-Dialog는 standard +15.7pt, explicit +3.4pt. (2) STAGE-SFT 자체 대비 +27.6pt(7.3→34.9), 환각 55.0→24.0으로 coverage bias가 교정됐다. (3) 같은 데이터를 base에 학습한 대조군은 thinking base보다도 standard +3.4pt, explicit +1.3pt 높고 환각은 23.5→15.0으로 줄었다. 2,000턴에서 JGA의 표준오차는 약 1.1pt이므로 +3.4pt는 유의하고, STAGE-SFT 초기화와 thinking base의 차이(−1.5 / −1.3)는 오차 범위 안이다.
- **초기화 효과.** STAGE-SFT 초기화는 base 초기화보다 standard −4.9pt, explicit −2.6pt 낮다. 단일 턴 부분집합에서의 codex 결론(base 초기화 우위)과 방향이 같다. 따라서 논문 주장은 "STAGE 데이터 생성 방법론을 대화로 확장하면 SLM의 상태 추적이 개선된다(데이터 효과)"로 잡고, "STAGE-SFT 체크포인트가 좋은 초기화다"는 주장은 하지 않는다.
- 파일럿 100턴(base 28.6)은 2,000턴(36.4)보다 base를 과소평가했다. 100턴 파일럿 수치는 인용하지 않는다.
- 진행 중: STAGE-SFT explicit 2,000턴, STAGE-Eval 200개 회귀 검사(STAGE-SFT / +STAGE-Dialog / base+STAGE-Dialog), 그리고 대화 데이터 비중을 높인 2차 믹스(STAGE-Dialog 8,000 + STAGE 2,000, `stage_dialog_mix_v2.jsonl`)의 STAGE-SFT 초기화 학습·평가(`lora_stage_sft_mix_v2`).

**회귀 검사 (STAGE-Eval 첫 200개, 자유 디코딩).** STAGE-SFT + STAGE-Dialog LoRA는 원래 STAGE-SFT와 같다(PFR 100 / EMR 60.5→61.0 / SCR 99.0 / VA 84.5→84.5). 즉 in-distribution 성능을 잃지 않고 SGD만 고쳐졌다. base + STAGE-Dialog는 STAGE-Eval VA 78.8, EMR 50.5로 STAGE-SFT(84.5)보다 낮지만 base(69.3)보다는 높다. 두 과제를 함께 보면 STAGE-SFT 초기화가 균형이 맞고, SGD 단독으로는 base 초기화가 낫다. 851개 전체 회귀(자유 디코딩, 아래 표)에서도 같다.

| STAGE-Eval 851 | PFR | EMR | SCR | NR | VA |
|---|---:|---:|---:|---:|---:|
| base, thinking 끔 | 99.5 | 38.2 | 91.1 | 5.7 | 69.1 |
| STAGE SFT | 99.6 | 63.3 | 97.6 | 1.4 | 84.9 |
| **STAGE SFT + STAGE-Dialog LoRA** | 100.0 | 63.7 | 98.0 | 1.2 | 84.6 |
| base + STAGE-Dialog LoRA | 100.0 | 49.2 | 93.2 | 4.6 | 76.6 |

STAGE-SFT + STAGE-Dialog는 in-distribution에서 STAGE-SFT와 동일(EMR +0.4, VA −0.3, 오차 범위)하고 SGD에서는 7.3→34.9다. base + STAGE-Dialog는 SGD는 가장 높지만 STAGE-Eval VA 76.6으로 STAGE-SFT보다 8.3pt 낮다. 두 과제를 동시에 만족하는 모델은 STAGE-SFT + STAGE-Dialog다.

**2차 믹스 결과 (2026-09-03 16:00, SGD 2,000턴) — 양성 확정.** 대화 데이터 비중을 올린 믹스(STAGE-Dialog 8,000 + STAGE 2,000, 대화당 절단점 3개, `stage_dialog_mix_v2.jsonl`)로 STAGE-SFT 위에 같은 설정(LoRA r16, 2 epoch, lr 1e-4)으로 학습한 `lora_stage_sft_mix_v2`.

| Qwen3-4B, SGD 2,000턴 | standard JGA / 환각 | explicit JGA / 환각 |
|---|---:|---:|
| base, thinking 켬 (가장 강한 기준선) | 36.4 / 23.5 | 37.5 / 22.5 |
| base, thinking 끔 (논문 기준 base) | 19.2 / 40.4 | 32.8 / 24.9 |
| STAGE SFT | 7.3 / 55.0 | 17.9 / 34.9 |
| STAGE SFT + STAGE-Dialog v1 (4k+3k) | 34.9 / 24.0 | 36.2 / 16.7 |
| **STAGE SFT + STAGE-Dialog v2 (8k+2k)** | **41.0 / 15.3** | **43.0 / 9.0** |
| base + STAGE-Dialog v1 | 39.8 / 15.0 | 38.8 / 17.5 |

- v2는 thinking base 대비 standard **+4.6pt**, explicit **+5.5pt**(표준오차 약 1.1pt), 환각 슬롯률은 23.5→15.3, 22.5→**9.0**. 논문 기준 base(thinking 끔) 대비 +21.8 / +10.2. STAGE-SFT 대비 +33.7 / +25.1. seen/unseen: standard 47.2/34.7, explicit 47.2/38.8.
- v1→v2 개선(+6.1 / +6.8)은 대화 데이터 양(4k→8k)과 비중(57%→80%)에서 온다. 누락 슬롯률은 12.7 / 18.2로 v1보다 높아졌으므로(비우는 쪽으로 기울어짐), 본문에는 JGA와 환각·누락을 함께 표기한다.
- 진행 중: v2의 STAGE-Eval 851 회귀(`stage_eval851_stage_dialog_v2`), base 초기화 + v2 믹스 대조군(`lora_base_mix_v2`)의 SGD 2,000턴과 STAGE-Eval 851 (`logs/v2_followup.log`).

**병행 경로 — source-grounded single-turn state extraction (완료; 양성).** `prepare_sgd.py --context latest-user --filter latest-user-grounded --select-eligible-first`는 예측을 보지 않고 target USER 발화에 non-empty gold가 모두 정규화 후 문자적으로 존재하는 turn만 유지한다. carry-over state·system-proposed value는 제외하므로, 이는 전체 DST가 아닌 source-grounded single-turn task다. SGD test의 적격 4,672/46,116개에서 서비스 균등·seen/unseen 50:50, seed 42로 2,000개를 고정했다.

SGD 데이터·서비스명·문항을 읽지 않는 `generate_stategrounded_sft_data.py`로 invented service/slot/value 기반 합성 2,000개를 생성했다. 단일 USER 발화, STAGE report+schema prompt, all-required와 미언급 슬롯의 `"no output"`을 사용했으며, STAGE Qwen3-4B SFT에서 LoRA r16, 3 epoch, lr 2e-5, effective batch 16으로 continuation했다.

| Qwen3-4B, explicit (n=2,000) | JGA | Slot accuracy | Hallucinated | Missing |
|---|---:|---:|---:|---:|
| base (thinking off) | 61.19 | 67.25 | **1.51** | 30.51 |
| STAGE + source-grounded continuation | 75.14 | 85.15 | 7.14 | 12.90 |
| **base + source-grounded continuation** | **84.04** | **92.20** | 3.68 | **5.42** |

Seen services JGA/slot accuracy는 52.38/56.71 → **76.34/81.13**, unseen은 70.00/77.79 → **73.94/89.17**이다. 따라서 base 대비 JGA +13.95pt, slot accuracy +17.90pt의 양성 결과이나, 환각 슬롯률은 +5.63pt이므로 전체-history SGD 음성 결과와 이 제한된 양성 결과를 모두 기록한다. 산출물: `outputs/sgd/qwen3_4b_{base,stategrounded}_explicit_latest_user_grounded_full_*`; adapter: `models/STAGE-Qwen3-4B-StateGroundedSFT/`.

**초기화 대조 (완료).** 같은 합성 2,000개, LoRA r16, 3 epoch, lr 2e-5, effective batch 16, seed 42를 **Qwen3-4B base**에서 동일하게 학습해 `Qwen3-4B-StateGroundedSFT`를 만들었다. 이는 STAGE-SFT가 없는 ``no output`` 데이터만의 효과를 분리하는 대조다. 결과는 JGA **84.04**, slot accuracy **92.20**으로 STAGE 초기화 continuation(75.14/85.15)을 각각 +8.90pt/+7.05pt 앞섰다; seen JGA/slot accuracy 84.52/92.80, unseen 83.57/91.59이다. 따라서 이 source-grounded subset의 양성은 **추가 state-extraction 데이터의 효과**이며, 기존 STAGE 초기화가 추가 이득을 준다고 주장할 수 없다. base continuation도 raw base보다 JGA +22.85pt, missing slot rate −25.09pt로 개선됐고, 환각은 +2.17pt 증가했다. 산출물: `outputs/sgd/qwen3_4b_basegrounded_explicit_latest_user_grounded_full_*`; adapter: `models/Qwen3-4B-StateGroundedSFT/`.

## 인프라 메모

- 추론: vLLM, 1× H200 (설정은 논문 Appendix C 참조: temp 0.6, top-p 1.0, max_new 3100, max_len 8192, seed 42)
- 기존 코드: `benchmark/inference.py` (추론), `benchmark/evaluate.py` (채점), `src/utils/vllm_inference.py`
- 체크포인트 위치는 실험 시작 전 확인 필요

### 2026-09-03 — Table-grounded STAGE 재학습 데이터 준비 (학습·평가 전)

- 기존 STAGE report의 서술 문단에는 설명용/가상 표가 섞여 있어, `## Sheet:` heading 바로 아래의 Markdown 표만 source로 파싱했다. Markdown·TSV·HTML은 이 표 셀을 그대로 변환한 표현이며, `gold_json`으로 source를 생성하지 않는다.
- 긴 문서의 표 중심 추출을 겨냥해 원 report가 3,500자 이상인 항목만 후보로 삼았다. 그 뒤 target의 모든 primitive 값이 실제 Sheet 표 문자열에 literal로 존재하는 항목만 남겼다(coverage=1.0). 18,560개 중 표 부재 10,525개, table-grounded 값 불완전 2,159개, 짧은 report 952개, 표가 너무 긴 7개를 제외하고 **적격 원천 4,913개 전부**를 사용했다(seed 42는 출력 셔플에만 사용).
- 각 source는 Markdown table / TSV / HTML table 3가지 실제 source 표현으로 만들었고, 전체 schema 3개와 결정론적 top-level field subset을 요청하는 Markdown/TSV 2개를 추가했다. subset은 같은 source에서 **요청한 필드만** JSON으로 내도록 학습시켜, schema에 보이는 모든 field를 채우려는 coverage bias를 줄이는 대조 과제다. 총 **18,503 examples / 4,913 independent sources / 195MB JSONL**다.
- 산출물: `data/sft/stage_table_grounded_all.jsonl` (ignored), metadata `data/sft/stage_table_grounded_all.metadata.json` (ignored), 재생성 `benchmark/build_table_grounded_stage_sft.py`. HF private dataset: `boradorish/STAGE-Table-Grounded-SFT` (`data/train.jsonl`, metadata, card); 업로드 재현: `benchmark/upload_table_grounded_stage_sft_to_hf.py`. 아직 모델 학습·DocuBench/Kleister 평가를 하지 않았으므로 성능 주장은 없다.

### 2026-09-03 — Source-grounded dialogue-state data packaging

- 기존 생성·검증 결과(6,000 spreadsheet-row records → 3,525 통과 대화 → 18,096 state examples)를 shared chat SFT 형식으로 패키징했다. standard(언급 슬롯만)와 explicit(전 슬롯 + `no output`)이 각각 9,048개다.
- 이 데이터는 SGD에서 생성·필터·학습 샘플을 전혀 사용하지 않는다. 대화 생성 뒤 지정 값이 USER 발화에 verbatim으로 존재하고, 미지정 record 값은 대화 전체에 부재하며, 화자 교대가 성립할 때만 보존한 원래 검증 결과를 사용한다.
- 산출물: `data/sft/source_grounded_dialogue_state_18096.jsonl` (ignored), 재현 패키징 `benchmark/prepare_dialogue_state_sft.py`, 업로드 `benchmark/upload_dialogue_state_sft_to_hf.py`, HF private dataset `boradorish/STAGE-Dialogue-State-SFT`.
