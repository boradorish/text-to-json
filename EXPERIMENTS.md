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

## 인프라 메모

- 추론: vLLM, 1× H200 (설정은 논문 Appendix C 참조: temp 0.6, top-p 1.0, max_new 3100, max_len 8192, seed 42)
- 기존 코드: `benchmark/inference.py` (추론), `benchmark/evaluate.py` (채점), `src/utils/vllm_inference.py`
- 체크포인트 위치는 실험 시작 전 확인 필요
