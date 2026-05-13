"""
glaiveai/glaive-function-calling-v2 와 NousResearch/hermes-function-calling-v1 에서
JSON 응답 샘플을 추출해 LLaMA-Factory 학습 데이터에 추가합니다.

사용법:
    python src/train/add_function_calling_data.py --inspect
    python src/train/add_function_calling_data.py
    python src/train/add_function_calling_data.py --glaive 5000 --hermes 2000
    python src/train/add_function_calling_data.py --output /root/LLaMA-Factory/data/custom-reasoning.json
    python src/train/add_function_calling_data.py --no-merge  # 기존 데이터 무시하고 새로 생성
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ─── Glaive ───────────────────────────────────────────────────────────────────

def _parse_glaive_chat(chat: str) -> list[dict]:
    """'USER: ...\nASSISTANT: ...' 형식을 role/content 딕셔너리 리스트로 파싱."""
    messages = []
    parts = re.split(r'\b(USER|ASSISTANT):\s*', chat)
    i = 1
    while i + 1 < len(parts):
        role = parts[i]
        content = parts[i + 1].replace('<|endoftext|>', '').strip()
        if content:
            messages.append({'role': role, 'content': content})
        i += 2
    return messages


def _extract_json_glaive(content: str) -> str | None:
    """<functioncall> 태그 또는 베어 JSON 추출 후 파싱 검증."""
    m = re.search(r'<functioncall>\s*(\{.*\})', content, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    try:
        stripped = content.strip()
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    return None


def convert_glaive(dataset, max_samples: int) -> list[dict]:
    results = []
    for row in tqdm(dataset, desc="Glaive 변환"):
        if len(results) >= max_samples:
            break
        system = (row.get('system') or '').strip()
        turns = _parse_glaive_chat(row.get('chat') or '')

        for i, turn in enumerate(turns):
            if turn['role'] != 'ASSISTANT':
                continue
            json_str = _extract_json_glaive(turn['content'])
            if json_str is None:
                continue
            # 바로 앞의 USER 메시지 찾기
            user_content = next(
                (turns[j]['content'] for j in range(i - 1, -1, -1) if turns[j]['role'] == 'USER'),
                None,
            )
            if not user_content:
                continue
            messages = []
            if system:
                messages.append({'role': 'system', 'content': system})
            messages.append({'role': 'user', 'content': user_content})
            messages.append({'role': 'assistant', 'content': json_str})
            results.append({'messages': messages})
            if len(results) >= max_samples:
                break
    return results


# ─── Hermes ───────────────────────────────────────────────────────────────────

def _extract_json_hermes(content: str) -> str | None:
    """<tool_call> 태그에서 JSON 추출. 다중 호출이면 None 반환(스킵)."""
    matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
    if len(matches) != 1:
        return None
    try:
        obj = json.loads(matches[0].strip())
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        return None


_HERMES_ROLE = {'system': 'system', 'human': 'user', 'gpt': 'assistant', 'tool': 'tool'}


def convert_hermes(dataset, max_samples: int) -> list[dict]:
    results = []
    for row in tqdm(dataset, desc="Hermes 변환"):
        if len(results) >= max_samples:
            break
        convs = row.get('conversations') or []

        for i, turn in enumerate(convs):
            if turn.get('from') not in ('gpt', 'assistant'):
                continue
            json_str = _extract_json_hermes(turn.get('value') or '')
            if json_str is None:
                continue
            # 이 턴까지의 메시지 구성
            messages = []
            for j in range(i):
                src = convs[j].get('from', '')
                role = _HERMES_ROLE.get(src, src)
                val = (convs[j].get('value') or '').strip()
                if val:
                    messages.append({'role': role, 'content': val})
            if not any(m['role'] == 'user' for m in messages):
                continue
            messages.append({'role': 'assistant', 'content': json_str})
            results.append({'messages': messages})
            if len(results) >= max_samples:
                break
    return results


# ─── Inspect ──────────────────────────────────────────────────────────────────

def inspect_dataset(name: str, n: int = 3) -> None:
    print(f"\n{'='*60}")
    print(f"데이터셋: {name}")
    print(f"{'='*60}")
    ds = load_dataset(name, split='train', streaming=True)
    for i, row in enumerate(ds):
        if i >= n:
            break
        print(f"\n--- 샘플 {i} ---")
        print(f"컬럼: {list(row.keys())}")
        for k, v in row.items():
            if k == 'conversations':
                print(f"  [conversations] ({len(v)}턴)")
                for turn in v:
                    role = turn.get('from', '?')
                    val = (turn.get('value') or '')
                    # gpt(assistant) 응답은 전체 출력해서 태그 형식 확인
                    limit = None if role == 'gpt' else 200
                    print(f"    [{role}] {val[:limit]}")
            else:
                print(f"  [{k}] {str(v)[:300]}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true', help='데이터셋 구조 출력만 (변환 안 함)')
    parser.add_argument('--glaive', type=int, default=5000, metavar='N', help='glaive 최대 샘플 수 (기본 5000)')
    parser.add_argument('--hermes', type=int, default=2000, metavar='N', help='hermes 최대 샘플 수 (기본 2000)')
    parser.add_argument('--output', default='/root/LLaMA-Factory/data/custom-reasoning.json', help='출력 JSON 경로')
    parser.add_argument('--no-merge', action='store_true', help='기존 파일 무시하고 새로 생성')
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset('glaiveai/glaive-function-calling-v2')
        inspect_dataset('NousResearch/hermes-function-calling-v1')
        return

    # ── 변환 ──
    print("glaive-function-calling-v2 로드 중...")
    glaive_ds = load_dataset('glaiveai/glaive-function-calling-v2', split='train')
    glaive_records = convert_glaive(glaive_ds, args.glaive)
    print(f"  → {len(glaive_records):,}개 추출")

    print("\nhermes-function-calling-v1 로드 중...")
    hermes_ds = load_dataset('NousResearch/hermes-function-calling-v1', split='train')
    hermes_records = convert_hermes(hermes_ds, args.hermes)
    print(f"  → {len(hermes_records):,}개 추출")

    new_records = glaive_records + hermes_records
    print(f"\n신규 합계: {len(new_records):,}개")

    # ── 기존 데이터와 병합 ──
    output_path = Path(args.output)
    if output_path.exists() and not args.no_merge:
        existing = json.loads(output_path.read_text(encoding='utf-8'))
        print(f"기존 데이터: {len(existing):,}개 → 병합 후: {len(existing) + len(new_records):,}개")
        combined = existing + new_records
    else:
        combined = new_records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n저장 완료: {output_path}  ({len(combined):,}개)")

    # ── 샘플 검증 출력 ──
    print("\n--- glaive 변환 샘플 (첫 1개) ---")
    if glaive_records:
        for m in glaive_records[0]['messages']:
            print(f"  [{m['role']}] {m['content'][:200]}")
    print("\n--- hermes 변환 샘플 (첫 1개) ---")
    if hermes_records:
        for m in hermes_records[0]['messages']:
            print(f"  [{m['role']}] {m['content'][:200]}")


if __name__ == '__main__':
    main()
