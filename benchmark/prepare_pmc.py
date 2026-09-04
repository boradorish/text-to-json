"""Build a real-world long-document benchmark from PubMed Central open-access articles (JATS XML).

Each article's XML is rendered to plain text the way a reader sees it (title, author line,
affiliations, abstract, body sections, reference list) and the gold JSON is taken from the
same XML's structured metadata: title, journal, year, keywords, authors (surname, given
names, affiliation text), and references (first-author surname, year, title). Nested arrays of
repeated records (authors, references) and verbatim short values make it a source-grounded
extraction task on 10k-30k-token documents. Fetched with NCBI E-utilities (open-access
subset, research articles, CC licenses); the XML is cached under --cache.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PROMPT_PREFIX = ("Extract the article metadata from the following scientific article text according to the JSON Schema. "
                 "Copy values as they appear in the text. Return exactly one JSON object.\n\n")


def text_of(el) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def fetch(url: str, retries: int = 4) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(2 + 2 * i)
    raise RuntimeError(url)


def search_ids(n: int, year: int) -> list[str]:
    q = f'open access[filter] AND {year}[pdat] AND "research article"[pt]'
    url = f"{EUTILS}/esearch.fcgi?db=pmc&term={urllib.parse.quote(q)}&retmax={n}&retmode=json"
    return json.loads(fetch(url))["esearchresult"]["idlist"]


def parse(article) -> tuple[str, dict] | None:
    front = article.find("front")
    if front is None:
        return None
    meta = front.find("article-meta")
    jm = front.find("journal-meta")
    title = text_of(meta.find(".//title-group/article-title"))
    journal = text_of(jm.find(".//journal-title")) if jm is not None else ""
    year = ""
    for pd in meta.findall("pub-date"):
        y = pd.find("year")
        if y is not None and text_of(y):
            year = text_of(y)
            if pd.get("pub-type") in ("epub", "ppub") or pd.get("date-type") == "pub":
                break
    affs = {}
    for aff in meta.iter("aff"):
        aid = aff.get("id") or ""
        parts = []
        for node in aff.iter():
            if node.tag == "label":
                continue
            if node.text and node is not aff:
                parts.append(node.text)
            elif node is aff and node.text:
                parts.append(node.text)
            if node.tail and node is not aff:
                parts.append(node.tail)
        affs[aid] = " ".join(" ".join(parts).split())
    authors = []
    for c in meta.iter("contrib"):
        if c.get("contrib-type") != "author":
            continue
        name = c.find("name")
        if name is None:
            continue
        sur, giv = text_of(name.find("surname")), text_of(name.find("given-names"))
        rids = [x.get("rid") for x in c.findall("xref") if x.get("ref-type") == "aff"]
        aff_txt = [affs[r] for r in rids if r in affs and affs[r]]
        if not aff_txt and len(affs) == 1:
            aff_txt = list(affs.values())
        authors.append({"surname": sur, "given_names": giv, "affiliations": aff_txt})
    if not title or not authors:
        return None
    keywords = [text_of(k) for k in meta.iter("kwd") if text_of(k)]
    abstract = text_of(meta.find("abstract"))
    body = article.find("body")
    body_parts = []
    if body is not None:
        for sec in body.iter("sec"):
            t = sec.find("title")
            if t is not None and text_of(t):
                body_parts.append("\n" + text_of(t) + "\n")
            for p in sec.findall("p"):
                body_parts.append(text_of(p))
        if not body_parts:
            body_parts = [text_of(p) for p in body.iter("p")]
    refs, ref_lines = [], []
    back = article.find("back")
    if back is not None:
        for i, ref in enumerate(back.iter("ref")):
            cit = None
            for tag in ("element-citation", "mixed-citation", "citation"):
                cit = ref.find(tag)
                if cit is not None:
                    break
            if cit is None:
                continue
            rtitle = text_of(cit.find("article-title")) or text_of(cit.find("chapter-title")) or text_of(cit.find("source"))
            ryear = text_of(cit.find("year"))
            first = cit.find(".//surname")  # <name> or <string-name>
            fsur = text_of(first)
            names = [(text_of(n.find("surname")) + " " + text_of(n.find("given-names"))).strip() for n in cit.iter() if n.tag in ("name", "string-name")][:6]
            src = text_of(cit.find("source"))
            if not rtitle:
                continue
            refs.append({"first_author_surname": fsur, "year": ryear, "title": rtitle})
            line = f"{i+1}. " + (", ".join(names) + ". " if names else "") + f"{rtitle}. " + (f"{src}. " if src and src != rtitle else "") + (f"{ryear}." if ryear else "")
            ref_lines.append(" ".join(line.split()))
    if len(refs) < 5:
        return None
    author_line = ", ".join(f"{a['given_names']} {a['surname']}".strip() for a in authors)
    aff_lines = [v for v in affs.values() if v]
    text = "\n".join([title, "", author_line, ""] + aff_lines + ["", journal + (f" ({year})" if year else ""), "",
                      ("Keywords: " + "; ".join(keywords)) if keywords else "", "", "Abstract", abstract, ""] + body_parts + ["", "References"] + ref_lines)
    gold = {"title": title, "journal": journal, "year": year, "keywords": keywords, "authors": authors, "references": refs}
    return text, gold


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["title", "journal", "year", "keywords", "authors", "references"],
          "properties": {
              "title": {"type": "string", "description": "Title of the article"},
              "journal": {"type": "string", "description": "Name of the journal"},
              "year": {"type": "string", "description": "Publication year"},
              "keywords": {"type": "array", "items": {"type": "string"}, "description": "Author keywords listed in the article (empty if none)"},
              "authors": {"type": "array", "description": "All authors in order", "items": {"type": "object", "additionalProperties": False, "required": ["surname", "given_names", "affiliations"],
                          "properties": {"surname": {"type": "string"}, "given_names": {"type": "string"}, "affiliations": {"type": "array", "items": {"type": "string"}, "description": "Affiliation lines of this author as printed"}}}},
              "references": {"type": "array", "description": "Every entry of the reference list in order", "items": {"type": "object", "additionalProperties": False, "required": ["first_author_surname", "year", "title"],
                             "properties": {"first_author_surname": {"type": "string"}, "year": {"type": "string"}, "title": {"type": "string"}}}}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--cache", type=Path, default=ROOT / "benchmark" / "data" / "realworld" / "pmc_xml")
    ap.add_argument("--output", type=Path, default=ROOT / "benchmark" / "data" / "realworld" / "pmc_oa_2024.jsonl")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-chars", type=int, default=120000, help="skip documents whose rendered text is longer (keeps prompts under ~32k tokens)")
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    ids = search_ids(int(a.n * 1.8), a.year)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    n_out, lengths, n_auth, n_ref = 0, [], 0, 0
    with a.output.open("w", encoding="utf-8") as fh:
        for i in range(0, len(ids), 10):
            batch = ids[i:i + 10]
            cached = [a.cache / f"PMC{pid}.xml" for pid in batch]
            if not all(c.exists() for c in cached):
                raw = fetch(f"{EUTILS}/efetch.fcgi?db=pmc&id={','.join(batch)}&retmode=xml")
                root = ET.fromstring(raw)
                arts = list(root.iter("article"))
                for pid, art in zip(batch, arts):
                    (a.cache / f"PMC{pid}.xml").write_bytes(ET.tostring(art))
                time.sleep(0.4)
            for pid, c in zip(batch, cached):
                if not c.exists():
                    continue
                try:
                    parsed = parse(ET.fromstring(c.read_bytes()))
                except ET.ParseError:
                    parsed = None
                if parsed is None:
                    continue
                text, gold = parsed
                if len(text) > a.max_chars or len(text) < 5000:
                    continue
                prompt = f"{PROMPT_PREFIX}=== Report ===\n{text}\n\n=== JSON Schema ===\n{json.dumps(SCHEMA, indent=2)}"
                rec = {"stem": f"pmc_{n_out:03d}", "dataset": "pmc_oa_2024", "source_id": f"PMC{pid}", "user_prompt": prompt,
                       "gold_json": json.dumps(gold, ensure_ascii=False), "json_schema": json.dumps(SCHEMA), "text_chars": len(text),
                       "n_authors": len(gold["authors"]), "n_references": len(gold["references"])}
                if tok is not None:
                    n = len(tok(prompt)["input_ids"]); rec["prompt_tokens"] = n; lengths.append(n)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1; n_auth += len(gold["authors"]); n_ref += len(gold["references"])
                if n_out >= a.n:
                    break
            if n_out >= a.n:
                break
    print(f"wrote {n_out} articles ({n_auth} authors, {n_ref} references) to {a.output}")
    if lengths:
        lengths.sort(); print(f"prompt tokens p50={lengths[len(lengths)//2]} p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}")


if __name__ == "__main__":
    main()
