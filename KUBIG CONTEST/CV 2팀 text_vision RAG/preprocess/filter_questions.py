"""
[본실험 1] 문항 필터링 — 선정된 60문서의 single-page 문항 추출

파일럿의 01_filter.py와 차이:
  - 문서를 자동 선정하지 않는다. docs_60.txt에 확정된 목록을 읽는다.
  - PDF 실제 페이지 수로 범위 검증한다 (evidence 최대 페이지가 아니라).

입력:  data/samples.json      (parquet 변환본, 1091문항)
       data/docs_60.txt       (선정된 60문서, 한 줄에 하나)
       data/raw/documents/    (PDF)
출력:  data/full_samples.json (266문항 예상)
       data/full_docs.txt     (렌더링용, docs_60.txt와 동일 내용)

실행: python 01_filter_full.py
"""
import json, ast
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

ROOT     = Path(__file__).resolve().parent.parent
SAMPLES  = ROOT / "data/samples.json"
DOCLIST  = ROOT / "data/docs_60.txt"
PDF_DIR  = ROOT / "data/raw/documents"
OUT      = ROOT / "data"

for p in (SAMPLES, DOCLIST, PDF_DIR):
    if not p.exists():
        raise SystemExit(f"경로 없음: {p}")

docs = [d.strip() for d in DOCLIST.read_text().split("\n") if d.strip()]
print(f"선정 문서 {len(docs)}개")

# ── PDF 페이지 수 (범위 검증에 필요) ──────────────────────
pages = {}
missing = []
for d in docs:
    p = PDF_DIR / d
    if not p.exists():
        missing.append(d); continue
    pages[d] = pymupdf.open(p).page_count
if missing:
    print(f"!! PDF 없음 {len(missing)}개")
    for m in missing: print("   ", m)
    raise SystemExit("PDF를 먼저 확보할 것")
print(f"총 {sum(pages.values()):,}페이지  (3 DPI 렌더링 {sum(pages.values())*3:,}장)")


def parse_list(v):
    """evidence_pages / evidence_sources 모두 문자열로 저장돼 있다."""
    if isinstance(v, list): return v
    if isinstance(v, str):
        try: return json.loads(v.replace("'", '"'))
        except Exception:
            try: return ast.literal_eval(v)
            except Exception: return [v]
    return list(v)


samples = json.loads(SAMPLES.read_text())
print(f"\n전체 문항 {len(samples)}")

# ── 필터 1: 선정 문서 소속 ────────────────────────────────
sel = set(docs)
s1 = [x for x in samples if x["doc_id"] in sel]
print(f"1. 선정 문서 소속       {len(samples)} → {len(s1)}")

# ── 필터 2: 답변 가능 ────────────────────────────────────
s2 = []
for x in s1:
    try:
        pgs = [int(p) for p in parse_list(x["evidence_pages"])]
    except Exception:
        continue
    if not pgs: continue
    if x.get("answer") in (None, "", "Not answerable"): continue
    s2.append({**x, "_pages": pgs})
print(f"2. 답변 가능            {len(s1)} → {len(s2)}")

# ── 필터 3: 페이지 범위 정상 ──────────────────────────────
# 1-based 확정(파일럿 검증 완료). 0이거나 실제 페이지 수를 넘으면 데이터 오류.
s3, bad = [], []
for x in s2:
    n = pages[x["doc_id"]]
    if min(x["_pages"]) < 1 or max(x["_pages"]) > n:
        bad.append((x["doc_id"], x["_pages"], n)); continue
    s3.append(x)
print(f"3. 페이지 범위 정상     {len(s2)} → {len(s3)}   (제외 {len(bad)}건)")
for d, p, n in bad[:10]:
    print(f"     {d[:44]:44} pages={p} (문서 {n}p)")

# ── 필터 4: single-page ──────────────────────────────────
# Recall@5 해석을 명확히 하기 위함. multi-hop은 검색 실패와 추론 실패가 섞인다.
s4 = [x for x in s3 if len(x["_pages"]) == 1]
print(f"4. single-page          {len(s3)} → {len(s4)}")

# ── 저장 ──────────────────────────────────────────────────
# question_id는 이 파일의 인덱스. A·B가 그대로 돌려줘야 한다.
out = []
for i, x in enumerate(s4):
    out.append({
        "question_id": i,
        "doc_id": x["doc_id"],
        "doc_type": x.get("doc_type"),
        "question": x["question"],
        "answer": x["answer"],
        "answer_format": x.get("answer_format"),
        "evidence_pages": x["_pages"],          # 1-based
        "evidence_sources": [t for t in parse_list(x["evidence_sources"]) if t] or ["(없음)"],
        "n_pages": pages[x["doc_id"]],
    })
(OUT / "full_samples.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
(OUT / "full_docs.txt").write_text("\n".join(docs))

# ── 구성 확인 ─────────────────────────────────────────────
print(f"\n저장: data/full_samples.json ({len(out)}문항)")
print("      data/full_docs.txt")

print("\n[doc_type]")
for k, v in Counter(x["doc_type"] for x in out).most_common():
    nd = len({x["doc_id"] for x in out if x["doc_type"] == k})
    print(f"  {str(k):<34}{nd:>3}문서 {v:>4}문항")

print("\n[근거유형]  한 문항이 복수 유형을 가질 수 있어 합계가 문항 수를 초과한다")
c = Counter()
for x in out: c.update(x["evidence_sources"])
tot = sum(c.values())
for k, v in c.most_common():
    print(f"  {k:<30}{v:>4}  ({v/tot:>5.1%})")

print("\n[answer_format]")
for k, v in Counter(x["answer_format"] for x in out).most_common():
    print(f"  {str(k):<10}{v:>4}  ({v/len(out):>5.1%})")

print("\n★ evidence_pages는 1-based. A·B에게 명시할 것.")
