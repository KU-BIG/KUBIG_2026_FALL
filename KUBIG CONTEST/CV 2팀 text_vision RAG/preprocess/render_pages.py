"""
[본실험 2] PDF 렌더링 + 문서 특성 통계

파일럿의 02_render.py와 차이:
  - 60문서 1,780페이지 x 3 DPI = 약 5,340장. 중단 후 재실행이 가능하도록
    이미 만든 파일은 건너뛴다.
  - 진행률과 소요 시간을 출력한다.
  - 문서별 통계를 저장해 결과 해석에 쓴다.

입력:  data/full_docs.txt, data/raw/documents/
출력:  data/pages/{dpi}/{doc_id}/{page:03d}.png   ← 파일명 1-based
       data/page_stats.json

실행: python 02_render_full.py
      python 02_render_full.py 144      (특정 DPI만)
"""
import json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

ROOT     = Path(__file__).resolve().parent.parent
PDF_DIR  = ROOT / "data/raw/documents"
OUT_ROOT = ROOT / "data/pages"
DPIS     = [int(sys.argv[1])] if len(sys.argv) > 1 else [144, 72, 36]

docs = [d.strip() for d in (ROOT / "data/full_docs.txt").read_text().split("\n") if d.strip()]
print(f"문서 {len(docs)}개 | DPI {DPIS}")

total_pages = sum(pymupdf.open(PDF_DIR / d).page_count for d in docs if (PDF_DIR / d).exists())
print(f"총 {total_pages:,}페이지 x {len(DPIS)} DPI = {total_pages*len(DPIS):,}장\n")

stats, done, skipped = [], 0, 0
t0 = time.time()

for i, doc_id in enumerate(docs, 1):
    pdf = PDF_DIR / doc_id
    if not pdf.exists():
        print(f"!! 없음: {doc_id}"); continue
    doc = pymupdf.open(pdf)
    n = doc.page_count

    for dpi in DPIS:
        out_dir = OUT_ROOT / str(dpi) / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, page in enumerate(doc):
            # ★ 파일명은 1-based. PyMuPDF 인덱스는 0-based이므로 +1.
            #   오프셋 버그 1순위 지점. 파일럿에서 1-based로 확정했다.
            png = out_dir / f"{idx + 1:03d}.png"
            if png.exists():
                skipped += 1; continue
            page.get_pixmap(dpi=dpi).save(png)
            done += 1

    # ── 문서 특성 (결과 해석에 사용) ──────────────────────
    for idx, page in enumerate(doc):
        r = page.rect
        text = page.get_text().strip()
        stats.append({
            "doc_id": doc_id,
            "page": idx + 1,                     # 1-based
            "width_pt": round(r.width, 1),
            "height_pt": round(r.height, 1),
            # 가로형은 ColPali의 448x448 리사이즈에서 왜곡이 더 크다
            "orientation": "landscape" if r.width > r.height else "portrait",
            # 텍스트 레이어가 없으면 스캔본. 텍스트 파이프라인이 구조적으로 불리
            "has_text_layer": len(text) > 50,
            "char_count": len(text),
        })

    el = time.time() - t0
    eta = el / i * (len(docs) - i)
    print(f"[{i:2d}/{len(docs)}] {doc_id[:52]:<52} {n:>4}p  "
          f"생성 {done:,} 건너뜀 {skipped:,}  ETA {eta/60:.0f}분")

Path(ROOT / "data/page_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

# ── 요약 ──────────────────────────────────────────────────
print("\n" + "=" * 62)
print(f"완료 {time.time()-t0:.0f}초 | 생성 {done:,}장 | 기존 {skipped:,}장")
print(f"저장: {OUT_ROOT}")

print("\n[페이지 방향]")
for k, v in Counter(s["orientation"] for s in stats).most_common():
    print(f"  {k:<12}{v:>6} ({v/len(stats):>5.1%})")
print("  landscape는 ColPali 448x448 리사이즈에서 왜곡이 크다. 결과 해석 시 참고.")

print("\n[텍스트 레이어]")
no = [s for s in stats if not s["has_text_layer"]]
print(f"  있음 {len(stats)-len(no):>6} / 없음 {len(no):>6} ({len(no)/len(stats):.1%})")
if no:
    dd = Counter(s["doc_id"] for s in no)
    print("  텍스트 레이어 없는 페이지가 많은 문서 (스캔본 후보):")
    for d, v in dd.most_common(8):
        tot = sum(1 for s in stats if s["doc_id"] == d)
        print(f"    {d[:48]:<48}{v:>4}/{tot}")
    print("  → 텍스트 파이프라인이 구조적으로 불리하므로 리포트에 명시할 것.")

ch = sorted(s["char_count"] for s in stats)
print(f"\n[페이지당 글자수] 중앙값 {ch[len(ch)//2]:,}  최대 {ch[-1]:,}")
over = sum(1 for c in ch if c > 2000)
print(f"  2000자 초과(ColBERT 512토큰 상한 초과 예상): {over:,} ({over/len(ch):.1%})")

print("\n★ 다음: data/pages/36/ 에서 5장 열어 글자 판독 가능 여부 확인")
