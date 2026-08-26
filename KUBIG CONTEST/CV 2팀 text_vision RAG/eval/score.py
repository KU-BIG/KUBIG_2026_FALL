"""
[후행 2] 채점 + 집계

answer_format(Str/Int/Float/List)별 규칙 채점.
규칙 기반이라 애매한 건 review 목록으로 빼서 눈으로 확인한다.

실행: python 04_score.py vision
      python 04_score.py text
출력: data/scored_{pipe}_{dpi}.json, data/review_{pipe}.json, 화면 표
"""
import json, re, sys, ast
from pathlib import Path
from collections import defaultdict

PIPE = sys.argv[1] if len(sys.argv) > 1 else "vision"
ROOT = Path(__file__).resolve().parent.parent
DPIS = [144, 72, 36]

# ── 집계 제외 문항 ────────────────────────────────────────
# q13: 검색 3개 다 실패인데 답변은 3개 다 정답. 문서를 안 보고 상식으로 답한 것
#      ("고양이 몇 마리?" → 0). 검색 성능과 무관해 지표를 왜곡한다.
# q26: AMCOR 질문이 BESTBUY 문서에 붙은 데이터 오류. 정답 페이지가 존재할 수 없음.
EXCLUDE = {13, 26}


def norm(s):
    """소문자화 + 구두점 제거. 'December 31,2018' vs 'December 31, 2018' 흡수."""
    s = str(s).lower().strip()
    s = re.sub(r"[\$,%()\[\]'\"\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def nums(s):
    return [float(x.replace(",", "")) for x in
            re.findall(r"-?\d[\d,]*\.?\d*", str(s).replace("%", ""))]


def as_list(s):
    if isinstance(s, list): return s
    try:
        v = ast.literal_eval(str(s))
        return v if isinstance(v, list) else [s]
    except Exception:
        return [x for x in re.split(r"[,\n;]", str(s)) if x.strip()]


def judge(pred, gold, fmt):
    """반환: (정답여부, 확신도). low는 눈으로 재확인 대상."""
    p, g = norm(pred), norm(gold)
    if not p:
        return False, "high"

    if fmt in ("Int", "Float"):
        pn, gn = nums(pred), nums(gold)
        if not gn: return p == g, "low"
        if not pn: return False, "high"
        # 상대오차 1% 허용 — 반올림 자릿수 차이(1.08% vs 1.1%) 흡수
        for x in pn:
            if abs(x - gn[0]) <= max(abs(gn[0]) * 0.01, 1e-6):
                return True, "high"
        return False, "high" if len(pn) == 1 else "low"

    if fmt == "List":
        # 모델이 대괄호 없이 서술형으로 답해도 항목이 다 있으면 정답 처리
        gl = [norm(x) for x in as_list(gold) if norm(x)]
        if not gl: return False, "low"
        hit = sum(1 for x in gl if x in p)
        if hit == len(gl): return True, "high"
        return False, "low" if hit else "high"

    if g and g in p: return True, "high"
    if p and p in g and len(p) > 3: return True, "low"
    return False, "low" if len(g.split()) > 3 else "high"


rows, review = {}, []
for dpi in DPIS:
    f = ROOT / f"data/answers_{PIPE}_{dpi}.json"
    if not f.exists():
        print(f"건너뜀 (없음): {f}"); continue
    recs = json.loads(f.read_text())
    for r in recs:
        ok, conf = judge(r["pred"], r["gold"], r["answer_format"])
        r["correct"], r["confidence"] = ok, conf
        r["excluded"] = r["question_id"] in EXCLUDE
        if conf == "low" and not r["excluded"]:
            review.append({"dpi": dpi, "qid": r["question_id"], "fmt": r["answer_format"],
                           "판정": ok, "gold": str(r["gold"])[:70], "pred": str(r["pred"])[:70]})
    rows[dpi] = recs
    (ROOT / f"data/scored_{PIPE}_{dpi}.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=2))

if not rows:
    sys.exit("채점할 파일이 없습니다.")

n_all = len(next(iter(rows.values())))
print(f"\n{PIPE} 파이프라인 | 전체 {n_all}문항 → 제외 {len(EXCLUDE)}건(q{', q'.join(map(str,sorted(EXCLUDE)))}) → 집계 {n_all-len(EXCLUDE)}문항")

print("\n" + "=" * 74)
print(f"{'DPI':>5} {'평균토큰':>9} {'Recall@5':>10} {'Recall@1':>10} {'답변정확도':>11} {'검색성공시':>13}")
print("=" * 74)
for dpi, allrecs in rows.items():
    recs = [r for r in allrecs if not r["excluded"]]
    n = len(recs)
    h = [r for r in recs if r["retrieved_hit"]]
    r1 = sum(1 for r in recs if r["top_pages"][0] in r["evidence_pages"]) / n
    tok = sum(r["n_input_tokens"] for r in recs) // n
    cacc = sum(r["correct"] for r in h) / len(h) if h else 0
    print(f"{dpi:>5} {tok:>9,} {sum(r['retrieved_hit'] for r in recs)/n:>9.1%} "
          f"{r1:>10.1%} {sum(r['correct'] for r in recs)/n:>11.1%} "
          f"{cacc:>10.1%} ({sum(r['correct'] for r in h)}/{len(h)})")

def breakdown(title, keyfn):
    print(f"\n[{title}]")
    keys = sorted({keyfn(r) for r in next(iter(rows.values())) if not r["excluded"]})
    print(f"  {'':26}" + "".join(f"{d:>10}" for d in rows))
    for k in keys:
        cells = []
        for dpi, allrecs in rows.items():
            sub = [r for r in allrecs if not r["excluded"] and keyfn(r) == k]
            cells.append(f"{sum(r['correct'] for r in sub)}/{len(sub)}")
        print(f"  {str(k)[:26]:26}" + "".join(f"{c:>10}" for c in cells))

gt = {s["question_id"]: s for s in json.loads((ROOT/"data/pilot_samples.json").read_text())}
breakdown("문서별 답변 정확도", lambda r: r["doc_id"][:22])
breakdown("근거 유형별 답변 정확도",
          lambda r: (gt[r["question_id"]]["evidence_sources"] or ["(없음)"])[0])
breakdown("answer_format별 답변 정확도", lambda r: r["answer_format"])

print("\n[문항별 상세]  ● 성공 / · 실패 / × 제외")
print(f"  {'qid':>4} {'검색 144/72/36':>16} {'답변 144/72/36':>16}  gold")
for qid in sorted(r["question_id"] for r in next(iter(rows.values()))):
    rr = [next(x for x in rows[d] if x["question_id"] == qid) for d in rows]
    if rr[0]["excluded"]:
        print(f"  {qid:>4} {'× 제외':>16} {'':>16}  {str(rr[0]['gold'])[:40]}")
        continue
    R = "  ".join("●" if r["retrieved_hit"] else "·" for r in rr)
    A = "  ".join("●" if r["correct"] else "·" for r in rr)
    print(f"  {qid:>4} {R:>16} {A:>16}  {str(rr[0]['gold'])[:40]}")

print(f"\n[눈으로 재확인 필요 {len(review)}건]")
for x in review:
    print(f"  {'O' if x['판정'] else 'X'} dpi{x['dpi']:<4} q{x['qid']:<3} [{x['fmt']}]")
    print(f"      gold: {x['gold']}")
    print(f"      pred: {x['pred']}")
(ROOT / f"data/review_{PIPE}.json").write_text(json.dumps(review, ensure_ascii=False, indent=2))
print(f"\n저장: data/scored_{PIPE}_*.json, data/review_{PIPE}.json")
