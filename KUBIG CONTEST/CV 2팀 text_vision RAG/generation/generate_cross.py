"""
[추가 실험] 교차 조건 T→V
  텍스트 리트리버(ColBERT)가 뽑은 top-5 페이지를,
  OCR 텍스트가 아니라 '페이지 이미지'로 생성기에 넣는다.

목적
  기존 두 조건은 검색 방식과 입력 형태가 함께 바뀌어 있다.
      T→T = ColBERT 검색 + OCR 텍스트 입력
      V→V = ColPali 검색 + 페이지 이미지 입력
  T→V를 추가하면 T→T와는 '입력 형태'만, V→V와는 '검색 방식'만 다르므로
  36 DPI 붕괴의 원인이 검색인지 입력인지 분리할 수 있다.

  T→V vs T→T 비교 → 같은 페이지를 줬을 때 이미지가 나은가 OCR이 나은가
  T→V vs V→V 비교 → 같은 이미지를 줬을 때 검색 방식 차이가 얼마인가

실행: python 03_generate_textv.py 144 / 72 / 36
출력: data/answers_text(v)_{dpi}.json
"""
import json, sys, gc, time, inspect
from pathlib import Path

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import AutoModelForImageTextToText as VLModel

DPI     = int(sys.argv[1]) if len(sys.argv) > 1 else 144
ROOT    = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data/full_samples.json"
RETR    = ROOT / f"retrieval_text_{DPI}.json"        # ★ 텍스트 리트리버 결과
PAGES   = ROOT / f"data/pages/{DPI}"                 # ★ 그러나 입력은 이미지
OUT     = ROOT / f"data/answers_text(v)_{DPI}.json"
MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"

# 03_generate_vision.py 와 동일한 값을 써야 V→V 와 비교 가능하다
MAX_PIXELS = 2560 * 28 * 28
MIN_PIXELS = 56 * 28 * 28

for p in (SAMPLES, RETR, PAGES):
    if not p.exists():
        sys.exit(f"경로 없음: {p}")

samples = {s["question_id"]: s for s in json.loads(SAMPLES.read_text())}
retr    = json.loads(RETR.read_text())
print(f"[T→V] DPI {DPI} | 문항 {len(retr)}건 | 검색=ColBERT, 입력=페이지 이미지")

missing = [str(PAGES / r["doc_id"] / f"{pg:03d}.png")
           for r in retr for pg in r["top_pages"]
           if not (PAGES / r["doc_id"] / f"{pg:03d}.png").exists()]
if missing:
    print(f"!! 이미지 누락 {len(missing)}개")
    for m in missing[:5]: print("   ", m)
    sys.exit(1)
print("이미지 전량 확인 완료")

kw = "dtype" if "dtype" in inspect.signature(VLModel.from_pretrained).parameters else "torch_dtype"
print(f"모델 로딩... ({VLModel.__name__}, {kw}=bfloat16)")
model = VLModel.from_pretrained(MODEL, device_map="auto",
                                attn_implementation="sdpa", **{kw: torch.bfloat16})
model.eval()
print("실제 dtype:", next(model.parameters()).dtype)
processor = AutoProcessor.from_pretrained(MODEL, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

# ★ 다른 두 조건과 한 글자도 달라선 안 된다
PROMPT = ("Answer the question based on the given document pages.\n"
          "Output only the final answer, with no explanation.\n\n"
          "Question: {q}\nAnswer:")

results = []
t0 = time.time()
for i, r in enumerate(retr):
    qid = r["question_id"]
    s   = samples[qid]

    content = [{"type": "image",
                "image": f"file://{PAGES / r['doc_id'] / f'{pg:03d}.png'}"}
               for pg in r["top_pages"]]
    content.append({"type": "text", "text": PROMPT.format(q=s["question"])})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids,
                       padding=True, return_tensors="pt").to(model.device)

    n_in  = inputs.input_ids.shape[1]
    grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []

    with torch.inference_mode():
        outp = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    pred = processor.batch_decode(outp[:, n_in:], skip_special_tokens=True)[0].strip()

    results.append({
        "question_id": qid, "doc_id": s["doc_id"], "question": s["question"],
        "gold": s["answer"], "answer_format": s["answer_format"],
        "evidence_pages": s["evidence_pages"], "top_pages": r["top_pages"],
        # retrieved_hit은 ColBERT 검색 기준이므로 T→T와 동일한 값이 된다
        "retrieved_hit": bool(set(s["evidence_pages"]) & set(r["top_pages"])),
        "pred": pred, "n_input_tokens": n_in, "image_grid_thw": grids,
    })
    if (i + 1) % 10 == 0 or i == 0:
        el = time.time() - t0
        print(f"[{i+1:3d}/{len(retr)}] q{qid:<4} tok={n_in:<6} "
              f"ETA {el/(i+1)*(len(retr)-i-1)/60:.0f}분  {pred[:40]}")

    del inputs, outp
    gc.collect(); torch.cuda.empty_cache()

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
toks = [r["n_input_tokens"] for r in results]
hit  = sum(r["retrieved_hit"] for r in results)
print(f"\n저장: {OUT}  ({time.time()-t0:.0f}초)")
print(f"Recall@5 {hit}/{len(results)} = {hit/len(results):.1%}  (T→T와 동일해야 정상)")
print(f"입력 토큰 평균 {sum(toks)//len(toks):,}  (V→V와 비슷해야 정상)")
