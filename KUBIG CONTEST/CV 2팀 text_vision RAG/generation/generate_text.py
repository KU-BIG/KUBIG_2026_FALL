"""
[후행 1-b] 텍스트 파이프라인 답변 생성 — Qwen2.5-VL에 top-5 페이지 OCR 텍스트 입력

★ 생성기·프롬프트·top-k를 비전과 완전히 동일하게 유지한다.
  다른 것은 오직 "페이지 내용이 이미지로 들어가느냐 텍스트로 들어가느냐"뿐.
  (M3DocRAG은 텍스트 경로에 Llama를 따로 썼으나, 그러면 검색기와 생성기가
   동시에 바뀌어 변수 분리가 안 된다. 여기서는 생성기를 고정한다.)

실행: python 03_generate_text.py 144 / 72 / 36
출력: data/answers_text_{dpi}.json
"""
import json, sys, gc, time, inspect
from pathlib import Path

import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import AutoModelForImageTextToText as VLModel

DPI     = int(sys.argv[1]) if len(sys.argv) > 1 else 144
ROOT    = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data/full_samples.json"
RETR    = ROOT / f"retrieval_text_{DPI}.json"
OCR     = ROOT / f"ocr_text_{DPI}.json"
OUT     = ROOT / f"data/answers_text_{DPI}.json"
MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"

# 페이지당 OCR 텍스트 상한(문자). 5장 합쳐도 컨텍스트에 들어가도록.
# 잘림이 발생하면 로그의 n_truncated로 확인된다.
MAX_CHARS_PER_PAGE = 6000

for p in [SAMPLES, RETR, OCR]:
    if not p.exists():
        sys.exit(f"경로 없음: {p}")

samples = {s["question_id"]: s for s in json.loads(SAMPLES.read_text())}
retr    = json.loads(RETR.read_text())
# ocr_text_{dpi}.json 은 [{doc_id, page_number, text}, ...] 형태
ocr     = {(r["doc_id"], r["page_number"]): r["text"]
           for r in json.loads(OCR.read_text())}
print(f"DPI {DPI} | 문항 {len(retr)}건 | OCR 페이지 {len(ocr)}개")

missing = [(r["doc_id"], pg) for r in retr for pg in r["top_pages"]
           if (r["doc_id"], pg) not in ocr]
if missing:
    print(f"!! OCR 누락 {len(missing)}개: {missing[:5]}")
    sys.exit(1)
print("OCR 전량 확인 완료")

print("모델 로딩...")
kw = "dtype" if "dtype" in inspect.signature(VLModel.from_pretrained).parameters else "torch_dtype"
model = VLModel.from_pretrained(MODEL, device_map="auto",
                                attn_implementation="sdpa", **{kw: torch.bfloat16})
model.eval()
print("실제 dtype:", next(model.parameters()).dtype)
processor = AutoProcessor.from_pretrained(MODEL)

# ★ 03_generate_vision.py 와 한 글자도 다르지 않아야 한다.
PROMPT = ("Answer the question based on the given document pages.\n"
          "Output only the final answer, with no explanation.\n\n"
          "Question: {q}\nAnswer:")

results = []
t0 = time.time()
n_trunc = 0
for i, r in enumerate(retr):
    qid = r["question_id"]
    s   = samples[qid]

    # 페이지 경계를 명시한다. 이미지는 5장이 자연히 구분되지만
    # 텍스트는 이어붙이면 경계가 사라져 조건이 비대칭해진다.
    parts, tr = [], 0
    for pg in r["top_pages"]:
        t = ocr[(r["doc_id"], pg)]
        if len(t) > MAX_CHARS_PER_PAGE:
            t = t[:MAX_CHARS_PER_PAGE]; tr += 1
        parts.append(f"--- Page {pg} ---\n{t}")
    n_trunc += tr
    ctx = "\n\n".join(parts)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": ctx + "\n\n" + PROMPT.format(q=s["question"])}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    n_in = inputs.input_ids.shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    pred = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()

    results.append({
        "question_id": qid, "doc_id": s["doc_id"], "question": s["question"],
        "gold": s["answer"], "answer_format": s["answer_format"],
        "evidence_pages": s["evidence_pages"], "top_pages": r["top_pages"],
        "retrieved_hit": bool(set(s["evidence_pages"]) & set(r["top_pages"])),
        "pred": pred, "n_input_tokens": n_in,
        "n_ctx_chars": len(ctx), "n_pages_truncated": tr,
    })
    print(f"[{i+1:2d}/{len(retr)}] q{qid:<3} tok={n_in:<6} {pred[:50]}")
    del inputs, out
    gc.collect(); torch.cuda.empty_cache()

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
toks = [r["n_input_tokens"] for r in results]
chars = [r["n_ctx_chars"] for r in results]
print(f"\n저장: {OUT}  ({time.time()-t0:.0f}초)")
print(f"입력 토큰  평균 {sum(toks)//len(toks)}  최소 {min(toks)}  최대 {max(toks)}")
print(f"컨텍스트   평균 {sum(chars)//len(chars):,}자   잘린 페이지 {n_trunc}개")
print("★ 비전과 토큰 수를 비교해 컨텍스트 비대칭 여부를 리포트에 명시할 것.")
