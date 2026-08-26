"""
[후행 1] 비전 파이프라인 답변 생성 — Qwen2.5-VL에 top-5 페이지 이미지 입력

M3DocRAG 3단계(question answering)에 해당.
검색된 페이지 '이미지'를 넣는다. 임베딩은 쓰지 않는다.

실행: python 03_generate_vision.py 144
      python 03_generate_vision.py 72
      python 03_generate_vision.py 36

출력: data/answers_vision_{dpi}.json
"""
import json, sys, gc, time, inspect
from pathlib import Path

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# transformers 4.x / 5.x 양쪽 호환
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import AutoModelForImageTextToText as VLModel

DPI      = int(sys.argv[1]) if len(sys.argv) > 1 else 144
ROOT     = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data/full_samples.json"
RETR     = ROOT / f"retrieval_vision_{DPI}.json"
PAGES    = ROOT / f"data/pages/{DPI}"
OUT      = ROOT / f"data/answers_vision_{DPI}.json"
MODEL    = "Qwen/Qwen2.5-VL-7B-Instruct"

# ★ 해상도 상한. 144 DPI A4(1190x1684 ≈ 2.0M px)가 축소되지 않도록 넉넉히 잡는다.
#   여기를 낮추면 144와 72가 같은 크기로 눌려서 DPI 실험이 무의미해진다.
#   OOM이 나면 낮추되, 세 DPI 전부 같은 값으로 다시 돌려야 공정하다.
MAX_PIXELS = 2560 * 28 * 28
MIN_PIXELS = 56 * 28 * 28      # 36 DPI가 강제 확대되지 않도록 낮게 유지

# ── 경로 검증 (여기서 걸러야 GPU 시간 안 버린다) ──────────
for p in [SAMPLES, RETR, PAGES]:
    if not p.exists():
        sys.exit(f"경로 없음: {p}")

samples = {s["question_id"]: s for s in json.loads(SAMPLES.read_text())}
retr    = json.loads(RETR.read_text())
print(f"DPI {DPI} | 문항 {len(retr)}건")

missing = [str(PAGES / r["doc_id"] / f"{pg:03d}.png")
           for r in retr for pg in r["top_pages"]
           if not (PAGES / r["doc_id"] / f"{pg:03d}.png").exists()]
if missing:
    print(f"!! 이미지 누락 {len(missing)}개")
    for m in missing[:5]: print("   ", m)
    sys.exit(1)
print("이미지 전량 확인 완료")

# ── 모델 로드 ─────────────────────────────────────────────
# transformers 5.x는 torch_dtype → dtype 으로 인자명이 바뀌었다.
# 잘못 넘기면 무시되고 fp32(28GB)로 올라가므로 시그니처를 보고 결정한다.
kw = "dtype" if "dtype" in inspect.signature(VLModel.from_pretrained).parameters \
     else "torch_dtype"
print(f"모델 로딩... ({VLModel.__name__}, {kw}=bfloat16)")
model = VLModel.from_pretrained(MODEL, device_map="auto",
                                attn_implementation="sdpa", **{kw: torch.bfloat16})
model.eval()
print("실제 dtype:", next(model.parameters()).dtype)

processor = AutoProcessor.from_pretrained(MODEL, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

# ★ 프롬프트는 텍스트 파이프라인과 한 글자도 다르면 안 된다.
#   여기가 두 경로의 유일한 통제 지점.
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
               for pg in r["top_pages"]]                    # 검색된 5장
    content.append({"type": "text", "text": PROMPT.format(q=s["question"])})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids,
                       padding=True, return_tensors="pt").to(model.device)

    n_in  = inputs.input_ids.shape[1]     # 입력 토큰 수 (경로 간 비대칭 확인용)
    grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    pred = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()

    results.append({
        "question_id": qid, "doc_id": s["doc_id"], "question": s["question"],
        "gold": s["answer"], "answer_format": s["answer_format"],
        "evidence_pages": s["evidence_pages"], "top_pages": r["top_pages"],
        "retrieved_hit": bool(set(s["evidence_pages"]) & set(r["top_pages"])),
        "pred": pred, "n_input_tokens": n_in, "image_grid_thw": grids,
    })
    print(f"[{i+1:2d}/{len(retr)}] q{qid:<3} tok={n_in:<6} {pred[:50]}")

    del inputs, out
    gc.collect(); torch.cuda.empty_cache()

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\n저장: {OUT}  ({time.time()-t0:.0f}초)")

# ── DPI가 실제로 반영됐는지 확인 ──────────────────────────
toks = [r["n_input_tokens"] for r in results]
print(f"입력 토큰  평균 {sum(toks)//len(toks)}  최소 {min(toks)}  최대 {max(toks)}")
print("★ 세 DPI의 평균 토큰이 서로 달라야 정상. 같으면 리사이즈로 뭉개진 것.")
