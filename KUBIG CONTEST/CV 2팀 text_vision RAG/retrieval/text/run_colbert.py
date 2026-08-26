import csv
import gc
import json
import time
from pathlib import Path

import torch

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


# ============================================================
# 0. 기본 설정
# ============================================================

BASE = Path(__file__).resolve().parent.parent.parent

COLBERT_DATA_ROOT = BASE / "colbert_data"
SAMPLES_PATH = BASE / "data" / "full_samples.json"
RESULT_DIR = BASE / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DPIS = [144, 72, 36]

DOCS = [
    x.strip()
    for x in (BASE / "data" / "full_docs.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

DOC_MAXLEN = 512
QUERY_MAXLEN = 32
DOC_BATCH_SIZE = 8
TOP_K = 5

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 1. 전체 질문 불러오기
# ============================================================

with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

samples = sorted(
    samples,
    key=lambda x: x["question_id"]
)

print(f"Loaded {len(samples)} questions.")

samples_by_doc = {
    doc: sorted(
        [
            sample
            for sample in samples
            if sample["doc_id"] == doc
        ],
        key=lambda x: x["question_id"],
    )
    for doc in DOCS
}

for doc in DOCS:
    print(
        f"{doc}: {len(samples_by_doc[doc])} questions"
    )

assert sum(
    len(v) for v in samples_by_doc.values()
) == len(samples), "Some questions have an unknown doc_id."


# ============================================================
# 2. ColBERTv2 모델 로드
# ============================================================

config = ColBERTConfig(
    doc_maxlen=DOC_MAXLEN,
    query_maxlen=QUERY_MAXLEN,
)

checkpoint = Checkpoint(
    "colbert-ir/colbertv2.0",
    colbert_config=config,
    verbose=0,
)

checkpoint.eval()

print("\nColBERT model loaded.")
print(f"Scoring device: {DEVICE}")


# ============================================================
# 3. Collection / metadata 불러오기
# ============================================================

def load_collection(dpi, doc_id):

    data_dir = (
        COLBERT_DATA_ROOT
        / str(dpi)
        / doc_id
    )

    collection_path = data_dir / "collection.tsv"
    metadata_path = data_dir / "metadata.json"

    if not collection_path.exists():
        raise FileNotFoundError(collection_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    passages = []

    with open(
        collection_path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.reader(
            f,
            delimiter="\t"
        )

        for row in reader:

            if len(row) < 2:
                continue

            pid = int(row[0])
            text = row[1]

            passages.append(
                (pid, text)
            )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    metadata_by_pid = {
        int(item["pid"]): item
        for item in metadata
    }

    assert len(passages) == len(metadata), (
        f"Collection/metadata mismatch: "
        f"{dpi} DPI / {doc_id}"
    )

    return passages, metadata_by_pid


# ============================================================
# 4. Document chunk embedding 생성
# ============================================================

def encode_documents(passages):

    texts = [
        text
        for _, text in passages
    ]

    with torch.inference_mode():

        output = checkpoint.docFromText(
            texts,
            bsize=DOC_BATCH_SIZE,
            keep_dims=False,
            showprogress=True,
        )

    # 현재 설치된 ColBERT에서 bsize + keep_dims=False 사용 시
    # 첫 번째 원소가 document embedding list
    if isinstance(output, tuple):
        doc_embeddings = output[0]
    else:
        doc_embeddings = output

    # ---------------------------------------------
    # 중요:
    # test_exact_colbert.py에서 발생했던
    # device / dtype mismatch 방지
    #
    # 모든 document embedding을
    # CUDA + float32로 통일
    # ---------------------------------------------

    doc_embeddings = [
        D.to(
            device=DEVICE,
            dtype=torch.float32
        )
        for D in doc_embeddings
    ]

    return doc_embeddings


# ============================================================
# 5. 한 질문에 대한 Exact ColBERT MaxSim
# ============================================================

def retrieve_question(
    question,
    passages,
    doc_embeddings,
    metadata_by_pid,
):

    # ---------------------------------------------
    # Query embedding
    # ---------------------------------------------

    with torch.inference_mode():

        query_embedding = checkpoint.queryFromText(
            [question]
        )[0]

    # document와 동일한 device / dtype
    Q = query_embedding.to(
        device=DEVICE,
        dtype=torch.float32
    )

    page_scores = {}

    # 디버깅용:
    # 각 page에서 최고 점수를 만든 chunk도 기록
    page_best_chunk = {}


    # ---------------------------------------------
    # 모든 chunk에 대해 exact MaxSim
    # ---------------------------------------------

    with torch.inference_mode():

        for (pid, _), D in zip(
            passages,
            doc_embeddings
        ):

            # 안전장치
            if (
                D.device != Q.device
                or D.dtype != Q.dtype
            ):
                D = D.to(
                    device=Q.device,
                    dtype=Q.dtype
                )

            # Q: [query_tokens, 128]
            # D: [document_tokens, 128]
            #
            # 결과:
            # [query_tokens, document_tokens]
            similarities = torch.matmul(
                Q,
                D.transpose(0, 1)
            )

            # 각 query token마다
            # 가장 높은 document-token similarity
            maxsim = (
                similarities
                .max(dim=1)
                .values
            )

            # ColBERT late interaction score
            score = (
                maxsim
                .sum()
                .item()
            )

            meta = metadata_by_pid[pid]

            page = int(
                meta["page_number"]
            )

            chunk_id = int(
                meta["chunk_id"]
            )

            # -------------------------------------
            # chunk → page max aggregation
            # -------------------------------------

            if (
                page not in page_scores
                or score > page_scores[page]
            ):

                page_scores[page] = score

                page_best_chunk[page] = (
                    chunk_id
                )


    # ---------------------------------------------
    # 페이지 점수 내림차순
    # ---------------------------------------------

    ranked_pages = sorted(
        page_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    top = ranked_pages[:TOP_K]


    top_pages = [
        int(page)
        for page, _ in top
    ]


    scores = [
        round(float(score), 4)
        for _, score in top
    ]


    return top_pages, scores


# ============================================================
# 6. DPI별 전체 retrieval
# ============================================================

summary = {}

overall_start = time.time()


for dpi in DPIS:

    dpi_start = time.time()

    print("\n")
    print("=" * 80)
    print(f"START DPI = {dpi}")
    print("=" * 80)

    dpi_results = []


    for doc_id in DOCS:

        doc_start = time.time()

        doc_samples = samples_by_doc[
            doc_id
        ]


        print("\n" + "-" * 80)
        print(f"DPI       : {dpi}")
        print(f"Document  : {doc_id}")
        print(
            f"Questions : "
            f"{len(doc_samples)}"
        )


        # ----------------------------------------------------
        # Collection
        # ----------------------------------------------------

        passages, metadata_by_pid = (
            load_collection(
                dpi,
                doc_id
            )
        )

        print(
            f"Chunks    : {len(passages)}"
        )


        if len(passages) == 0:
            raise RuntimeError(
                f"No chunks found: "
                f"{dpi} DPI / {doc_id}"
            )


        # ----------------------------------------------------
        # 이 문서 chunk embedding은 한 번만 생성
        # ----------------------------------------------------

        doc_embeddings = (
            encode_documents(
                passages
            )
        )


        print(
            f"Document embeddings ready: "
            f"{len(doc_embeddings)} chunks"
        )


        # 확인용
        print(
            "First embedding:",
            doc_embeddings[0].shape,
            doc_embeddings[0].device,
            doc_embeddings[0].dtype,
        )


        # ----------------------------------------------------
        # 이 문서에 해당하는 질문들 검색
        # ----------------------------------------------------

        for sample in doc_samples:

            qid = int(
                sample["question_id"]
            )

            question = sample[
                "question"
            ]

            gold_pages = [
                int(x)
                for x in sample.get(
                    "evidence_pages",
                    []
                )
            ]


            top_pages, scores = (
                retrieve_question(
                    question=question,
                    passages=passages,
                    doc_embeddings=doc_embeddings,
                    metadata_by_pid=metadata_by_pid,
                )
            )


            # top-5가 정말 5개인지 확인
            if len(top_pages) != TOP_K:
                raise RuntimeError(
                    f"qid={qid}: "
                    f"only {len(top_pages)} "
                    f"pages retrieved"
                )


            hit = any(
                gold in top_pages
                for gold in gold_pages
            )


            dpi_results.append({
                "question_id": qid,
                "doc_id": doc_id,
                "question": question,
                "top_pages": top_pages,
                "scores": scores,
            })


            print(
                f"qid={qid:2d} | "
                f"gold={gold_pages} | "
                f"top5={top_pages} | "
                f"hit={hit}"
            )


        # ----------------------------------------------------
        # 문서 하나 끝
        # ----------------------------------------------------

        doc_elapsed = (
            time.time() - doc_start
        )

        print(
            f"Document finished in "
            f"{doc_elapsed:.1f} sec"
        )


        # GPU memory 정리
        del doc_embeddings
        del passages
        del metadata_by_pid

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ========================================================
    # DPI 하나 완료
    # ========================================================

    dpi_results = sorted(
        dpi_results,
        key=lambda x: x["question_id"]
    )


    # 전체 문항 수와 결과 수가 일치하는지 확인
    assert len(dpi_results) == len(samples), (
        f"{dpi} DPI: "
        f"expected {len(samples)} results, "
        f"got {len(dpi_results)}"
    )


    qids = [
        x["question_id"]
        for x in dpi_results
    ]

    assert len(set(qids)) == len(qids), (
        f"{dpi} DPI: duplicate qids"
    )


    # --------------------------------------------------------
    # JSON 저장
    # --------------------------------------------------------

    output_path = (
        RESULT_DIR
        / f"retrieval_text_{dpi}.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dpi_results,
            f,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------------
    # Recall@5
    # --------------------------------------------------------

    result_by_qid = {
        item["question_id"]: item
        for item in dpi_results
    }


    hits = 0


    for sample in samples:

        qid = int(
            sample["question_id"]
        )

        gold_pages = [
            int(x)
            for x in sample.get(
                "evidence_pages",
                []
            )
        ]

        top_pages = (
            result_by_qid[qid]
            ["top_pages"]
        )


        if any(
            gold in top_pages
            for gold in gold_pages
        ):
            hits += 1


    recall_at_5 = (
        hits / len(samples)
    )


    dpi_elapsed = (
        time.time() - dpi_start
    )


    summary[str(dpi)] = {
        "questions": len(samples),
        "hits_at_5": hits,
        "recall_at_5": recall_at_5,
        "elapsed_seconds": dpi_elapsed,
    }


    print("\n" + "=" * 80)

    print(
        f"DPI {dpi} FINISHED"
    )

    print(
        f"Recall@5 = "
        f"{hits}/{len(samples)} "
        f"= {recall_at_5:.4f}"
    )

    print(
        f"Elapsed = "
        f"{dpi_elapsed:.1f} sec"
    )

    print(
        f"Saved to: {output_path}"
    )

    print("=" * 80)


# ============================================================
# 7. Summary 저장
# ============================================================

overall_elapsed = (
    time.time() - overall_start
)

summary["settings"] = {
    "model": "colbert-ir/colbertv2.0",
    "retrieval": "exact_maxsim",
    "chunk_size": 480,
    "chunk_overlap": 50,
    "doc_maxlen": DOC_MAXLEN,
    "query_maxlen": QUERY_MAXLEN,
    "top_k": TOP_K,
}

summary["overall_elapsed_seconds"] = (
    overall_elapsed
)


summary_path = (
    RESULT_DIR
    / "retrieval_text_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n")
print("=" * 80)
print("ALL TEXT RETRIEVAL FINISHED")
print("=" * 80)

for dpi in DPIS:

    s = summary[str(dpi)]

    print(
        f"{dpi} DPI: "
        f"Recall@5 = "
        f"{s['hits_at_5']}/"
        f"{s['questions']} "
        f"= {s['recall_at_5']:.4f}"
    )

print(
    f"\nTotal elapsed: "
    f"{overall_elapsed:.1f} sec"
)

print(
    f"Summary: {summary_path}"
)