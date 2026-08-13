# 작업 히스토리 & 의사결정 로그

이 파일은 시간 순으로 기록한다. 각 항목은 (1) 무엇을 했는지 (2) 어떤 판단을 왜 내렸는지 (3) 어떤 오류/막힘이 있었고 어떻게 풀었는지를 남긴다. 계획서(`docs/PLAN.md`) 자체를 바꾸는 결정은 여기 근거를 적고 PLAN.md도 함께 갱신한다.

세션이 끊겨도 이 파일 + `hyunju` 브랜치의 커밋 로그를 보면 진행 상황을 그대로 따라잡을 수 있어야 한다.

---

## 2026-08-13 — Kickoff: 환경 점검 및 저장소 세팅

**한 일**
- 실행 환경 점검: GPU 1×A100-SXM4-40GB (여유), CPU 30 core, RAM 216GB, 디스크 여유 439GB, 인터넷 접속(github.com / ftp.ncbi.nlm.nih.gov / huggingface.co 모두 200 OK) 정상.
- `git@github.com:hyunju4d/cv-session-1.git` 클론 확인. `hyunju` 브랜치 생성 (base: `main`, 최초 커밋 `91022e5`).
- 프로젝트 디렉토리 구조 확정 (아래 "디렉토리 구조" 절 참고).
- 파이썬 환경 확인: torch 2.8.0+cu128 설치되어 있으나 `scanpy`, `anndata`, `harmonypy`, `scib`가 없음 → 다음 단계에서 설치 필요.

**판단 및 근거**
- **git 인증 방식**: `git@github.com:` 형태 SSH URL을 시도했더니 로컬에 SSH 키가 없어 `Permission denied (publickey)`. 다만 `git config --global -l`에 이미 `url.https://github.com/.insteadof=git@github.com:` 재작성 규칙과 `gh auth git-credential` 헬퍼가 설정되어 있어, git 명령에서 `git@github.com:...` 형태를 그대로 써도 실제로는 HTTPS + `gh` 토큰(스코프 `repo` 포함)으로 인증됨을 `git ls-remote`로 확인. → SSH 키를 새로 만들 필요 없이 기존 `gh` 인증을 그대로 사용하기로 결정.
- **커밋 대상 브랜치**: 사용자가 명시한 대로 `main`이 아닌 `hyunju` 브랜치에서만 커밋. `main`은 건드리지 않는다.
- **커밋 및 푸시 정책**: 세션이 끊겨도 확인 가능해야 한다는 요구사항이 있으므로, 로컬 커밋만으로는 부족하다고 판단 — 매 마일스톤마다 `hyunju` 브랜치에 커밋 후 즉시 origin으로 push한다. (push는 사용자가 이미 이 repo/branch 사용을 명시적으로 요청한 범위 내 행동으로 판단.)
- **디렉토리 구조**: 계획서의 Phase 구조(0~3)를 코드 레이아웃에 그대로 반영하기 위해 `src/experiments/phase{0,1,2,3}_*.py` 형태로 실험 스크립트를 분리하고, 재사용되는 로직(인코더, 지표, 데이터 로딩)은 별도 모듈(`src/encoders`, `src/metrics`, `src/data`)로 뺀다. 실험 스크립트는 "모듈을 불러다 실행하고 결과 표를 저장"하는 역할만 하도록 해서, 나중에 어떤 실험이 계획서 몇 번 항목에 대응하는지 추적하기 쉽게 한다 (→ `docs/CODE_MAP.md`).
- **대용량 파일 정책**: 원본 GSE194122 데이터(수 GB~수십 GB 예상), MatchCLOT 등 외부 레포, 학습 체크포인트, `.h5ad` 등은 git에 커밋하지 않는다(`.gitignore` 처리). 대신 다운로드/재현 스크립트와 체크섬을 커밋해서 "코드로 재현 가능"하게 유지. 결과 요약 표(csv/json/md)와 작은 figure는 git에 포함.

**막힘/이슈**
- (없음 — 환경 점검 단계는 순조롭게 통과)

**다음 단계**
- Phase 0: gap 지표 모듈 구현 (합성 데이터로 단위 테스트) — 실제 데이터 없이도 먼저 검증 가능.
- Phase 0: MatchCLOT 레포 확보, pretrained weight 존재 여부 확인.
- Phase 1: GSE194122 실제 데이터 다운로드 경로 확인 (OpenProblems NeurIPS2021 competition 배포본 우선 탐색, 없으면 GEO supplementary 직접 사용).

---

## 2026-08-13 (계속) — Phase 0 지표 모듈 구현 + MatchCLOT 조사 + 데이터 확보

**한 일**
- `pip install`로 scanpy/anndata/harmonypy/POT/statsmodels/pytest 등 설치. 이 과정에서 numpy 1.26→2.5, pandas 2.1→3.0, scipy 1.11→1.18, scikit-learn 1.3→1.9로 의존성 해결에 따라 자동 업그레이드됨. torch/CUDA는 그대로 유지되고 정상 동작 확인(`torch.cuda.is_available()==True`, A100 인식). `requirements.txt`에 실제 설치된 정확한 버전을 고정해 기록.
- `src/metrics/gap_metrics.py` 구현: `delta_gap`(주지표, unit-normalized centroid distance), `alignment`/`uniformity`(Wang & Isola 분해), `linear_separability`(logistic regression CV accuracy), `topk_retrieval_accuracy`. `gap_report()`로 4개 지표를 한번에 계산하는 진입점 추가.
- `tests/test_gap_metrics.py`: 순수 합성 데이터로 15개 단위테스트 작성 및 전체 통과 확인 (`python -m pytest tests/` → 15 passed). 검증한 성질: gap 없음→delta_gap≈0, shift 커질수록 delta_gap 단조증가, 분포 겹칠 때 linear separability≈chance, 완전분리시≈1.0, paired 노이즈 커질수록 alignment 악화, retrieval accuracy는 노이즈 커질수록 chance(k/n) 근처로 하락 등.
- `external/MatchCLOT`에 원 저자 레포(AI4SCR/MatchCLOT) clone. 아키텍처/전처리/학습 코드 확인.
- GSE194122 원본 데이터(OpenProblems NeurIPS2021 BMMC, GEO supplementary) 다운로드 완료: `data/raw/cite_BMMC_processed.h5ad.gz`(587MB, GEX+ADT), `data/raw/multiome_BMMC_processed.h5ad.gz`(2.7GB, GEX+ATAC). gzip 무결성 확인 후 압축 해제 진행 중.

**판단 및 근거 (중요)**

1. **Pretrained MatchCLOT 가중치는 사용 불가로 판단.** `docs/source/quickstart.md`에 명시된 사전학습 GEX2ATAC 가중치 다운로드 링크(`ibm.box.com/s/3qhv2usv4n3aif2v3hml5eu5mmko5jbi`)에 접속 시 Box 앱 쉘만 반환되고 404 처리됨 (링크 만료 또는 접근 제한으로 추정). GEX2ADT용 사전학습 가중치는 애초에 공개 링크 자체가 문서에 없음. → **계획서 1단계의 "인코더 3종(pretrained/from-scratch/linear CCA+OT)" 중 pretrained 조건은 이번 실행에서 제외**하고, from-scratch 재학습 인코더를 주 baseline으로, linear CCA+OT를 경량 robustness baseline으로 사용하기로 결정. (원래 계획에서도 "gap 순위가 인코더 선택에 강건한지" 확인이 목적이었지 pretrained 자체가 목적이 아니었으므로, 목적 달성에는 지장 없음. 다만 "원 논문이 재현한 SOTA 수치와 직접 비교"는 못 하게 된 한계로 기록.)
2. **MatchCLOT 원본 학습 코드(`train.py`/`run.py`)를 그대로 쓰지 않기로 결정.** 원본은 `catalyst==22.4`, `torch==1.13.1`, `anndata==0.8.0`, `pandas==1.5.1` 등 오래된 버전에 고정되어 있어, 현재 환경(torch 2.8+cu128, anndata 0.13, pandas 3.0)과 충돌 가능성이 크고, 별도의 legacy 가상환경을 만드는 것은 이후 수십 개 실험 조건(입력 차원을 바꿔가며 재학습)을 유연하게 반복하기 어렵게 만듦. → **`matchclot/embedding/models.py`(순수 torch, 의존성 없음: `Encoder`, `Modality_CLIP`, `symmetric_npair_loss`)와 `matchclot/preprocessing/preprocess.py`(LSI, TF-IDF, harmony 래퍼)의 핵심 로직만 그대로 가져다 쓰고, 학습 루프는 catalyst 없이 순수 PyTorch로 새로 작성**한다. 이렇게 하면 아키텍처/손실함수(=MatchCLOT의 실제 기여분)는 그대로 보존하면서, 실험 조건별 입력 차원 변경·시드 반복 등을 코드로 유연하게 제어할 수 있다.
3. **데이터 소스: OpenProblems competition의 "phase2" 분할 파일 대신 GEO의 원본 processed h5ad를 사용.** MatchCLOT의 `run.py`는 competition이 자체적으로 나눈 `train_mod*.h5ad`/`test_mod*.h5ad` 분할을 기대하지만, 이 분할은 우리가 검증할 수 없는 방식으로 만들어졌고 pretrained 모델이 그 test set으로 이미 검증됐을 가능성이 있어 우리 목적(우리가 직접 정의한 held-out split으로 data leakage 방지)에는 안 맞는다. GEO의 `*_BMMC_processed.h5ad`는 GEX+ADT 또는 GEX+ATAC이 하나의 AnnData에 함께 들어있는 원본 통합본이므로, **우리가 직접 train/test split, batch 구성, cell-type subsetting을 모두 통제해서 재현성 있게 만들 수 있다는 장점**이 있어 이쪽을 채택.
4. **harmony 패키지 선택.** MatchCLOT 원본은 GPU 가속 `harmony-pytorch`(`from harmony import harmonize`)를 쓰지만, 우리 Phase 1-2 배치효과 분석(계획서 2번 항목)은 MatchCLOT 파이프라인 재현이 목적이 아니라 독립적인 confound 분석이 목적이므로, 생태계에서 더 널리 검증된 `harmonypy`(R Harmony의 python 포트, scanpy 표준)를 사용하기로 결정. MatchCLOT 아키텍처를 그대로 재현하는 encoder 학습 코드 안에서는 원본과 동일하게 harmony-pytorch도 함께 설치해 fidelity를 유지.

**막힘/이슈**
- IBM Box pretrained weight 링크 접근 불가 (위 판단 1 참고). 대안 조치 완료, 더 이상 막힘 아님.
- (해결됨) git 커밋 시 `user.name` 미설정 오류 → 전역 git config를 바꾸지 않고 커밋마다 `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL`/`GIT_COMMITTER_NAME`/`GIT_COMMITTER_EMAIL` 환경변수를 지정하는 방식으로 우회 (git 설정을 영구적으로 바꾸지 않기 위한 선택).

**다음 단계**
- h5ad 압축 해제 완료 확인 후 구조 파악 (obs/var/layers, batch/cell type 라벨 컬럼, GEX/ADT/ATAC 분리 방법).
- `src/data/` 모듈: 로딩 + train/test split + batch-aware split 구현.
- `src/encoders/`: linear CCA+OT baseline, MatchCLOT 아키텍처 기반 from-scratch 학습 래퍼 구현.
- Phase 1 baseline 실행.

---

## 2026-08-13 (계속 2) — 실제 데이터 구조 확인, Phase 1 데이터/인코더 모듈 구현, OOM 버그 수정

**한 일**
- h5ad 실제 구조 확인: cite(90,261 cells, GEX 13,953 + ADT 134, batch 12개=4 site×donor, cell_type 45종, obs에 이미 `is_train` 컬럼 존재하나 우리는 별도 split 사용), multiome(69,249 cells, GEX 13,431 + ATAC 116,490, batch 13개, cell_type 22종). 계획서에 적힌 세포수·ADT 134차원과 정확히 일치함을 확인.
- `src/data/loading.py`, `src/data/preprocessing.py`(GEX HVG+정규화, ADT CLR, ATAC LSI), `src/encoders/linear_baseline.py`(CCA), `src/encoders/matchclot_arch.py`(MatchCLOT 아키텍처 vendored + 순수 PyTorch 학습 루프) 구현 및 커밋.
- `src/experiments/phase1_baseline.py`으로 linear CCA 인코더 첫 실행 → **버그 발견 및 수정** (아래).
- `src/metrics/variance_partitioning.py`(batch-effect confound 분석용 group R² + permutation p-value) 구현, 단위테스트 5개 통과.

**판단 및 근거 (중요) — 실행 중 발견한 버그**

1. **CCA가 2000차원 HVG 행렬에서 비정상적으로 느림.** `sklearn.cross_decomposition.CCA`(NIPALS 기반)를 72,208×2000 GEX 행렬에 직접 fit 시도했더니 2~3분 만에 CPU-time 50분+ 소모하며 끝나지 않음. → **GEX를 PCA로 100차원까지 먼저 축소한 뒤 CCA를 적용**하도록 `src/data/preprocessing.py::pca_reduce` 추가. 이는 속도 문제 회피일 뿐 아니라 실제로 표준적인 방법(Seurat의 CCA도 원본 유전자가 아니라 PCA 성분 위에서 수행)이라 근거도 있음. 적용 후 CCA fit이 84초로 단축.
2. **`gap_metrics.uniformity()`가 실제 데이터 규모에서 메모리 폭발(OOM 직전) 발생.** 합성 데이터 단위테스트는 n≤500으로 작아서 몰랐는데, held-out test set(18,053 cells, 32차원 임베딩)에 대해 실행하니 `(n,n,d)` 형태의 완전 pairwise 텐서를 만들면서 프로세스가 약 40GB+ 상주 메모리를 소비하며 계속 증가 (n=18,053, d=32 기준 naive 방식은 이론상 ~83GB 필요). **실행 중 kill로 중단하고 수정**: (a) unit-normalize된 벡터에서는 `||a-b||² = 2 - 2·(a·b)`이므로 d를 거치지 않고 n×n 행렬 하나(matmul)로 계산 — 메모리를 d배 절감. (b) 그래도 n이 수만 단위면 n² 자체가 부담이므로 `max_n`(기본 5000) 초과 시 랜덤 서브샘플링 추가. `topk_retrieval_accuracy()`도 동일한 이유로 float32 사용 + `max_n`(기본 20000) 캡 추가(단, retrieval은 정의상 "전체 후보 풀 안에서" top-k를 찾는 task이므로 query/target을 같은 인덱스로 함께 서브샘플링 — 후보 풀이 줄어드는 실질적 trade-off이며, 그냥 공짜 근사가 아님을 코드 docstring에 명시).
   - 이 버그는 **실제 규모 데이터로 처음 실행해봐야 드러나는 종류**였다는 점이 중요 — 합성 단위테스트만으로는 통과했었음. 이후 모든 지표 함수를 실제 데이터에 처음 적용할 때는 먼저 작은 서브셋으로 메모리/시간 추정을 해보고 진행하기로 함.
   - 회귀 테스트 추가: `tests/test_gap_metrics.py`에 (a) naive 계산과 결과가 일치하는지, (b) n=20,000~30,000에서 OOM 없이 도는지 확인하는 테스트 3개 추가.

**막힘/이슈**
- 위 두 건 모두 발견 즉시 프로세스를 죽이고 코드 수정 후 재실행하는 방식으로 대응 완료. 데이터 손상이나 되돌릴 수 없는 부작용은 없음(둘 다 순수 계산 중 발생한 문제).

**다음 단계**
- Phase 1 baseline 재실행 (linear CCA, GEX-ADT / GEX-ATAC).
- MatchCLOT-arch(from-scratch) 인코더로도 Phase 1 baseline 실행 (encoder (b)).
- Phase 1 batch confound 분석 스크립트 작성 및 실행.

---

## 2026-08-13 (계속 3) — Phase 1 baseline 첫 실측 결과 (linear CCA), harmonypy 버그 수정

**한 일**
- `phase1_baseline.py` 재실행(OOM 수정 후) 완료. **실제 데이터 기준 첫 결과**:

  | pair | delta_gap | linear_separability | top5_retrieval_acc | alignment |
  |---|---|---|---|---|
  | cite (GEX-ADT) | 0.0467 | 0.530 | 0.062 | 0.832 |
  | multiome (GEX-ATAC) | 0.1559 | 0.679 | 0.123 | 0.762 |

- `harmonypy.run_harmony()`의 반환값 `Z_corr` 방향(orientation)을 실제로 검증 — 이 버전(2.0.0, C++ 백엔드)은 입력과 **같은** (cells × features) 방향으로 반환함을 작은 합성 데이터로 직접 확인. `_harmony_correct()`가 classic pure-python harmonypy의 관례(transposed)를 가정하고 `.T`를 붙이고 있었던 걸 발견 → 수정. `tests/test_phase1_batch_confound.py`에 방향 고정 회귀테스트 + Harmony가 실제로 배치 분리를 줄이는지 확인하는 sanity test 추가.

**판단 및 해석 (중요 — 잠정적 결과이므로 신중하게 기록)**

- **linear CCA 결과만 놓고 보면 계획서의 가설(GEX-ADT gap > GEX-ATAC gap)과 반대 방향**이 나왔다 (GEX-ATAC의 delta_gap이 GEX-ADT의 약 3.3배). 이걸 바로 "가설 기각"으로 해석하면 안 된다고 판단하는 이유:
  - **CCA는 애초에 두 모달리티 간 상관을 최대화하도록 학습되는 방법**이라, "gap"(정렬 안 된 정도)을 목적함수 자체에서 최소화하는 셈이다. 반면 MatchCLOT의 실제 학습 방식(InfoNCE/contrastive)은 정렬을 유도하긴 하지만 Liang et al.(2022)의 원 발견 자체가 "contrastive 학습을 해도 gap이 완전히 없어지지 않는다"는 것이었으므로, CCA와 contrastive 인코더가 만드는 "gap"은 성격이 다르다.
  - 따라서 이번 CCA 결과는 (a) "CCA로 억지로 정렬해도 GEX-ATAC 쪽이 더 안 풀린다" 정도의 약한 신호로만 해석하고, (b) **원래 계획서가 검증하고자 한 가설(정보 비대칭 → contrastive 인코더의 gap)은 encoder (b) MatchCLOT-arch(from-scratch, InfoNCE)의 결과가 나와야 제대로 판단 가능**하다는 게 현재 판단. CODE_MAP.md에도 "잠정치, MatchCLOT-arch 결과와 함께 봐야 함"으로 표시.
- retrieval 정확도(top-5)가 두 pair 모두 낮음(6~12%, chance보다는 확실히 높지만 절대적으로 낮음) — CCA 32차원 공유공간이 cross-modal retrieval에는 약하다는 뜻. 이 자체도 "CCA가 alignment는 어느 정도 잡아도 개별 세포 수준 식별력은 약하다"는 인코더별 특성 차이로 이해.

**막힘/이슈**
- 없음 (harmonypy 버그는 batch confound 실행 전에 합성 데이터 스모크테스트로 미리 잡아서, 실제 실행에는 영향 없음).

**다음 단계**
- Encoder (b) MatchCLOT-arch로 동일한 Phase 1 baseline 재실행 (진짜 비교의 핵심).
- Phase 1 batch confound 스크립트 실행.
- Phase 2 실험 A(dial swipe) 실행.

---

## 2026-08-13 (계속 4) — 전체 계획서(Phase 0~3) 코드 작성 완료, encoder (b) 첫 실측 결과

**한 일**
- Phase 2 실험 B(`phase2_expB_crosstype.py`, `src/data/cell_lineage.py`), 실험 C(`phase2_expC_lineage.py`), Phase 3 통합분석(`phase3_integration.py`, Baron & Kenny mediation) 구현 완료. 이걸로 **계획서 Phase 0~3 전체 코드가 작성되고 단위테스트(48개, 전부 통과)를 거친 상태**가 됨. 아직 실행 전인 것: batch confound, 실험 A/B/C, 통합분석.
- Encoder (b) MatchCLOT-arch(from-scratch, contrastive) baseline 실행 완료 — **계획서가 실제로 검증하려던 핵심 비교**.

**실행 중 발견 및 수정한 버그 2건 (실행 전에 합성데이터로 잡음, 실제 실행에는 영향 없음)**
1. `phase2_expB_crosstype.py`의 `_cosine_sim_pairs()` 내부에서 encoder_modality1/2 배정이 실제 학습 시(`train_modality_clip(gex, other)` → modality1=GEX, modality2=other) 방향과 반대로 되어 있었는데, 호출부에서도 인자 순서를 반대로 넘기고 있어서 **두 실수가 우연히 상쇄되어 결과는 맞지만 코드는 매우 위험한 상태**였음. 둘 중 하나만 나중에 "고치면" 조용히 깨지는 전형적인 함정이라 판단, 명확한 단일 규약으로 리팩터링. 입력 차원이 서로 다른 합성 데이터로 회귀테스트 작성(잘못 배정되면 바로 shape 에러로 터지도록).
2. (사소) exp B 초안에 실행되지 않는 죽은 코드(빈 placeholder 루프) 발견, 실행 전 제거.

**Phase 1 baseline 최종 비교 (linear CCA vs MatchCLOT-arch, 둘 다 3 seed 평균)**

| pair | encoder | delta_gap (mean±std) | linear_separability | top5_retrieval_acc |
|---|---|---|---|---|
| cite (GEX-ADT) | linear CCA | 0.0467 | 0.530 | 0.062 |
| cite (GEX-ADT) | MatchCLOT-arch | 0.0838 ± 0.0032 | 0.783 | 0.184 ± 0.002 |
| multiome (GEX-ATAC) | linear CCA | 0.1559 | 0.679 | 0.123 |
| multiome (GEX-ATAC) | MatchCLOT-arch | 0.0897 ± 0.0020 | 0.845 | 0.249 ± 0.002 |

**판단 및 해석 (중요 — 이번 세션의 핵심 실측 결과)**

- **두 인코더 모두에서 GEX-ATAC의 delta_gap이 GEX-ADT보다 크거나 같게 나왔다 — 계획서 원 가설("GEX-ADT gap이 GEX-ATAC보다 유의미하게 크다")과 반대 방향.** 다만 크기 차이는 인코더에 따라 크게 다르다: linear CCA에서는 ATAC이 ADT보다 ~3.3배 컸는데, MatchCLOT-arch(contrastive)에서는 그 차이가 ~1.07배로 훨씬 작아졌다. → **"어느 쪽 gap이 더 큰가"의 방향 자체는 인코더 선택에 어느 정도 강건해 보이지만, 그 격차의 크기는 인코더(목적함수)에 크게 의존한다**는 것이 현재까지의 잠정 결론. (baseline 인코더 3종 비교의 원래 취지—"gap 순위가 인코더에 강건한지 확인"—가 실제로 의미 있는 발견을 만들어냈다.)
- **gap과 downstream 성능(top-5 retrieval)이 같은 방향으로 안 움직인다**: MatchCLOT-arch에서 GEX-ATAC이 GEX-ADT보다 delta_gap도 크고 top5_retrieval_acc도 더 높다(0.249 vs 0.184). 즉 "gap이 크면 매칭 성능이 나쁘다"는 순진한 예상과 다르게, 이 데이터에서는 gap이 큰 쪽이 오히려 retrieval을 더 잘한다. 이는 이미지-텍스트 CLIP 문헌에서도 보고된 바 있는 패턴(gap 크기와 downstream 성능이 항상 반비례하지는 않음)과 일치하며, "gap 자체"와 "매칭 가능성"을 같은 것으로 섞어서 해석하면 안 된다는 근거가 된다.
- **주의할 한계**: 이번 MatchCLOT-arch 학습은 원 논문의 7000 epoch가 아니라 150 epoch로 축소했고(계산자원/실험 조건 수 트레이드오프, `matchclot_arch.py` docstring에 근거 명시), pretrained weight와 비교도 못 했다(링크 만료). 그래서 이 delta_gap 절대값이나 정확한 비율을 논문 수준 결론으로 취급하면 안 되고, **"방향이 두 인코더에서 일관된다"는 정성적 신호** 정도로만 우선 받아들이는 게 안전하다고 판단. Phase 2 실험(정보 비대칭 조작)의 결과가 이 방향성과 일관되게 나오는지가 다음 검증 포인트.
- Seed 간 분산이 매우 작다(std 0.002~0.003 수준) — 150 epoch 정도로도 이 정도 크기 데이터·모델에서는 학습이 안정적으로 수렴한다는 뜻으로 해석, 반복 seed 수(3개)가 부족하지 않다는 근거로 사용 가능.

**다음 단계**
- Phase 1 batch confound, Phase 2 실험 A/B/C, Phase 3 통합분석을 순서대로 백그라운드 실행 (이미 작성된 스크립트 그대로).
- 각 실행이 끝나는 대로 HISTORY.md에 결과 반영.

---

## 2026-08-13 (계속 5) — Phase 1 batch confound 실행 완료 (실측 결과)

**한 일:** `phase1_batch_confound.py` 전체(full / matched-N × harmony on/off, 8개 조건) 실행 완료.

| pair | condition | harmony | delta_gap | linear_sep | top5_retrieval | r2_batch (p) | r2_modality | celltype_silhouette |
|---|---|---|---|---|---|---|---|---|
| cite | full | off | 0.0467 | 0.530 | 0.062 | 0.1018 (0.005) | 0.0000 | 0.0898 |
| cite | full | on | 0.0477 | 0.558 | 0.028 | 0.0221 (0.005) | 0.0000 | 0.0985 |
| multiome | full | off | 0.1559 | 0.679 | 0.123 | 0.0858 (0.005) | 0.0038 | 0.0539 |
| multiome | full | on | 0.1638 | 0.684 | 0.070 | 0.0189 (0.005) | 0.0045 | 0.0591 |
| cite | matchedN(n=69,249) | off | 0.0479 | 0.531 | 0.074 | 0.0996 (0.005) | 0.0000 | 0.0909 |
| cite | matchedN | on | 0.0483 | 0.547 | 0.036 | 0.0220 (0.005) | 0.0000 | 0.0982 |
| multiome | matchedN(=전체와 동일 N) | off/on | (full과 동일) | | | | | |

**판단 및 해석**

1. **Matched-N 조건이 사실상 full과 동일** — cite를 69,249개로 subsample해도 delta_gap(0.0467→0.0479)이 거의 안 변함. multiome은 애초에 두 데이터셋 중 더 작은 쪽(69,249)이라 matchedN의 n이 곧 자기 자신의 전체 크기와 같아서 subsampling이 실질적으로 발생하지 않음(정확히 같은 값 출력, 버그 아니라 설계상 당연한 결과 — HISTORY에 명시해 나중에 "왜 두 행이 똑같지?"로 헷갈리지 않게 함). **결론: 두 데이터셋의 세포 수 차이(9만 vs 7만)가 gap 크기 차이의 원인이 아니다.**
2. **Harmony on/off — 배치효과가 gap의 주원인이 아님을 재확인** (이전 항목에서 이미 기록한 내용과 일관). 배치가 설명하는 분산(r2_batch)이 두 pair 모두 ~78% 감소하는데도 delta_gap은 거의 그대로(cite는 오히려 미세 증가, multiome도 미세 증가). Cell-type silhouette도 유지/소폭 상승 → 과교정 없음.
3. **예상 밖의 흥미로운 부작용 발견**: Harmony 적용 후 **top5_retrieval_acc가 뚜렷하게 하락**했다 (cite 0.062→0.028, multiome 0.123→0.070 — 절반 이하로). 반면 linear_separability는 오히려 소폭 상승(cite 0.530→0.558, multiome 0.679→0.684). 즉 Harmony는 "배치 매크로 구조"는 잘 지우지만(r2_batch↓), 개별 세포 수준의 미세 대응 관계(retrieval에 필요한 정보)는 오히려 훼손하는 것으로 보인다. 가능한 원인: 본 구현에서 GEX와 ADT/ATAC을 각각 독립적으로(같은 배치 라벨 기준이지만 서로 참조 없이) harmonize한 뒤 CCA를 새로 fit하다 보니, 두 모달리티 사이에 남아있던 세포 수준 미세 정렬 정보가 배치 보정 과정에서 함께 뭉개졌을 가능성. **이건 배치효과 제거와 gap 측정이 서로 다른 목적의 전처리라는 걸 보여주는 유용한 발견**으로 기록 — 이후 실험에서 "batch corrected" 조건과 "raw" 조건의 downstream 성능을 같은 잣대로 비교할 때 이 트레이드오프를 염두에 둬야 함.

**다음 단계:** 실험 A(정보량 dial swipe) 실행 중.

---

## 2026-08-13 (계속 6) — 실험 A(정보량 dial swipe) 실행 완료 — 세션의 핵심 결과

**한 일:** `phase2_expA_dial_swipe.py` 전체(quantity 5조건 + quality 4조건, 각 3 seed = 27회 학습) 실행 완료. GEX-ADT(cite) pair 기준.

**Quantity 축 (HVG 개수 50→134→500→2000→전체 13,953)**

| n_genes | delta_gap (평균±표준편차) | top5_retrieval (평균) |
|---|---|---|
| 50 | 0.5875 ± 0.0023 | 0.0245 |
| 134 | 0.3313 ± 0.0009 | 0.0576 |
| 500 | 0.1363 ± 0.0024 | 0.1426 |
| 2,000 | 0.0838 ± 0.0032 | 0.1843 |
| 13,953(전체) | 0.0758 ± 0.0066 | 0.1189 |

**Quality 축 (134 또는 36개 유전자, "무엇을" 넣었는지 비교)**

| condition | n_genes | delta_gap (평균±표준편차) |
|---|---|---|
| random_134hvg | 134 | 0.3313 ± 0.0009 |
| adt_matched (ADT gene_id와 실제 매칭되는 유전자) | 36 | 0.3634 ± 0.0019 |
| stat_matched_random (발현통계량만 맞춘 무작위 유전자) | 36 | 0.7516 ± 0.0102 |
| adt_matched_pair_shuffled (매칭 유전자, 세포 대응관계는 셔플) | 36 | 0.9831 ± 0.0011 |

**판단 및 해석 (이번 세션에서 가장 중요한 실측 결과)**

1. **Quantity 축은 계획서의 원 가설과 정반대다.** 유전자 수가 50→13,953으로 늘수록(3 seed 모두 표준편차가 극히 작아 신뢰도 높음) delta_gap이 0.588→0.076으로 **거의 8배 단조 감소**한다. 원 계획서는 "정보량↑→gap↑" (정보 비대칭이 커질수록 gap도 커진다, Schrödi et al.의 단순 독해)을 예상했는데 정반대다. **재해석**: 정보 비대칭 이론을 "GEX가 ADT보다 정보를 많이 가질수록 gap이 커진다"는 식으로 단순화하면 안 되고, 오히려 "GEX가 ADT와 매칭할 수 있을 만큼 충분한 정보를 갖고 있는가"가 핵심일 가능성이 크다 — 유전자가 너무 적으면 ADT 신호와 대응할 만한 정보 자체가 부족해서 gap이 커지고, 유전자가 늘수록 그 부족이 해소되며 gap이 줄어든다는 쪽이 데이터와 더 잘 맞는다.
2. **하지만 gap과 실제 매칭 성능(top5 retrieval)은 다른 지점에서 최적점을 가진다.** retrieval은 2,000개에서 정점(0.184)을 찍고 전체 유전자(13,953개)에서는 오히려 떨어진다(0.119) — gap은 계속 (조금씩) 줄어드는데도 그렇다. 즉 "정보가 많을수록 좋다"도 아니고 "정보가 적을수록 gap이 크다"도 완전한 그림이 아니며, **너무 많은(대부분 무관한) 유전자를 넣으면 개별 세포 수준의 매칭에는 오히려 노이즈로 작용**하는 것으로 보인다. Harmony 분석에서 이미 나온 교훈("gap 지표와 downstream 성능은 같이 안 움직일 수 있다")이 여기서도 재확인됨.
3. **Quality 축은 계획서의 가설과 방향이 일치한다 — 그리고 매우 뚜렷하다.** 같은 36개 유전자 수를 놓고 비교했을 때:
   - ADT와 생물학적으로 대응되는 유전자(adt_matched) → gap 0.363
   - 발현 통계만 맞춘 무관한 무작위 유전자(stat_matched_random) → gap 0.752 (**2배 이상**)
   - 이건 "얼마나 많이"가 아니라 "무엇이 들어있는지"가 gap에 실제로 영향을 준다는 걸 깨끗하게 보여준다. 원 계획서가 "공유되는 정보량이 gap을 좌우한다"고 예상한 부분이 quality 축에서는 그대로 확인됨.
   - 대조군(adt_matched_pair_shuffled, 같은 유전자인데 세포 짝만 무작위로 섞음) → gap 0.983으로 가장 크다. **올바른 유전자 내용보다 올바른 세포 대응관계가 더 근본적**이라는 당연하지만 확인이 필요했던 사실을 재확인 (내용이 맞아도 짝이 틀리면 그보다 더 나쁘다).
   - 참고로 adt_matched(36개, gap 0.363)가 random_134hvg(134개, gap 0.331)와 거의 같은 수준인 것도 흥미롭다 — 유전자 수는 훨씬 적은데(36 vs 134) gap은 비슷하다는 건, "생물학적으로 관련된 소수 유전자"가 "무작위로 고른 훨씬 많은 유전자"만큼의 정렬력을 갖는다는 뜻으로, quality가 quantity의 부족을 상당 부분 상쇄할 수 있음을 시사한다.
4. **한계 (다음에 검증해야 할 것)**: adt_matched 유전자가 계획서가 기대한 134개가 아니라 **36개**만 확보됐다 (ADT 항체 상당수가 isotype control이거나 gene_id가 없어서 GEX 쪽과 매칭이 안 됨). 그래서 quality 축의 "134개 무작위 vs 134개 매칭"이라는 원래 설계는 못 지켰고, 대신 "36개 매칭 vs 36개 통계-매칭 무작위"로 개수를 맞춰 비교했다 — 이게 실제로 공정한 비교이고, random_134hvg는 참고용 규모 비교로만 사용. 이 36이라는 숫자 자체도 흥미로운 정보(ADT 패널의 실제 생물학적 매칭 가능 비율)이므로 CODE_MAP/논문화 단계에서 명시해야 함.

**다음 단계:** 실험 B(cross-cell-type 미스매칭) 실행 중. 실험 C, Phase 3 통합분석 순서대로 진행.

---

## 2026-08-13 (계속 7) — 실험 B, C, Phase 3 통합분석 실행 완료 — 전체 계획서 실행 1회전 완료

**한 일:** 남은 실행 체인(`phase2_expB_crosstype.py`, `phase2_expC_lineage.py`, `phase3_integration.py`)이 전부 성공적으로 끝남 (exit code 0). 이걸로 **계획서 Phase 0~3 전체를 한 바퀴 다 돌렸다.**

### 실험 B — Cross-cell-type 미스매칭 결과

| condition | n_pairs | mean_sim | null 대비 p-value |
|---|---|---|---|
| true_pair | 2000 | 0.647 | 0.002 |
| same_type_diff_object | 2000 | 0.412 | 0.002 |
| same_lineage_diff_type | 2000 | 0.171 | 0.002 |
| diff_lineage | 2000 | -0.003 | **1.000** |
| random_pair (null) | 2000 | 0.009 | — |

**해석**: 계획서가 예상한 그대로 깨끗한 단조 감소 패턴이 나왔다 — true_pair > same_type_diff_object > same_lineage_diff_type > diff_lineage ≈ random_pair(null). 특히 diff_lineage의 p-value가 정확히 1.0으로, "완전히 다른 계통이면 무작위 쌍과 통계적으로 구분이 안 된다"는 걸 아주 깔끔하게 보여준다. 반대로 "같은 세포타입이면 다른 개체여도" 유사도가 여전히 매우 높다(0.412, null 대비 유의) — 인코더가 개별 세포를 외운 게 아니라 세포타입 수준의 일반화된 정보를 학습했다는 뜻이고, 이건 계획서가 exp B를 설계한 목적(개체 각인 vs 세포타입 편향 구분) 그대로 확인된 것이다.

### 실험 C — 단일 계통 heterogeneity dose-response 결과

| condition | n | n_cell_types | heterogeneity(entropy) | delta_gap | top5_retrieval |
|---|---|---|---|---|---|
| full_all_lineages | 90,261 | 45 | 3.069 | 0.0802 | 0.182 |
| single_lineage_T_CD4 | 14,621 | 5 | 1.074 | 0.0768 | 0.208 |
| matchedN_for_T_CD4 | 14,621 | 45 | 3.071 | **0.0547** | 0.228 |
| single_lineage_T_CD8 | 10,663 | 9 | 1.982 | 0.1113 | 0.183 |
| matchedN_for_T_CD8 | 10,663 | 45 | 3.054 | **0.0620** | 0.276 |
| single_lineage_Myeloid_Mono | 24,328 | 2 | 0.343(최저) | 0.0775 | 0.120 |
| matchedN_for_Myeloid_Mono | 24,328 | 45 | 3.079 | **0.0634** | 0.221 |
| single_lineage_B_cell | 8,977 | 5 | 1.401 | 0.0704 | 0.159 |
| matchedN_for_B_cell | 8,977 | 45 | 3.091 | **0.0576** | 0.222 |
| single_lineage_NK_ILC | 8,391 | 4 | 0.911 | 0.0772 | 0.145 |
| matchedN_for_NK_ILC | 8,391 | 45 | 3.089 | **0.0531** | 0.222 |

**판단 및 해석 (중요 — 계획서 가설을 반증하는 방향의 결과, 숨기지 않고 그대로 기록)**

- **5개 계통 전부에서, N을 맞췄을 때 오히려 matchedN(전체 이질성 유지) 쪽의 delta_gap이 single-lineage(이질성 낮춤) 쪽보다 작다.** 즉 이질성을 낮추면 gap이 줄어들 거라는 계획서의 가설과 **정반대 방향**의 결과다. Myeloth_Mono가 heterogeneity entropy 기준 가장 낮은 계통(0.343)인데도 gap이 특별히 작지 않다는 것도 같은 방향.
- 계획서 원문이 이 실험에 대해 미리 정해둔 반증 기준을 그대로 인용하면: *"만약 이 조건에서 gap이 줄어들지 않는다면 인과관계가 single-cell 데이터에서는 성립하지 않는다는 증거가 된다"*. **이 기준에 따르면 이번 결과는 "heterogeneity 감소 → gap 감소"라는 인과 경로가 (적어도 이 구현·이 데이터에서는) 성립하지 않는다는 증거로 해석해야 한다.** 좋은 결과가 아니라고 숨기는 게 아니라, 계획서가 미리 정해둔 판정 기준을 그대로 적용한 정직한 결론이다.
- 가능한 이유(추측, 추가 검증 필요): 단일 계통으로 좁히면 ADT 134개 항체 패널 중 그 계통에서는 거의 발현되지 않는(즉 사실상 상수에 가까운) 마커가 많아져서, contrastive loss가 학습에 쓸 수 있는 유효한 대조 신호(negative pair들 사이의 변별력)가 오히려 줄어드는 것일 수 있다 — heterogeneity 감소가 "GEX가 인코딩해야 할 정보량 감소"로 이어지긴 하지만, 동시에 "InfoNCE가 학습에 활용할 수 있는 유효 신호"도 함께 줄어들어 오히려 정렬이 더 어려워질 가능성. 이건 검증 안 된 사후 가설이므로 그대로 확정하지 않고 다음 조사 항목으로 남겨둠.
- retrieval 성능은 오히려 matchedN 쪽이 대체로 더 높다(예: T_CD8 matchedN 0.276 vs single 0.183) — gap과 downstream 성능이 여기서도 같은 방향으로 안 움직인다는 걸 재확인.

### Phase 3 — 통합 mediation 분석 (실험 A quantity 축 15개 관측치 기준)

```
Step 1 (asymmetry -> gap):        coef=-0.0852, p<0.001   (정보 비대칭↑ → gap↓, exp A 결과와 일관)
Step 2 (asymmetry -> performance): coef=+0.0204, p=0.003   (정보 비대칭↑ → 성능↑)
Step 3 (asymmetry+gap -> perf):    asymmetry coef=-0.0092(비유의, p=0.196), gap coef=-0.3475(유의, p<0.001)
→ Mediation 신호 확인됨: gap이 유의하고 asymmetry의 직접효과가 사실상 사라짐 (0.0204 → -0.0092)
```

**해석**: 통계적으로는 "정보 비대칭 → gap → 성능"이라는 매개 구조 자체는 확인된다(gap이 유의한 매개변수, 직접효과 소멸). 다만 **방향은 계획서 원 가설과 반대**다 — 여기서 "정보 비대칭"은 log(유전자수/134)로 정의했으므로, 비대칭이 커질수록(유전자가 많아질수록) gap은 줄고 성능은 오른다. 그리고 gap 자체는 성능에 음의 방향으로 작동한다(gap↑ → 성능↓, 계획서가 원래 기대한 방향). **주의할 점**: 이 회귀는 asymmetry_index가 정의된 exp A quantity 축 15개 관측치에만 기반한다(quality 축·exp C는 이 척도로 환산 불가능하다고 판단해 제외 — `phase3_integration.py` docstring에 명시). 표본이 작으므로 통계적으로 유의하긴 해도 이 계수 크기를 과신하지 않아야 한다.

**세션 전체를 관통하는 핵심 그림 (Phase 0~3 1회전 결론 요약)**
1. Batch effect는 gap의 원인이 아니다 (Phase 1).
2. "정보량"(유전자 개수) 자체는 계획서 예상과 반대로 gap을 줄인다 — 아마도 "충분한 정보"의 문제였지 "너무 많은 정보"의 문제가 아니었던 것 같다 (실험 A quantity).
3. "정보의 질"(내용이 실제로 관련 있는가)은 계획서 예상대로 gap에 영향을 준다 (실험 A quality).
4. Cross-cell-type 정보는 계획서 예상대로 계통 거리에 따라 매끄럽게 감쇠한다 (실험 B).
5. Heterogeneity 감소가 gap을 줄인다는 가설은 **이번 데이터·구현에서는 반증됐다** (실험 C) — 계획서가 스스로 정한 반증 기준에 따름.
6. Gap 크기와 downstream 매칭 성능은 자주 같이 안 움직인다 — 이건 여러 실험(Harmony, quantity 축, exp C)에서 반복적으로 나타난 패턴이라 우연이 아닐 가능성이 크다.

**다음 단계 (2회차로 넘어갈 때 우선순위)**
- 실험 C의 반증 결과가 재현되는지 다른 계통 조합·다른 seed로 재확인.
- exp A quantity 축을 GEX-ATAC(multiome) pair에도 적용해서 방향이 일관되는지 확인.
- "gap과 성능이 왜 자주 어긋나는지"를 직접 조사하는 후속 분석(예: gap을 uniformity/alignment로 분해해서 어느 성분이 성능과 실제로 상관되는지).

---

## 디렉토리 구조 (참고용, 바뀔 때마다 갱신)

```
cv-session-1/
├── README.md
├── docs/
│   ├── PLAN.md            # 계획서 원문 (기준 문서)
│   ├── HISTORY.md          # 이 파일 — 시간순 작업/판단/오류 로그
│   └── CODE_MAP.md         # 계획서 항목 ↔ 코드 파일 매핑
├── src/
│   ├── data/                # 데이터 다운로드, 로딩, 전처리, split
│   ├── encoders/             # MatchCLOT wrapper, from-scratch encoder, linear CCA+OT baseline
│   ├── metrics/              # gap 지표 (Δgap, alignment-uniformity, linear separability, retrieval)
│   ├── experiments/          # phase1_*.py ~ phase3_*.py, 계획서 실행 스크립트
│   └── utils/                # 공통 유틸 (seed 고정, 로깅 등)
├── external/                 # 서드파티 레포 클론 (git에는 미포함, setup 스크립트로 재현)
├── data/                      # 다운로드된 원본/중간 데이터 (git에는 미포함)
└── results/
    ├── tables/                # 실험 결과 요약 (csv/json, git 포함)
    └── figures/               # 결과 그림 (git 포함, 큰 원본은 제외)
```
