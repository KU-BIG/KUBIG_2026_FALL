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
