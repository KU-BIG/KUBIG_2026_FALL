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
