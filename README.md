# Same Gap, Two Worlds

### 이미지-텍스트와 단일세포 멀티오믹스에서의 modality gap 비교 분석

> KUBIG(고려대학교 빅데이터 학회) 23기 프로젝트 팀 **"Come back to me 현주"** — 윤현주, 이병각, 정준호

CLIP류 대조학습(contrastive learning) 기반 멀티모달 모델의 임베딩 공간에는, 서로 다른 모달리티가 하나로
섞이지 않고 분리된 두 군집을 이루는 **modality gap** 현상이 존재한다. 이 프로젝트는 Schrödi et al.
(ICLR 2025)이 제시한 "**모달리티 간 정보 비대칭(information imbalance)이 modality gap과 object bias를
유발한다**"는 가설이

1. 이미지-텍스트라는 특정 도메인에 국한된 현상인지, 아니면
2. 대조학습 기반 멀티모달 표현 학습 전반에 걸친 더 일반적인 현상인지

를 확인하기 위해, **서로 다른 도메인(이미지-텍스트 vs. single-cell multi-omics)과 서로 다른 정보 비대칭
조작 방식(인위적 조작 vs. 생물학적으로 이미 존재하는 자연발생 비대칭)으로 동일한 가설을 교차검증**한다.

---

## 목차

- [연구 배경](#연구-배경)
- [연구 질문과 접근 방식](#연구-질문과-접근-방식)
- [저장소 구조](#저장소-구조)
- [트랙별 핵심 결과](#트랙별-핵심-결과)
  - [1. Image-Text 트랙 (SemArt)](#1-image-text-트랙-semart)
  - [2. Single-Cell Multi-Omics 트랙 (BMMC)](#2-single-cell-multi-omics-트랙-bmmc)
- [통합 결론](#통합-결론)
- [재현 방법](#재현-방법)
- [참고 문헌](#참고-문헌)
- [팀](#팀)

---

## 연구 배경

| 문헌 | 기여 |
|---|---|
| Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021 | 이미지-텍스트를 각각 별도 인코더로 임베딩한 뒤 대조학습으로 정렬하는 dual-encoder 구조. 이후 모든 멀티모달 대조학습 연구의 표준 아키텍처가 됨 |
| Liang et al., *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*, NeurIPS 2022 | modality gap 현상을 처음 정의·명명. 원인으로 (1) 초기화 시 표현이 좁은 원뿔로 제한되는 cone effect, (2) 온도(temperature) 파라미터가 학습 중에도 분리를 유지시키는 경향을 제시. gap 크기가 zero-shot 성능·공정성과 연결됨을 실증 |
| Schrödi et al., *Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models*, ICLR 2025 (Oral) | modality gap과 object bias가 사실은 **정보 비대칭(information imbalance)** 이라는 하나의 공통 원인에서 비롯된다고 주장. 이 프로젝트 전체가 검증하는 핵심 가설(H1)의 출처 |
| Levi & Gilboa, *Double-Ellipsoid Geometry of CLIP*, ICML 2025 | 텍스트 집합 내부의 전형성(conformity)으로 CLIP 임베딩 공간의 기하를 설명. Image-Text 트랙의 `conformity-groundedness/` 서브프로젝트가 여기에 groundedness(접지성) 축을 추가해 확장 |
| Fahim et al. (2024), *It's Not a Modality Gap* | linear separability 지표의 출처. `semart-pipeline/`이 채택 |

## 연구 질문과 접근 방식

**연구 질문.** 위 문헌들이 공통으로 다루는 modality gap과, Schrödi et al.의 "정보 비대칭이 gap을 유발한다"는
가설이 이미지-텍스트 도메인에 국한된 현상인지, 아니면 대조학습 기반 멀티모달 표현 학습 전반에 걸친 더
일반적인 현상인지 확인한다.

**접근 방식.** 이미지-텍스트(CLIP, SemArt) 데이터에서 정보 비대칭을 인위적으로 조작하는 실험과, single-cell
multi-omics(GEX-ADT, GEX-ATAC) 데이터에서 생물학적으로 이미 존재하는 자연발생 정보 비대칭을 활용하는
실험을 병행한다. 이미지-텍스트 쌍은 사람이 작성한 캡션에 기반한 매칭이라 노이즈가 섞일 수 있는 반면,
single-cell 데이터는 **동일 세포에서 물리적으로 동시 측정된 진짜 ground-truth 쌍**을 가지므로, 매칭
노이즈를 원천적으로 배제하고 정보 비대칭의 순수 효과만 분리해서 볼 수 있다는 점에서 두 도메인은 서로를
보완하는 교차검증 축이 된다.

## 저장소 구조

```
cv-session-1/
├── README.md                                  # 이 문서 — 프로젝트 전체 개요
│
├── image-text/                                # 트랙 1: 이미지-텍스트 (SemArt, frozen CLIP)
│   ├── README.md                              #   두 서브프로젝트의 관계 · 공통 결론
│   ├── semart-pipeline/                       #   1-A. Stage A/B 정량 스윕 파이프라인 (이병각)
│   │   ├── README.md, docs/                   #     데이터 전처리·실험 설계·결과 기록
│   │   ├── config/                            #     Stage A/B 파라미터, 검증 허용오차
│   │   ├── src/                               #     data / embeddings / metrics / interventions / downstream
│   │   ├── outputs/{results,figures}/         #     Step 1~7c 산출물 (JSON + PNG)
│   │   └── tests/
│   └── conformity-groundedness/                #   1-B. Conformity × Groundedness 탐색 (정준호)
│       ├── README.md, PROJECT_v2.md, PROJECT_v3.md
│       ├── src/                               #     이미지 열화(degrade) / 임베딩(encode) / 지표(metrics)
│       ├── scripts/                           #     Phase별 실행 스크립트 (v2: U-curve·마스킹, v3: conformity·groundedness)
│       └── results/{*.json,figures/}
│
├── single-cell-multiomics/                    # 트랙 2: Single-Cell Multi-Omics (BMMC, MatchCLOT) (윤현주)
│   ├── README.md
│   ├── docs/                                  #   PLAN.md(계획) · HISTORY.md(의사결정 로그) · CODE_MAP.md · SLIDE_HANDOFF.md
│   ├── src/
│   │   ├── data/                              #   로딩·전처리·batch split·cell lineage 매핑
│   │   ├── encoders/                          #   linear CCA baseline / MatchCLOT-arch(InfoNCE 재구현)
│   │   ├── metrics/                           #   gap_metrics(Δgap·alignment·separability·retrieval) · variance_partitioning
│   │   └── experiments/                       #   phase1(baseline·batch confound) · phase2(exp A/B/C) · phase3(통합 mediation)
│   ├── results/{tables/*.csv, figures/}
│   ├── tests/
│   ├── data/, external/                       #   원본 데이터·MatchCLOT 클론 (용량 문제로 git 미포함, 스크립트로 재현)
│   └── requirements.txt
│
└── (각 트랙 하위 디렉토리의 .gitignore가 대용량 데이터·임베딩 캐시·가상환경을 별도로 관리)
```

세 디렉토리는 원래 `hyunju`(single-cell) / `gak`(SemArt 파이프라인) / `junho`(conformity·groundedness) 세
개의 독립된 브랜치에서 각자 진행되던 작업이다. 이 저장소는 세 브랜치의 **커밋 히스토리를 보존한 채**
(`git log`로 각 파일의 원 저자·작업 과정을 그대로 추적 가능) 위와 같은 트랙 구조로 재구성한 것이다. 각
디렉토리는 자기 완결적인 서브프로젝트로, 자체 `README.md` / `requirements.txt` / `.gitignore` / `tests/`를
가진다.

## 트랙별 핵심 결과

### 1. Image-Text 트랙 (SemArt)

**[`image-text/semart-pipeline/`](image-text/semart-pipeline)** — 텍스트를 visual(시각 관련)/contextual(맥락
정보)로 재분류하고, 토큰 길이를 통제(Stage A, 9.5% 차이)한 뒤 비대칭을 확대(Stage B, 34.9% 차이)하며
$\Delta_{gap}$·retrieval·zero-shot classification을 측정.

| 조건 | $\Delta_{gap}$(visual) | $\Delta_{gap}$(contextual) | 판정 |
|---|---|---|---|
| Stage A (토큰 예산 통제) | 0.8541 | 0.8609 | **H1 방향 지지** (t=7.96, p<0.001) |
| Stage B (토큰 비대칭 확대, treated n=3,854) | 0.8914 | 0.8629 | **H1 방향과 반대로 역전** |

→ 정보 관련성(quality) 효과는 존재하지만, 정보량(quantity) 비대칭이 커지면 그 효과가 역전된다 — **정보량이
정보 종류보다 gap에 더 지배적**일 수 있음. 이후 embedding-shift 개입 실험에서는 gap을 줄인다고 다
좋아지는 것도 아님이 드러났다: retrieval은 모든 조건에서 gap이 작아질수록 오히려 나빠지고(R@1 기준
0.142→0.0175까지 급락), zero-shot classification은 gap을 원래보다 76% 더 벌린 지점( $\lambda=2.2$)에서
오히려 성능이 peak(micro acc. 0.682)를 찍는다.

**[`image-text/conformity-groundedness/`](image-text/conformity-groundedness)** — 이미지 열화(블러·다운샘플·crop)로
Schrödi의 U자 곡선을 먼저 찾아본 뒤(v2 트랙, **3종 열화 모두에서 재현 실패** — 단조 증가만 관찰), Double-Ellipsoid의
conformity(전형성) 개념에 **groundedness(접지성)** 축을 추가해 재접근(v3 트랙).

| 실험 | 결과 |
|---|---|
| 물체 마스킹 vs 배경 마스킹 (같은 면적) | 캡션이 지시하는 물체를 가리면 gap이 **3.5배 더 크게 증가** (+0.0609 vs +0.0043) → object bias 메커니즘을 직접 뒷받침 |
| Conformity vs Groundedness 산점도 | 사실상 무상관 — conformity(전형성)만으로는 groundedness(접지성)를 전혀 설명 못함. SemArt에는 COCO와 달리 "독특하지만 이미지와 무관한" 텍스트가 대량 존재 |
| Groundedness 기반 캡션 압축 | 앞쪽/무작위 절단은 성능을 깎지만, groundedness 상위 문장만 남기면 길이를 절반(91.9→46.9 단어)으로 줄이면서 **retrieval 성능은 오르고(24.2%→27.4%) gap은 줄어듦** |

→ **"정보의 양보다 캡션과의 관련성"** 이 gap과 downstream 성능을 동시에 설명하는 축이라는 것을 실증적
응용으로까지 보인 결과. `semart-pipeline/`의 결론과 방향이 일치한다 (자세한 관계는 [`image-text/README.md`](image-text/README.md) 참고).

### 2. Single-Cell Multi-Omics 트랙 (BMMC)

**[`single-cell-multiomics/`](single-cell-multiomics)** — 사람 골수(BMMC, GEO GSE194122) 단일세포 데이터의
GEX-ADT(RNA-단백질, CITE-seq)와 GEX-ATAC(RNA-DNA, Multiome) 쌍에서, MatchCLOT(dual-encoder + InfoNCE) 구조를
직접 재구현해 baseline·batch confound 분리·정보 비대칭 개입(quantity/quality) 세 단계로 검증.

| 실험 | 핵심 결과 |
|---|---|
| Baseline (생물학적 조절 단계 수 vs gap) | GEX-ADT(여러 조절 단계, 격차 큼)가 GEX-ATAC(같은 조절 층위, 격차 작음)보다 gap이 클 것이라는 예상과 **반대로**, 두 인코더 모두에서 GEX-ATAC의 gap이 같거나 더 큼 |
| Quantity 축 (HVG 유전자 수 50→13,953개) | gap이 0.588→0.076으로 **약 8배 단조 감소** — "정보량이 많을수록 비대칭도 커져 gap도 커질 것"이라는 원 가설과 정반대 |
| Quality 축 (36개 유전자 고정, 내용만 변경) | ADT와 생물학적으로 대응되는 유전자 세트(0.363)가 무관한 유전자 세트(0.752)보다 gap이 **2배 이상 작음** — 세포 쌍을 무작위로 셔플하면 gap이 가장 커짐(0.983), 올바른 대응관계가 가장 근본적인 요소임을 확인 |

→ 표면적으로 quantity 축과 quality 축의 결과가 반대 방향처럼 보이지만, 재해석하면 하나로 수렴한다: HVG
개수를 늘리는 것은 발현 분산이 큰(=cell identity를 구분하는) 유전자를 우선 포함시키는 과정이라 **정보
비대칭을 키우는 게 아니라 ADT와 공유되는 구조를 오히려 복원**한 것이었다. 즉 **modality gap을 결정하는
것은 두 모달리티 사이의 정보량 차이나 생물학적 거리 자체가 아니라, 두 모달리티가 공유하는 학습 가능한
구조의 비율(shared exploitable structure)** 이다.

## 통합 결론

두 트랙은 서로 다른 도메인·서로 다른 정보 비대칭 조작 방식(인위적 조작 vs. 자연발생 비대칭)으로 수행됐지만
같은 결론에 도달한다:

> **Modality gap을 줄이는 것은 두 모달리티 사이에 놓인 정보의 "양"이 아니라, 두 모달리티가 실제로 공유하고
> 인코더가 학습 가능한 형태로 정렬해낼 수 있는 "구조의 비율"이다.** Schrödi et al.의 정보 비대칭 가설은
> 이 공유 구조 비율의 관점으로 재정의될 때, 이미지-텍스트 트랙의 quantity/quality 역전 현상과 single-cell
> 트랙의 baseline/quantity/quality 세 실험 결과를 하나의 설명으로 통합할 수 있다.

동시에 두 트랙 모두 **gap의 크기와 downstream 성능이 항상 같은 방향으로 움직이지는 않는다**는 것도
공통으로 확인했다 — 이미지-텍스트에서는 gap을 원본보다 더 벌린 지점에서 classification이 오히려
peak를 찍었고, single-cell에서는 GEX-ATAC이 GEX-ADT보다 gap도 크고 retrieval 성능도 더 높았다. modality
gap은 임베딩 공간에서의 정렬 정도를 보여주는 지표이지, 정렬의 정밀도나 downstream 유용성을 보장하는
지표는 아니라는 뜻이다.

## 재현 방법

각 트랙은 독립적인 Python 환경(자체 `requirements.txt`)을 사용하는 별도 서브프로젝트다. 재현 절차·데이터
준비 방법·전체 실행 커맨드는 각 디렉토리의 README에 정리되어 있다.

```bash
# Image-Text · SemArt Stage A/B 파이프라인
cd image-text/semart-pipeline && pip install -r requirements.txt
# → README.md 참고 (config/, Step 1~7c 스크립트 실행 순서)

# Image-Text · Conformity/Groundedness
cd image-text/conformity-groundedness && pip install -r requirements.txt
# → README.md의 "데이터" 절 참고 (COCO/SemArt 각자 준비 필요, 라이선스상 git 미포함)

# Single-Cell Multi-Omics
cd single-cell-multiomics && pip install -r requirements.txt
# → docs/PLAN.md, docs/CODE_MAP.md 참고 (GSE194122 다운로드, MatchCLOT 클론 필요)
```

## 참고 문헌

1. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
2. Liang, W., et al. (2022). *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS.
3. Schrödi, S., et al. (2025). *Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models.* ICLR (Oral).
4. Levi, M., & Gilboa, G. (2025). *Double-Ellipsoid Geometry of CLIP.* ICML.
5. Fahim, A., et al. (2024). *It's Not a Modality Gap.*
6. Gossi, F., et al. (2022). *MatchCLOT: Single-Cell Modality Matching with Contrastive Learning and Optimal Transport.* bioRxiv.
7. Luecken, M. D., et al. (2021). *NeurIPS 2021 Multimodal Single-Cell Data Integration Challenge* (OpenProblems, GEO accession [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)).
8. García, N., Renoust, B., & Nakashima, Y. (2018). *ContextNet: Art Text Retrieval and Attribute-Guided Art Search* — SemArt 데이터셋. [noagarcia.github.io/SemArt](https://noagarcia.github.io/SemArt/).

## 팀

**KUBIG 23기 · Come back to me 현주**

| 이름 | 담당 트랙 | 디렉토리 |
|---|---|---|
| 윤현주 | Single-Cell Multi-Omics (BMMC, MatchCLOT 재구현) | [`single-cell-multiomics/`](single-cell-multiomics) |
| 이병각 | Image-Text · SemArt Stage A/B 정량 스윕 파이프라인 | [`image-text/semart-pipeline/`](image-text/semart-pipeline) |
| 정준호 | Image-Text · Conformity/Groundedness 탐색 및 응용 | [`image-text/conformity-groundedness/`](image-text/conformity-groundedness) |
