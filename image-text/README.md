# Image-Text Track — CLIP Modality Gap on SemArt

이 디렉토리는 프로젝트의 "이미지-텍스트" 축을 담당하는 두 개의 독립된 서브프로젝트를 담고 있다. 둘 다
frozen CLIP에서 뽑은 임베딩만으로 분석하는 사후 분석(post-hoc analysis)이며, 같은 데이터셋
([SemArt](https://noagarcia.github.io/SemArt/): 유럽 고전 회화 21,384점 + 작품 해설)과 같은 상위
질문("Schrödi et al.의 정보 비대칭(information imbalance) 가설이 modality gap을 얼마나 잘 설명하는가")을
공유하지만, 서로 다른 실험 설계로 접근한다.

| 서브프로젝트 | 담당 | 접근 방식 |
|---|---|---|
| [`semart-pipeline/`](semart-pipeline) | 이병각 | 텍스트를 **visual(시각 관련) / contextual(맥락 정보)** 로 재분류하고, 토큰 길이를 통제(Stage A)·비대칭 확대(Stage B)한 뒤 동일한 Step 1~7c 파이프라인으로 $\Delta_{gap}$·retrieval·zero-shot classification을 측정하는 **정량 스윕형 파이프라인** |
| [`conformity-groundedness/`](conformity-groundedness) | 정준호 | 이미지 열화(블러·다운샘플·crop·물체/배경 마스킹)로 U자 곡선을 먼저 찾아본 뒤(v2, 반증됨), Double-Ellipsoid conformity 개념에 **groundedness(접지성)** 축을 추가해 두 지표의 (무)상관 관계를 밝히고 이를 캡션 압축에 응용하는 **탐색형 사후 분석** |

## 두 서브프로젝트가 만나는 지점

두 트랙은 독립적으로 수행됐지만 **같은 결론으로 수렴**한다: *"gap을 줄이는 것은 정보량(quantity) 자체가 아니라
정보의 관련성(quality/groundedness)이다."*

- `semart-pipeline/`은 이것을 **통제 실험**으로 보인다 — 토큰 수를 맞춘 Stage A에서는 H1(visual gap < contextual
  gap)이 성립하지만, 토큰 비대칭을 34.9%까지 벌린 Stage B에서는 세 지표( $\Delta_{gap}$, paired cosine, pair
  margin) 모두 역전된다. 즉 **정보량이 정보 종류(quality)를 압도**한다.
- `conformity-groundedness/`는 이것을 **관찰 + 응용**으로 보인다 — 이미지를 얼마나 열화시키든(정보량↓) U자
  곡선은 나오지 않고 단조 증가만 나오지만, **같은 면적을 가려도 캡션이 지시하는 물체를 가리면 gap이 3.5배 더
  크게 증가**한다(quality가 quantity보다 지배적). 이 발견을 실제로 활용해 groundedness 기준으로 캡션을
  절반으로 압축하면 — 단순 절단/무작위 절단과 달리 — retrieval 성능은 오르고 gap은 줄어든다.

두 결과가 언뜻 반대로 들리지만("정보량이 지배적" vs "정보 관련성이 지배적") 실제로는 상충하지 않는다:
`semart-pipeline/`의 Stage B에서 텍스트를 자를 때 사라진 정보는 대부분 여전히 이미지와 관련된(grounded)
문장이었고, `conformity-groundedness/`의 마스킹 실험은 애초에 "무엇을 지우는가"만 다르게 통제했다. 즉
**"길이를 줄일 때 무엇이 사라지는가"가 방향을 결정**하며, 이는 다음 트랙([`../single-cell-multiomics/`](../single-cell-multiomics))의
최종 결론("gap을 결정하는 것은 정보량이 아니라 두 모달리티가 공유하는 학습 가능한 구조의 비율")과도 같은
축 위에 있다.

## 참고 문헌 (두 서브프로젝트 공통)

| 문헌 | 역할 |
|---|---|
| Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021 | 대조학습 dual-encoder 백본 |
| Liang et al., *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*, NeurIPS 2022 | $\Delta_{gap}$ 정의, embedding shift 개입, loss landscape probing 방법론 |
| Schrödi et al., *Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models*, ICLR 2025 (Oral) | 이 프로젝트 전체의 핵심 가설(H1)의 이론적 출처 |
| Levi & Gilboa, *Double-Ellipsoid Geometry of CLIP*, ICML 2025 | `conformity-groundedness/`가 확장한 conformity(전형성) 개념의 출처 |
| Fahim et al. (2024), *It's Not a Modality Gap* | `semart-pipeline/`이 채택한 linear separability 지표의 출처 |

각 서브프로젝트의 정확한 실험 설계·재현 방법·전체 수치는 해당 디렉토리의 README와 `docs/`를 참고할 것.
