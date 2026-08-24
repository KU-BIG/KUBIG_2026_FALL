# 26_2_Contest — Steam 멀티모달 게임 추천

Steam 게임의 사용자 interaction과 Text, Image, Tabular 정보를 결합한 추천 시스템
프로젝트입니다. 현재는 기존 사용자·신규 사용자 추천, 다양성 reranking, 이미지 카드 UI까지
포함한 Streamlit MVP가 동작합니다.

## 현재 최종 결과

- 기준 카탈로그: Steam 게임 50,872개
- 최종 known-user 모델: **MF 40% + Multimodal 60%**
- Multimodal game embedding: MiniLM Text + CLIP Image + SVD Tabular → 64D
- 3-seed 평균: Recall@10 `0.8165`, NDCG@10 `0.5609`
- 신규 사용자: 선호 태그·좋아하는 게임 입력 지원
- UI: Steam 이미지 카드, 모델 비교 탭, 다양성 조절, CSV 다운로드

## 바로 실행

```bash
pip install -r recommendation_mvp/requirements.txt
streamlit run recommendation_mvp/app.py
```

Python `3.12`를 권장합니다. 자세한 사용법은
[`recommendation_mvp/README.md`](./recommendation_mvp/README.md)를 확인하세요.

## 폴더 안내

| 폴더 | 내용 | 먼저 읽을 문서 |
|---|---|---|
| [`recommendation_mvp/`](./recommendation_mvp/) | 최종 Streamlit 서비스, 배포 데이터, 모델 artifact | [`README.md`](./recommendation_mvp/README.md) |
| [`mvp_recommendation/`](./mvp_recommendation/) | BPR, inference, cold-start, MMR Python 모듈 | 추천 서비스 README 참고 |
| [`game_fusion/`](./game_fusion/) | Text + Image + Tabular 64D game embedding | [`game_fusion_README.md`](./game_fusion/game_fusion_README.md) |
| [`game_fusion/downstream_evaluation/`](./game_fusion/downstream_evaluation/) | 실제 interaction 기반 3-seed 평가 | [`README.md`](./game_fusion/downstream_evaluation/README.md) |
| [`Data_process/`](./Data_process/) | Kaggle base + HF metadata 보완 데이터와 전처리 | [`README.md`](./Data_process/README.md) |
| [`text_data/`](./text_data/) | MiniLM Text embedding과 TextTower | 폴더 내 README |
| [`tabular_embedding/`](./tabular_embedding/) | 정형 데이터 embedding과 TabularTower | 폴더 내 README |
| [`scripts/`](./scripts/) | 학습, 평가, 추천, 검증 실행 스크립트 | 각 스크립트 `--help` |

## 팀원이 알아야 할 모델 선택

최종 서비스는 `game_fusion/emb_game_concat_64.npy/.csv`를 사용합니다.
`emb_game_finetuned_64`와 `emb_game_partial_fusion_tuned_64`는 synthetic interaction으로 만든
smoke-test 결과이므로 배포 모델로 사용하지 않습니다. 실제 interaction downstream 평가에서
frozen concat의 추천 성능이 더 높았습니다.

## Git LFS

대용량 모델과 embedding 일부는 Git LFS로 관리합니다. clone 후 모델 파일이 정상적으로
받아지지 않았다면 다음을 실행하세요.

```bash
git lfs install
git lfs pull
```

## 현재 단계와 다음 작업

현재는 **멀티모달 추천 MVP 구현 및 sampled-candidate 오프라인 검증 완료** 단계입니다.
다음 우선순위는 전체 50,872개 게임 대상 full-catalog 평가, 팀원 사용성 테스트, 실제
클릭·찜·플레이 피드백 수집입니다.
