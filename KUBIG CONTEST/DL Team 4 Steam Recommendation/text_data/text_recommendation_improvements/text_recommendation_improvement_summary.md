# Text Recommendation Improvement Experiments

## 실험 설정

- Train: date < 2022-04-01
- Validation: 2022-04-01 <= date < 2022-07-01
- Test: 2022-07-01 <= date < 2023-01-01
- 평가 split: test
- Train interaction >= 5 user를 seed 42로 sample
- Sampled train users: 100,000
- Base eval users before positive-history filtering: 5,000

## 비교 결과 (@10)

| experiment | method | min_positive_history | alpha | n_eval_users | Recall@10 | NDCG@10 | Coverage | Cold Recall@10 | Avg Train Popularity | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| popularity_ref_pos>=5 | popularity_reference | 5 | nan | 4534 | 0.021885 | 0.011887 | 0.000393 | 0.000000 | 195287.008337 | 0.033776 |
| popularity_ref_pos>=3 | popularity_reference | 3 | nan | 4938 | 0.021529 | 0.011423 | 0.000393 | 0.000000 | 195366.604212 | 0.032956 |
| popularity_ref_pos>=1 | popularity_reference | 1 | nan | 5000 | 0.021262 | 0.011281 | 0.000393 | 0.000000 | 195382.703140 | 0.032547 |
| hours_weighted_pos>=1 | hours_weighted_mean | 1 | nan | 5000 | 0.007934 | 0.005859 | 0.140588 | 0.005256 | 4443.669560 | 0.020454 |
| hours_weighted_pos>=3 | hours_weighted_mean | 3 | nan | 4938 | 0.007831 | 0.005804 | 0.137207 | 0.005329 | 4452.004982 | 0.020337 |
| pos_minus_0.5_neg_pos>=1 | pos_minus_neg | 1 | 0.500000 | 5000 | 0.007591 | 0.005117 | 0.197063 | 0.005229 | 3825.468860 | 0.019907 |
| hours_weighted_pos>=5 | hours_weighted_mean | 5 | nan | 4534 | 0.007693 | 0.005566 | 0.124115 | 0.005062 | 4339.140406 | 0.019562 |
| pos_minus_0.5_neg_pos>=3 | pos_minus_neg | 3 | 0.500000 | 4938 | 0.007281 | 0.004878 | 0.191952 | 0.004863 | 3828.741576 | 0.018941 |
| pos_minus_0.2_neg_pos>=1 | pos_minus_neg | 1 | 0.200000 | 5000 | 0.007130 | 0.004846 | 0.147311 | 0.004796 | 3596.640780 | 0.018246 |
| simple_mean_pos>=1 | simple_mean | 1 | nan | 5000 | 0.007153 | 0.004811 | 0.131015 | 0.004859 | 3421.391580 | 0.018133 |
| pos_minus_0.2_neg_pos>=3 | pos_minus_neg | 3 | 0.200000 | 4938 | 0.006815 | 0.004577 | 0.142495 | 0.004424 | 3596.112049 | 0.017241 |
| simple_mean_pos>=3 | simple_mean | 3 | nan | 4938 | 0.006838 | 0.004642 | 0.126926 | 0.004488 | 3415.993236 | 0.017238 |
| simple_mean_pos>=5 | simple_mean | 5 | nan | 4534 | 0.006868 | 0.004570 | 0.113422 | 0.004468 | 3215.124151 | 0.017040 |
| pos_minus_0.5_neg_pos>=5 | pos_minus_neg | 5 | 0.500000 | 4534 | 0.006640 | 0.004599 | 0.169248 | 0.004081 | 3704.309660 | 0.017013 |
| pos_minus_0.2_neg_pos>=5 | pos_minus_neg | 5 | 0.200000 | 4534 | 0.006586 | 0.004455 | 0.125177 | 0.004081 | 3414.806242 | 0.016374 |

## 자동 선택 best setting

- experiment: `hours_weighted_pos>=1`
- method: `hours_weighted_mean`
- min_positive_history: 1
- alpha: nan
- selection_score: 0.020454

## Baseline 대비 변화

- Recall@10: 0.007153 -> 0.007934 (+0.000781)
- NDCG@10: 0.004811 -> 0.005859 (+0.001048)
- Coverage: 0.131015 -> 0.140588 (+0.009573)
- Cold Recall@10: 0.004859 -> 0.005256 (+0.000397)
- Avg Train Popularity: 3421.39 -> 4443.67 (+1022.28)

## 해석

best 설정은 정확도 지표에서 baseline보다 개선됐다.

log1p(hours) weighting은 오래 플레이한 positive item을 user vector에 더 강하게 반영한다. 성능이 좋아졌다면 단순 추천 여부보다 playtime이 취향 강도를 더 잘 표현했다는 뜻이고, 나빠졌다면 hours가 취향보다 인기작/장시간 플레이 장르에 user vector를 과하게 끌고 갔을 가능성이 있다.

positive_mean - alpha * negative_mean은 비추천한 게임 방향을 user vector에서 빼는 방식이다. 개선되면 dislike signal이 의미 있었던 것이고, 악화되면 negative interaction이 적거나 noisy해서 좋아하는 장르와 가까운 방향까지 함께 깎았을 수 있다.

positive history threshold를 높이면 user vector는 안정적이지만 평가 user가 줄어든다. 따라서 threshold별 결과는 n_eval_users와 함께 해석해야 한다.