# 영화 추천을 위한 Latent Factor Model

이 레포지토리는 **Latent Factor Model**을 활용한 개인화 영화 추천 시스템을 구현하고 평가하기 위한 세 개의 Python 스크립트를 포함하고 있습니다.  
이 시스템은 **협업 필터링** 기반으로 행렬 분해를 통해 사용자가 평가하지 않은 영화의 평점을 예측합니다.

---

## 구성

### 1. `ratings_prediction.py`
- **목적**: 
  Stochastic Gradient Descent(SGD)을 사용하여 행렬 분해 기반의 **Latent Factor Model**을 학습하고, 사용자-영화 평점을 예측합니다.
- **주요 기능**:
  - MovieLens 데이터셋을 활용하여 사용자와 영화 간의 잠재 요인을 학습.
  - 모든 사용자-영화 조합에 대해 예측 평점을 계산하고 `predicted_ratings.csv` 파일로 저장.

---

### 2. `LatentFactorModel.py`
- **목적**: 
  예측된 평점을 기반으로 특정 사용자에게 영화 추천 리스트를 생성합니다.
- **주요 기능**:
  - `predicted_ratings.csv`를 로드하여 예측된 평점 데이터를 활용.
  - 사용자가 아직 평가하지 않은 영화 중 높은 평점을 받은 영화를 추천.

---

### 3. `EvaluationCriterion_LFM.py`
- **목적**: 
  **Latent Factor Model**의 추천 성능을 평가합니다.
- **주요 기능**:
  - 특정 사용자(`base_user_id`)의 실제 평점과 예측 평점 간의 절대 차이를 계산.
  - 절대 차이가 1.0 이하인 영화 비율을 통해 추천 모델의 정확도를 평가.
