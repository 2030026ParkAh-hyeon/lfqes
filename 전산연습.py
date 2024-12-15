# data 수를 매년 1월 1일에 해당하는 값으로 불러들여 30개로 설정
# Linear regression과 Polynomial regression 비교
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# month 단위로 나뉘는 시간 data를 각 연도별로 하나씩 가져와서 규모를 축소한다(Linear Regression에 더 적합하게 하도록 데이터 간소화)
years = np.arange(1994, 2025)  # 1994년부터 2024년까지 30년을 범위로 잡는다.
print(years)

# 실제 금값 데이터를 매년 12월 1일의 데이터로 30개 가져온다.
real_prices = np.array([
    799.48, 796, 733.84, 564.48, 561.48, 530.76, 492.74, 492.39,
    578.87, 695.26, 732.97, 817.69, 983.78, 1206.41, 1305.4,
    1588.84, 2022.51, 2140.34, 2277.41, 1630.89, 1620.86, 1426.38,
    1505.27, 1659.52, 1609.75, 1870.24, 2294.97, 2069.98, 1939.26, 2122.75, 2655.33
])

# 데이터를 연산이 가능한 형태로 변환한다.
data = pd.DataFrame({'Year': years, 'Real_Gold_Price': real_prices})

# 목표 변수는 2025년의 금값이므로, 시간을 독립 변수, 실제 금값을 종속 변수로 한다.
# 그러나 입력된 데이터 값이 1차원이므로 reshape하여 모델의 2차원 배열을 맞춰줘야 함-> 수정 요청
X = data['Year'].values.reshape(-1,1) # X를 2차원 변수로 만들어야 함
y = data['Real_Gold_Price'].values 

# 먼저 Linear regression을 학습
linear_model = LinearRegression()
linear_model.fit(X, y)

# 학습한 값으로 예측한다.
y_pred_linear = linear_model.predict(X)

# Linear Regression의 MSE를 계산해본다
# MSE 계산 방법과 코드 요청 -> 아래와 같이 제시 (y와 y_pred로 오차율을 측정하는 것을 볼 수 있음)
mse_linear = mean_squared_error(y, y_pred_linear)
print(f"Linear Regression MES : {mse_linear}")

# Polynomial Regression에 대해 학습하고 예측, MSE를 구해보기
degree = 3 # 여러 degree 중 가장 적합한 degree가 3이었으므로
poly = PolynomialFeatures(degree=degree) # 바뀌는 Degree에 유동적으로 적용될 수 있도록
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

mse_poly = mean_squared_error(y, y_pred_poly)
print(f"Polynomial Regression (Degree={degree}) MSE: {mse_poly}")

# 실제 금값과 각 모델의 예측값을 함께 나타낸 시각화 그래프
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='red', label='Real Prices')
plt.scatter(X, y_pred_linear, color='green', label=f'Linear Regression (MSE={mse_linear:.2f}))')
plt.scatter(X, y_pred_poly, color='blue', label=f'Polynomial Regression (Degree={degree}, MSE={mse_poly:.2f}))')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Gold Price (USD per oz)')
plt.title('Gold Price Prediction with Linear and Polynomial Regression')
plt.show()

# 2025년 금값 예측
X_2025 = np.array([[2025]])
y_pred_2025_linear = linear_model.predict(X_2025)
y_pred_2025_poly = poly_model.predict(poly.transform(X_2025))

print(f"2025년 Linear Regression_Gold Price 예측: {y_pred_2025_linear[0]:.2f}")
print(f"2025년 Polynomial Regression_Gold Price 예측: {y_pred_2025_poly[0]:.2f}")



# 다양한 data로 Polynomial Regression 모델 적용
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# GitHub에서 raw 형태로 CSV 파일을 다운로드 (어디서든 csv 파일을 같은 디렉토리에 넣지 않고도 바로 다운로드하여 코드에 실행할 수 있도록 조치)
url = 'https://raw.githubusercontent.com/username/repository/main/data/gold_prices.csv'  # GitHub raw URL로 변경
gold_data = pd.read_csv(url)  

gold_data = gold_data.sort_values(by='date', ascending=True) # 데이터가 내림차순이므로 보기 편하게 오름차순으로 변경

# 데이터 확인
print(gold_data.head())


# 저장했던 gold_prices.scv 파일을 1994년 데이터부터 불러온다
gold_data = pd.read_csv(r'C:\Users\p0974\OneDrive\바탕 화면\2024-2 전산물리학\전산 기말\gold_prices.csv', skiprows=10)

# 날짜 부분의 date column이 2024-12-01 형식으로 되어있는데, 이는 연산이 불가하므로 datetime 형식으로 바꾼다.
# 형식을 바꿔 덮어씌운다.
gold_data['date'] = pd.to_datetime(gold_data['date'])

# data가 month 단위로 되어있으나 연단위로 추이를 보기 위해서 year 기준으로 다시 정리할 수 있냐고 요청->year column을 만든다.
gold_data['Year'] = gold_data['date'].dt.year


# 금값 상승이나 하락에 연관되는 경제적 이슈를 반영하기 위해 세 가지 사건(한국의 IMF, 코로나 팬데믹, 우크라이나 전쟁)을 제시하고
# 어떻게 예외로 반영할 것인지 요청-> 이진 변수를 사용하여 해당 사건이 일어난 연도를 범위 내에 설정한다.
gold_data['IMF_Crisis'] = ((gold_data['Year'] >= 1997) & (gold_data['Year'] <= 1998)).astype(int)
# IMF 사건이 일어난 1997년부터 1998년에 해당하는 column은 모두 1로 출력되도록 이진 변수 설정
gold_data['COVID_Pandemic'] = ((gold_data['Year'] >= 2020) & (gold_data['date'].dt.month >=3)).astype(int) # 2023년 3월 이후 지속됨
gold_data['Ukraine_War'] = ((gold_data['Year'] >= 2022) & (gold_data['date'].dt.month >=2)).astype(int) # 2022년 2월부터, 구체적인 시기 제시
# 두 사건은 구체적인 시기 제시, 오래 지속되는 경향 반영


# 최종 목표는 Polynomial regression을 사용하여 2025년의 금값을 예측하는 것이다.
# 목표 변수를 2025년의 금값으로, 독립 변수는 날짜와 경제적 이슈 3가지로, 종속 변수는 이에 따른 금값으로 설정한다.
X = gold_data[['Year', 'IMF_Crisis', 'COVID_Pandemic', 'Ukraine_War']]
y = gold_data['real'] # real은 csv 파일 내의 실제 당시 금값


# 여기에서 데이터는 training에 사용되는 데이터와 test 데이터로 나눈다.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


# 비선형적인 데이터를 머신러닝 하기 위해 Polynomial regression model을 가져온다.
from sklearn.preprocessing import PolynomialFeatures
# Linear regression 모델이 이후 다항식 변환 후에 상수항을 중복학습하여 예측이 빗나갈 수 있기에
# include_bias = False 항을 추가
poly = PolynomialFeatures(degree = 3, include_bias = False) # 학습 후 degree를 조정하여 가장 MSE가 작은 degree 찾기
# Polynomial 모델로 학습한다.
X_poly = poly.fit_transform(X) # 여러 X 변수들을 위에서 제시한 polynomial regression(degree=2)에 적용시킨 것이다
# poly에 대해 정해준 뒤 구체적으로 LinearRegression 모델과 PolynomialFeature를 어떻게 학습하는지 코드요청
# ->아래 3줄로 학습 코드 출력
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=7)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)


# 학습된 모델로 예측값을 생성하기 / train과 test를 split 했으므로 각 구하고 모든 데이터에 대해서도 예측

y_train_pred = poly_model.predict(X_train_poly)
y_test_pred = poly_model.predict(X_test_poly)
y_pred_all = poly_model.predict(poly.transform(X))

# 학습이 잘 되었는지 MSE로 모델의 오차율 확인과 위에서 학습한 예측값을 비교하는 코드 요청 (MSE가 작을수록 예측이 잘 된 것)
from sklearn.metrics import mean_squared_error
y_pred_poly = poly_model.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f"Polynomial Regression Mean squared Error: {mse_poly}")

comparison_poly = pd.DataFrame({'Actual': y_test, 'Predicted (Polynomial)':y_pred_poly})
print("다항 회귀 예측값 vs 실제값")
print(comparison_poly.head())

train_comparison = pd.DataFrame({'Actual':y_train, 'Predicted (Train)': y_train_pred})
test_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

print("Train data 예측값 vs 실제값")
print(train_comparison.head())

print("Test data 예측값 vs 실제값")
print(test_comparison.head())

# 2025년 금값 예측 시에 X 변수에 넣었던 이진 변수 설정
imf_2025 = 0
covid_2025 = 0
ukraine_2025 = 0
# 2025년에는 모든 경제적 이슈 없다는 가정 하에 예측 진행 하도록 0으로 입력
X_2025 = pd.DataFrame({'Year': [2025], 'IMF_Crisis': [imf_2025], 'COVID_Pandemic': [covid_2025], 'Ukraine_War': [ukraine_2025]})
X_2025_poly = poly.transform(X_2025)
# 위와 같이 Polynomial regression을 이용하여 예측하기 (위의 단계에 _2025를 삽입하여 반복)

pred_2025_poly = poly_model.predict(X_2025_poly)

print(f"2025년 gold price 예측: {pred_2025_poly[0]}") # 이진 변수 0

# 예측한 금값 추이와 실제 금값 추이를 시각화하여 비교
plt.figure()
plt.plot(gold_data['Year'], gold_data['real'], label='Actual Gold Price', color='green')
plt.plot(gold_data['Year'], y_pred_all, label='Predicted Gold Price', color='red')
plt.title('Gold price Prediction with Polynomial Regression')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Gold price (USD per oz)')
plt.show()


# Self Evaluation
# A0
# 지금까지 배운 코드들을 완전히 자유롭게 다루지 못합니다.
# 그러나 처음 시작할 때 Hello, world! 코드도 뭔지 잘 몰랐던 상태에서
# 이제는 chat gpt를 이용하여 원하는 방향으로 코드를 작성하고 수정할 수 있습니다.
# 간단한 오류의 원인을 에러 코드를 보고 해석하여 원인을 파악할 수 있습니다.
# 코드를 짜면서 이해하고 적절한 모델을 찾기 위해 여러 자료를 찾아보면서
# 상황에 필요한 데이터를 구하고 가공할 수 있게 되었으며, 처음부터 다시 살펴나가면서 머신 러닝에 대한 이해가 깊어졌습니다.
# 따라서 A0가 적합하다고 생각합니다.

