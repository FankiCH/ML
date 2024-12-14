from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from xgboost import XGBRegressor

# Загрузка данных
df = pd.read_csv("/Users/aleksandravdeev/Documents/dataset.csv")

# Создание новых признаков
df['tempo_bin'] = pd.cut(df['tempo'], bins=[0, 90, 130, 300], labels=['low', 'medium', 'high'])
df['duration_minutes'] = df['duration_ms'] / 60000
df['speechiness_bin'] = df['speechiness'].apply(lambda x: 'high' if x > 0.33 else 'low')
df['energy_to_dance_ratio'] = df['energy'] / (df['danceability'] + 1e-5)

# Подготовка данных для модели
df_encoded = pd.get_dummies(df[['danceability', 'energy', 'tempo_bin', 'loudness', 'duration_minutes', 
                                'speechiness_bin', 'energy_to_dance_ratio', 'popularity']], drop_first=True)

X = df_encoded.drop(columns=['popularity'])
y = df_encoded['popularity']

# Линейная регрессия
linear_model = LinearRegression()
linear_scores = cross_val_score(linear_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Средний RMSE
print("Линейная регрессия - средний RMSE:", abs(linear_scores.mean()) ** 0.5)


# Деревья
dt_model = DecisionTreeRegressor(random_state=42)
dt_scores = cross_val_score(dt_model, X, y, cv=5, scoring='neg_mean_squared_error')

print("Decision Tree - средний RMSE:", abs(dt_scores.mean()) ** 0.5)

# Модификации градиентного бустинга
xgb_model = XGBRegressor(random_state=42)
xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')

print("XGBoost - средний RMSE:", abs(xgb_scores.mean()) ** 0.5)

# Создание нейронной сети
model_nn = Sequential()
model_nn.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(1))  # Выходной слой

model_nn.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели с кросс-валидацией
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model_nn.fit(X_train, y_train, epochs=50, verbose=0, batch_size=32)
    y_pred = model_nn.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_scores.append(rmse)

print("Нейронная сеть - средний RMSE:", sum(rmse_scores) / len(rmse_scores))