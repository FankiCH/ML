from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

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

# Инициализация моделей
models = {
    'Линейная регрессия': LinearRegression(),
    'Дерево решений': DecisionTreeRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'Нейронная сеть': Sequential([
        Input(shape=(X.shape[1],)),  # Используем Input для указания размера входных данных
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Выходной слой
    ])
}

# Компиляция нейронной сети
models['Нейронная сеть'].compile(loss='mean_squared_error', optimizer='adam')

# Кросс-валидация и сбор результатов
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# Для каждой модели выполняем кросс-валидацию
for name, model in models.items():
    if name == 'Нейронная сеть':  # Для нейронной сети вручную вычислим RMSE
        rmse_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train, epochs=50, verbose=0, batch_size=32)
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmse_scores.append(rmse)
        results[name] = np.mean(rmse_scores)  # Средний RMSE для нейронной сети
    else:
        # Для других моделей используем cross_val_score
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        results[name] = np.mean(np.abs(scores)) ** 0.5  # Средний RMSE

# Выводим результаты для всех моделей
for name, rmse in results.items():
    print(f"{name} - Средний RMSE: {rmse}")

# Выбор лучшей модели
best_model_name = min(results, key=results.get)
print(f"\nЛучшая модель: {best_model_name} с RMSE: {results[best_model_name]}")

'''
Мой вывод:
Линейная регрессия - Средний RMSE: 22.22660127694841
Дерево решений - Средний RMSE: 21.408896117865254
XGBoost - Средний RMSE: 20.116024984588353
Нейронная сеть - Средний RMSE: 21.60486880998186

Лучшая модель: XGBoost с RMSE: 20.116024984588353
'''
