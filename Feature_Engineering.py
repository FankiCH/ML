import pandas as pd  # Для работы с таблицами
import matplotlib.pyplot as plt  # Для создания графиков
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("/Users/aleksandravdeev/Documents/dataset.csv")

# Создание новых признаков
df['energy_to_dance_ratio'] = df['energy'] / (df['danceability'] + 1e-5)  # Добавляем небольшое число для избежания деления на 0
df['tempo_bin'] = pd.cut(df['tempo'], bins=[0, 80, 120, 180, 300], labels=['slow', 'medium', 'fast', 'very fast'])
df['duration_minutes'] = df['duration_ms'] / 60000
df['speechiness_bin'] = pd.cut(df['speechiness'], bins=[0, 0.33, 0.66, 1.0], labels=['low', 'medium', 'high'])

# Проверка новых колонок
print(df[['energy_to_dance_ratio', 'tempo_bin', 'duration_minutes', 'speechiness_bin']].head())

# Корреляция новых и старых признаков с таргетом
correlations = df.corr(numeric_only=True)['popularity'].sort_values(ascending=False)
print("Корреляция признаков с популярностью:\n", correlations) # Получаем список признаков, отсортированный по степени их корреляции с таргетом. Оцениваем, добавили ли новые признаки дополнительную ценность.


df_encoded = pd.get_dummies(df[['danceability', 'energy', 'tempo_bin', 'loudness', 'duration_minutes', 'speechiness_bin', 
                                'energy_to_dance_ratio', 'popularity']], drop_first=True)

X = df_encoded.drop(columns=['popularity'])
y = df_encoded['popularity']

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Важность признаков
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Отображение важности
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="coolwarm")
plt.title("Feature Importance на основе Decision Tree")
plt.show()
# Вывод: Этот график показывает, какие признаки больше всего влияют на предсказание популярности.