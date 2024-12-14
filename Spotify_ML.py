import pandas as pd  # Для работы с таблицами
import matplotlib.pyplot as plt  # Для создания графиков
import seaborn as sns 

sns.set_theme(style="whitegrid")  

# Загрузка данных из файла
df = pd.read_csv("/Users/aleksandravdeev/Documents/dataset.csv")  # Укажи свой путь к файлу

# Проверяем гипотезу, чего в Spotify больше, популярных треков или же непопулярных
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True, color='blue')
plt.title("Распределение кол-ва треков по популярности")
plt.xlabel("Популярность")
plt.ylabel("Количество треков")
plt.show()
# Вывод: в Spotify популярных треков гораздо меньше нежели непопулярных

# Сравниваем, зависит ли популярность треков от их танцевальность
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='danceability', y='popularity', alpha=0.3, color='purple')
plt.title("Зависимость популярности от танцевальности")
plt.xlabel("Танцевальность")
plt.ylabel("Популярность") # 
plt.show()
# Вывод: популярность треков никак не зависит от их "танцевальности"

# Сравниваем, зависит ли популярность треков от их темпа
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='tempo', y='popularity', alpha=0.5, color='green')
plt.title("Зависимость популярности от темпа трека")
plt.xlabel("Темп (BPM)")
plt.ylabel("Популярность")
plt.show()
# Вывод: популярность треков никак не зависит от их "темпа"

# Смотрим насколько громкие треки в Spotify
plt.figure(figsize=(10, 6))
sns.histplot(df['loudness'], bins=30, kde=True, color="green")
plt.title("Распределение громкости треков")
plt.xlabel("Громкость (dB)")
plt.ylabel("Количество треков")
plt.show()
# Вывод: Больше всего треков имееют громкость в -16

# проверям средний темп по разным жанрам
top_genres = df['track_genre'].value_counts().head(10).index
genre_tempo = df[df['track_genre'].isin(top_genres)].groupby('track_genre')['tempo'].mean().sort_values()

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_tempo.index, y=genre_tempo.values, palette="muted")
plt.title("Средний темп треков в разных жанрах")
plt.xlabel("Жанр")
plt.ylabel("Темп (BPM)")
plt.xticks(rotation=45)
plt.show()
# Вывод: самый высокий темп у power-pop, а самый низкий у оперы

# Топ-10 артистов по средней популярности
top_artists = df.groupby("artists")['popularity'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_artists.values, y=top_artists.index, palette="rocket")
plt.title("Топ-10 артистов по средней популярности треков")
plt.xlabel("Средняя популярность")
plt.ylabel("Артист")
plt.show()
# Вывод: Первое место занял - Sam Smith;Kim Petras

# Смотрим, насколько процентов треки записанны в живую
plt.figure(figsize=(10, 6))
sns.histplot(df['liveness'], bins=30, kde=True, color="pink")
plt.title("Распределение натуральности треков")
plt.xlabel("Натуральность (Liveness)")
plt.ylabel("Количество треков")
plt.show()
# Вывод: Во всём Spotify в большенстве своём преобладают треки, имеющие меньше 20% натурльности

# Проверяем, зависит ли позитивность песни на её популярность
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='valence', y='popularity', color="cyan", alpha=0.5)
plt.title("Влияние веселости трека на популярность")
plt.xlabel("Веселость (Valence)")
plt.ylabel("Популярность")
plt.show()
# Вывод: Так же как и с 'танцевальностью' популярность песни не зависит от жанра

# Корреляционная матрица
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr(numeric_only=True)  # Вычисляем корреляцию для числовых данных
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляция между числовыми признаками")
plt.show()
# теперь мы можем искать на графике 

# Корреляция popularity с ключевыми признаками
popularity_corr = df.corr(numeric_only=True)['popularity'].sort_values(ascending=False)
print("Корреляция popularity с другими признаками:\n", popularity_corr)
# Вывод: Можем сразу увидеть численную зависимость popularity с другими признаками в консоли

# Проверяем есть ли зависимость популярности трека с 'танцевальностью' и темпом трека одновременно
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='danceability', y='tempo', size='popularity', hue='popularity', palette="coolwarm", alpha=0.6)
plt.title("Влияние танцевальности и темпа на популярность треков")
plt.xlabel("Танцевальность")
plt.ylabel("Темп (BPM)")
plt.legend(title="Популярность", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# Вывод: Популярность трека никак не зависит от танцевальности и темпа одновременно

