import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import numpy as np
import umap
import seaborn as sns

# Загружаем стоп-слова для русского языка из библиотеки nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("russian"))
stemmer = SnowballStemmer("russian")


# Функция предобработки текста
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление знаков пунктуации и специальных символов
    text = re.sub(r'[^\w\s]', '', text)
    # Токенизация
    tokens = word_tokenize(text)
    # Удаление стоп-слов
    tokens = [word for word in tokens if word not in stop_words]
    # Стемминг
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Объединение токенов обратно в текст
    preprocessed_text = ' '.join(stemmed_tokens)
    print('---------')
    print(preprocessed_text)
    return preprocessed_text


# Чтение текстового файла с новостями на русском языке
with open('lab4_source.txt', 'r', encoding='utf-8') as file:
    news = file.readlines()

# Предобработка текстов
preprocessed_news = [preprocess_text(news_) for news_ in news]


def find_common_phrases(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    common = set(words1) & set(words2)
    phrases = []
    print(common)
    for word in common:
        if text1.count(word) > 1 and text2.count(word) > 1:
            phrases.append(word)
    return phrases[:]


clusters = {}  # Словарь для хранения кластеров

for i in range(len(preprocessed_news)):
    cluster_found = False  # Флаг, указывающий на то, был ли найден кластер для текущего документа
    for cluster_name, cluster_data in clusters.items():
        for data in cluster_data:
            similarity = fuzz.token_set_ratio(preprocessed_news[i], data[0])  # Рассчитываем сходство текущего документа с документами в существующих кластерах
            if similarity > 70:  # Порог сходства для считается дубликатом
                common_phrases = find_common_phrases(preprocessed_news[i], data[0])  # Находим общие слова
                cluster_data.append((preprocessed_news[i], similarity))  # Добавляем текущий документ в существующий кластер
                cluster_found = True  # Устанавливаем флаг, что кластер найден
                break  # Прекращаем проверку текущего кластера
        if cluster_found:
            break  # Прекращаем проверку остальных кластеров, если уже найден кластер для текущего документа

    if not cluster_found:  # Если для текущего документа не был найден подходящий кластер
        clusters[f"Cluster {len(clusters) + 1}"] = [(preprocessed_news[i], 100)]  # Создаем новый кластер для текущего документа с полным сходством

# Вывод информации о добавленных кластерах в консоль
for cluster_name, cluster_data in clusters.items():
    print(f"{cluster_name}:")
    for data in cluster_data:
        print(f"Data: {data}")
    print("---------------------------------------")

# Рассчет сходства между всеми парами документов
similarity_matrix = np.zeros((len(preprocessed_news), len(preprocessed_news)))
for i in range(len(preprocessed_news)):
    for j in range(len(preprocessed_news)):
        similarity = fuzz.token_set_ratio(preprocessed_news[i], preprocessed_news[j])
        similarity_matrix[i][j] = similarity

print(similarity_matrix)

# Compute the 2-dimensional embedding of the data using UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(similarity_matrix)

# Визуализация
plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
for i, txt in enumerate(range(len(similarity_matrix))):
    plt.annotate(f"Doc {i}", (embedding[i, 0], embedding[i, 1]))

plt.title('Визуализация UMAP')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=[f"Doc {i}" for i in range(len(similarity_matrix))], yticklabels=[f"Doc {i}" for i in range(len(similarity_matrix))])
plt.title('Тепловая карта матрицы сходства')
plt.show()
