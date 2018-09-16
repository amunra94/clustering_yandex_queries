"""
Простая кластеризация запросов по вхождению слова.
В зависимости от условий задачи необходимо определить количество кластеров.
В своей реализации применил метод частотного преобразования слов.
В качестве кластеризатора выбрал метод Kmeans, он отлично подходит для анализа большого количества данных
и несложно интерпретируется.
В данном случае выбрал 40 кластеров для разделения:
    Возможно другое количество, которое определяется исходя из условий.В лексему можно брать пары слов или более
    для нахождения паттернов.

    Для определния прогнозируемого числа кластеров  можно применить сети Кохонена, но необходимо знать условия задачи.

    В данном случае кластеры получились по вхождению слова или пары слов:
        - слушать
        - скачать
        - смотреть
        - официальный сайт
        - сколько
        - кластер с английскими словами
        - и т.д.

    Результаты можно увидеть в выходных файлах названных по номеру кластеру.



"""


import requests
import re
import time
import pickle
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
import collections
import nltk

# Добавление стоп-слов для очистки данных
STOP_WORDS = nltk.corpus.stopwords.words('russian')
STOP_WORDS.extend(['это','1','2','3','4','5','6','7','8','9','0'])

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'
}
SIZE_DATA = 10000
TIME_SLEEP = 1
NAME_FDATA = 'data.pkl'

def get_data():
    """
    Получение данных с прямого эфира
    :return:
    """
    queries = []
    while (True):
        if len(queries) >= SIZE_DATA:
            break
        page = requests.get('https://export.yandex.ru/last/last20x.xml', headers=HEADERS)
        queries += get_text(page.content)
        time.sleep(TIME_SLEEP)
    return queries

def get_text(content):
    """
    Функция получения текста из xml
    :param content:
    :return:
    """
    strings = content.decode('utf-8').split('<item found=')
    queries = []
    for query in strings:
        query_text = re.findall(r'>.*<', query)
        if len(query_text) != 0:
            queries.append(query_text[0][1:-1])
    return queries

def tokenizer(text):
    """
    Токенизатор
    :param text: входной текст
    :return: список слов
    """
    text = re.sub('<[^>]*>&#', '', text)
    text = text.lower()
    tokenized = [word for word in text.split() if word not in STOP_WORDS and word.isnumeric() is not True]
    return tokenized



if __name__ == '__main__':
    # vect_hash = HashingVectorizer(decode_error='ignore',
    #                           n_features=2 ** 21,
    #                           preprocessor=None,
    #                           tokenizer=tokenizer)

    # Загрузка данных
    try:
        with open(NAME_FDATA, 'rb') as f_data:
            data = pickle.load(f_data)
    except IOError:
        data = get_data()
        with open(NAME_FDATA, 'wb') as f_data:
            pickle.dump(data, f_data)

    # Очистка данных
    for i in range(len(data)):
        tokenized = ' '.join(tokenizer(data[i]))
        data[i] = tokenized

    # Техника частотного преобразования слов
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None,
                            ngram_range=(1, 1),
                            stop_words=None,
                            tokenizer=tokenizer)
    # В качестве метода для кластеризации взял KMeans
    text_cluster = Pipeline([('tfidf', tfidf),
                             ('cluster', KMeans(n_clusters=40, random_state=0)),
                             ])

    text_cluster.fit(data)


    cluster = text_cluster.steps[1][1]
    labels = cluster.labels_

    # Группировка по маркерам
    labeled_data = collections.defaultdict(list)
    for raw, label in zip(data, labels):
            labeled_data[label].append(raw+'\n')

    # Вывод данных 40 кластеров в текстовые файлы
    for key in labeled_data:
        with open(str(key)+'.txt', 'w') as fout:
            fout.writelines(labeled_data[key])



