import logging
import multiprocessing
import re
import sqlite3
import gensim
from gensim.models import Word2Vec


# функція для обробки тексту
def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

# відкриваємо наш датасет
conn = sqlite3.connect('mysqlite3.db')
c = conn.cursor()

# створюємо файл в який запишем опрацьований текст готовий до тренування
with open('data/tweets.txt', 'w', encoding='utf-8') as f:
    for row in c.execute('SELECT ttext FROM sentiment'):
        if row[0]:
            tweet = preprocess_text(row[0])
            print(tweet, file=f)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# тренуємо модель, використовуємо gensim для векторного подання слів
data = gensim.models.word2vec.LineSentence('daat/tweets.txt')
model = Word2Vec(data, size=200, window=5, min_count=3, workers=multiprocessing.cpu_count())

# зберігаємо модель
model.save("w2v/tweets_model.w2v")
