# Аналіз тональності текстів за допомогою згорткової нейронної мережі
## Вступ
Уявіть, що у вас є абзац тексту. Чи можна зрозуміти, яку емоцію несе цей текст: радість, смуток, гнів? Спростимо собі задачу і будемо класифікувати емоцію як позитивну нейтральну, або як негативну, без уточнень. Є багато способів вирішувати такі завдання, і один з них - згорткові нейронні мережі (Convolutional Neural Networks). CNN спочатку були розроблені для обробки зображень, однак вони успішно справляються з вирішенням завдань у сфері автоматичної обробки текстів. В проекті розглянуто аналіз тональності російськомовних текстів за допомогою згорткової нейронної мережі, для якої векторні уявлення слів були сформовані на основі навченої Word2Vec моделі.
## Вирішення завдання
### Крок 1. Дані для навчання
Для навчання було вибрано корпус текстів, сформований на основі російськомовних повідомлень з Twitter. Він містить 114 991 позитивних, 111 923 негативних твітів, а також базу нерозмічених твітів об'ємом 17 639 674 повідомлень.
```python
# розпаковуємо розмічені датасети
n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
data_positive = pd.read_csv('data/positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])
data_negative = pd.read_csv('data/negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])

sample_size = min(data_positive.shape[0], data_negative.shape[0])
raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                           data_negative['text'].values[:sample_size]), axis=0)
labels = [1] * sample_size + [0] * sample_size
```
* Перед початком навчання тексти пройшли процедуру попередньої обробки:
  * приведення до нижнього регістру;
  * заміна «ё» на «е»;
  * посилань на токен «URL»;
  * заміна згадки користувача на токен «USER»;
  * видалення знаків пунктуації.
```python
# функція для обробки тексту
def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

# обробляєм текст
data = [preprocess_text(t) for t in raw_data]
```


### Крок 2. Векторне відображення слів.
Вхідними даними згорткової нейронної мережі є матриця з фіксованою висотою n, де кожен рядок являє собою векторне відображення слова в просторі ознак розмірності k. Для формування embedding-шару нейронної мережі було використано утиліту дистрибутивної семантики Word2Vec, призначену для відображення семантичного значення слів в векторний простір. Word2Vec знаходить взаємозв'язки між словами згідно з припущенням, що в схожих контекстах зустрічаються семантично близькі слова. Детальніше про Word2Vec можна прочитати в оригінальній [статті](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf).

База нерозмічених твітів зберігається в SQL-форматі і містить понад 17,5 млн. Записів. Для зручності роботи вона була конвертувана в SQLite.
```python
# відкриваємо наш датасет
conn = sqlite3.connect('mysqlite3.db')
c = conn.cursor()

# створюємо файл в який запишем опрацьований текст готовий до тренування
with open('data/tweets.txt', 'w', encoding='utf-8') as f:
    for row in c.execute('SELECT ttext FROM sentiment'):
        if row[0]:
            tweet = preprocess_text(row[0])
            print(tweet, file=f)
 ```
Далі за допомогою бібліотеки Gensim була навчена Word2Vec-модель з наступними параметрами:

* size = 200 - розмірність простору ознак;
* window = 5 - кількість слів з контексту, яке аналізує алгоритм;
* min_count = 3 - слово має зустрічатися мінімум три рази, щоб модель його враховувала.
```python
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# тренуємо модель, використовуємо gensim для векторного подання слів
data = gensim.models.word2vec.LineSentence('daat/tweets.txt')
model = Word2Vec(data, size=200, window=5, min_count=3, workers=multiprocessing.cpu_count())

# зберігаємо модель
model.save("w2v/tweets_model.w2v")
 ```
 ### Підготовка ваг для Embedding шару
 На наступному етапі кожен текст був відображений у масиві ідентифікаторів токенів. Я вибрав розмір тексту s = 26, оскільки при данному значенні повністю покрито 99,71% усіх текстів у сформованому корпусі.  Кінцева розмірність матриці пропозицій склала s × d = 26 × 200.
 Було використане Word2Vec embeddings, які були отримані на попередньому кроці. Це обчислювально ефективна модель для вивчення вбудованих слів, розроблена Google. Детальний посібник з підготовки embedding шару доступний за [адресою](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
 
 ```python
 # відображення кожного речення в масив ідентифікаторів токенів
# максимальна кількість слів в реченні
SENTENCE_LENGTH = 26
# розмір словника
NUM = 100000


def get_sequences(tokenizer, x):
    sequences = tokenizer.texts_to_sequences(x)
    return pad_sequences(sequences, maxlen=SENTENCE_LENGTH)

tokenizer = Tokenizer(num_words=NUM)
tokenizer.fit_on_texts(x_train)

x_train_seq = get_sequences(tokenizer, x_train)
x_test_seq = get_sequences(tokenizer, x_test)

# створення і налаштування нейронної мережі
# загружаем навчену модель векторів слів
w2v_model = Word2Vec.load('models/w2v/tweets_model.w2v')
DIM = w2v_model.vector_size
# заповняєм embedding шар нулями
embedding_matrix = np.zeros((NUM, DIM))
# додаємо NUM=100000 слів які зустрічалися найбільше з навчальної вибірки в embedding шар
for word, i in tokenizer.word_index.items():
    if i >= NUM:
        break
    if word in w2v_model.wv.vocab.keys():
        embedding_matrix[i] = w2v_model.wv[word]

 ```
