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
  * заміна «е» на «е»;
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
