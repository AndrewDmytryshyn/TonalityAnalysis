# Аналіз предметної області
*Розділ розробив Чередник Віталій*

## 1. Вступ

Даний документ надає описову інформацію по проекту, а саме, надає основні визначення термінів, які будуть використані в проекті, описує підходи та способи вирішення завдання. На основі цих підходів та способів вирішення завдання приведена їх порівняльна характеристика, а також висновки.

## 2. Основні визначення

### 2.1 Термінологія, яка буде використана в проекті або його стосується

* **Аналіз тональності тексту** - клас методів контент-аналізу в комп'ютерній лінгвістиці, призначений для автоматизованого виявлення в текстах емоційно забарвленої лексики і емоційної оцінки авторів (думок) по відношенню до об'єктів, мова про які йде в тексті.
* **Тональність** - емоційне ставлення автора висловлювання до деякого об'єкту (об'єкту реального світу, події, процесу або їх властивостями / атрибутам), виражене в тексті. Емоційна складова, виражена на рівні лексеми або комунікативного фрагмента, називається лексичною тональністю (або лексичним сентиментом).
* **Машинне навчання** - один з розділів штучного інтелекту, алгоритми, що дозволяють комп'ютеру робити висновки на підставі даних, не слідуючи жорстко заданим правилами. Тобто машина може знайти закономірність у складних і багато-параметричних завданнях (які мозок людини не здатен вирішити), таким чином знаходячи більш точні відповіді.
* **Word embedding** - стисле векторне представлення слова.

### 2.2 Актуальність

Великий інтерес до аналізу настрою текстів користувачів викликаний стрімким розвитком соціальних мереж, таких як Facebook, Twitter, Instagram. Аналіз тональності текстів відноситься до класу методів, заснованих на обчислювальній обробці, що використовується для ідентифікації, вилучення та аналізу суб'єктивної інформації, думок, що виражають користувачі у своїх текстах. Це дозволяє класифікувати ставлення автора до різних тем в позитивні, нейтральні та негативні категорії. Аналіз тонального забарвлення тексту має багато застосувань у таких галузях, як бізнес-аналітика, політика, соціологія і т.д. Такі дані дозволяють збирати та аналізувати ставлення, настрої та думки людей щодо різних подій.

### 2.3 Завдання

Основні задачі аналізу тональності тексту:
* вилучення сутностей та їх категоризація вилучити всі вирази сутностей в емоційно забарвленому документі та виконати категоризацію та розбиття їх на класи сутностей;
* вилучення аспектів сутностей та їх категоризація. Вилучити всі вирази сутностей в емоційно забарвленому документі та виконати категоризацію та розбиття їх на класи сутностей;
* вилучення власників думки та їх класифікація. Вилучити власників думки та класифікувати їх;
* вилучення часу та стандартизація. Необхідно вилучити всі часові проміжки, коли були висловлені думки їх власниками; 
* класифікація емоційного забарвлення аспектів. Необхідно визначити, чи є думка що до аспекту позитивною, негативною чи нейтральною, або надати числове значення що визначає думку.

### 2.4 Проблеми

Основні проблеми, що виникають при аналізі тональності тексту:
* позитивне чи негативне слово може приймати протилежний відтінок при використанні в іншій предметній області;
* наявність неологізмів та помилок у словах, що у багатьох випадках робить використання тональних словників недоцільним;
* речення, що містить в собі слово емоційного забарвлення, може мати нейтральний сентимент;
* речення, що містять сарказм з наявністю чи відсутністю слів емоційного забарвлення є дуже складними для аналізу.

Всі описані вище проблеми є досить серйозними труднощами для виконання аналізу тональності, що базується на лексичному словнику. Саме тому, методи, що використовують лексичний словник, як правило, застосовуються рідше, ніж альтернативні методи.

## 3. Підходи та способи вирішення завдання

### 3.1 Підходи та способи вирішення завдання

Всі підходи до аналізу тональності тексту можна розділити на три групи.

#### 3.1.1 Підходи на основі правил (rule-based)
Найчастіше в таких підходах використовуються вручну задані правила класифікації і емоційно розмічені словники. Ці правила, зазвичай на основі емоційних ключових слів і їх спільного використання з іншими ключовими словами, розраховують клас тексту. Незважаючи на прекрасну ефективність в текстах з якоїсь певної тематики, методи на основі правил мають погану здатність до узагальнення. Крім того, вони вкрай трудомісткі у створенні, особливо коли немає доступу до потрібного тонального словника. Останнє особливо характерно для української мови, тому що нею не настільки багато джерел, як англійською, особливо в сфері аналізу тональності. Українські тональні словники містять тільки інформацію про тональне забарвлення слів від позитивного до негативного, без характеристик емоцій. Таким чином, не існує альтернатив таким потужним англомовним добіркам з великими емоційними характеристиками, як SenticNet, SentiWordNet і SentiWords.

#### 3.1.2 Підходи на основі машинного навчання
Підходи на основі мащинного навчання використовують автоматичне вилучення ознак з тексту і застосування алгоритмів машинного навчання. Класичними алгоритмами класифікації полярності є наївний баєсівський класифікатор (Naive Bayes Classifier), дерево рішень (Decision Tree), логістична регресія (Logistic Regression) і метод опорних векторів (Support Vector Machine). В останні роки увагу дослідників привертають методи глибокого навчання, які значно перевершують традиційні методи в аналізі тональності. Одна з головних особливостей систем на основі машинного навчання - автоматичне вилучення ознак з тексту. У простих підходах для представлення тексту у векторному просторі зазвичай використовується модель «мішок слів» (bag of words). У більш складних системах для генерування word embedding застосовуються моделі дистрибутивної семантики, наприклад, Word2Vec, GloVe або FastText. Також є алгоритми генерування word embedding на рівні речень або параграфів, які призначені для перенесення навчання в різних задачах обробки мови.
До таких алгоритмів відносяться ELMo , Universal Sentence Encoder (USE), Bidirectional Encoder Representations from Transformers (BERT), Enhanced Language Representation with Informative Entities (ERNIE) і XLNet. Одним з їхніх головних недоліків з точки зору генерування word embedding є потреба у великих масивах текстів для навчання. Втім, це справедливо для всіх методів машинного навчання, тому що всім алгоритмам для навчання потрібні розмічені набори даних.

#### 3.1.3 Гібридні підходи
Гібридні підходи об'єднують в собі підходи двох попередніх видів. Мовні моделі часто застосовуються в гібридних алгоритмах, як і рішення на основі правил. З одного боку, комбінація методів на основі правил і машинного навчання зазвичай дозволяє домогтися більш точних результатів. А з іншого - гібридні підходи успадковують труднощі і обмеження складових їх алгоритмів.

## 4. Висновки

Створення робочого сервісу для аналізу емоційного забарвлення тексту є виправданим і актуальним завданням, адже проаналізувавши предметну область та переглянувши вже існуючі сервіси, можна зробити висновок, що сучасні передові сервіси не підтримують роботу з текстом українською мовою.

## 5. Посилання