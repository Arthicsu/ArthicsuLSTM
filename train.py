from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import numpy as np
import re, pickle


# 1. Загрузка и предобработка данных
with open('src/models/dataset/train_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')  # убираем первый невидимый символ
    text = re.sub(r'[^А-я ]', '', text)  # заменяем все символы кроме кириллицы и пробела на пустые символы

# 2. Создание токенизатора (на уровне символов)
num_characters = 34  # 33 буквы + пробел
tokenizer = Tokenizer(num_words=num_characters, char_level=True, lower=False)
tokenizer.fit_on_texts([text])
with open('src/models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Словарь для обратного преобразования индекса в символ
index_to_char = {v: k for k, v in tokenizer.word_index.items()}

# 3. Преобразование текста в последовательность индексов
# (вместо one-hot encoding используем индексы для Embedding слоя)
sequences = tokenizer.texts_to_sequences([text])[0]

# 4. Создание обучающей выборки с использованием индексов
inp_chars = 6  # длина входной последовательности
n = len(sequences) - inp_chars

X = []
Y = []

for i in range(n):
    X.append(sequences[i:i + inp_chars])
    Y.append(sequences[i + inp_chars])

X = np.array(X)
Y = np.array(Y)

# Преобразуем Y в one-hot encoding для categorical_crossentropy
Y_one_hot = to_categorical(Y, num_classes=num_characters)

print(f"Размер X: {X.shape}")
print(f"Размер Y: {Y_one_hot.shape}")

# 5. Создание модели с Embedding и LSTM
model = Sequential([
    # Embedding слой: преобразует индексы (0-33) в плотные векторы размерности 64
    Embedding(input_dim=num_characters, output_dim=64, input_length=inp_chars),

    # LSTM слой (вместо SimpleRNN)
    # return_sequences=False (по умолчанию) - возвращает только последний выход
    LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2),

    # Выходной слой
    Dense(num_characters, activation='softmax')
])

model.summary()

# 6. Компиляция и обучение
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

history = model.fit(X, Y_one_hot, batch_size=64, epochs=100, validation_split=0.1)

model.save('src/models/lstm_model.keras')
print("Модель сохранена как 'nlp_model.keras'")



# 8. Тестирование генерации
print("\n" + "="*50)
print("ГЕНЕРАЦИЯ ТЕКСТА")
print("="*50)

seed_phrases = ["привет", "жизнь", "будь", "я хочу", "самое главное"]

for seed in seed_phrases:
    result = generate_text(seed, length=100)
    print(f"\nНачало: '{seed}'")
    print(f"Результат: {result}")
    print("-"*50)

# 10. Пример работы с разной температурой (softmax temperature)
def generate_with_temperature(seed_text, length=50, temperature=1.0):
    """
    Генерация с контролем "креативности" через температуру

    Args:
        seed_text: начальная строка
        length: количество символов
        temperature: параметр температуры (меньше - более детерминировано,
                    больше - более случайно)
    """
    generated = seed_text

    for _ in range(length):
        if len(generated) < inp_chars:
            input_seq = generated
        else:
            input_seq = generated[-inp_chars:]

        x = tokenizer.texts_to_sequences([input_seq])[0]

        if len(x) < inp_chars:
            x = [0] * (inp_chars - len(x)) + x

        x = np.array(x).reshape(1, inp_chars)

        pred = model.predict(x, verbose=0)[0]

        # Применяем температуру
        pred = np.log(pred + 1e-10) / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))

        next_char_idx = np.random.choice(len(pred), p=pred)
        next_char = index_to_char.get(next_char_idx, ' ')

        generated += next_char

    return generated

print("\n" + "="*50)
print("ГЕНЕРАЦИЯ С РАЗНОЙ ТЕМПЕРАТУРОЙ")
print("="*50)

seed = "я верю"
for temp in [0.5, 1.0, 1.5]:
    result = generate_with_temperature(seed, length=80, temperature=temp)
    print(f"\nТемпература = {temp}:")
    print(f"{result}")
    print("-"*50)