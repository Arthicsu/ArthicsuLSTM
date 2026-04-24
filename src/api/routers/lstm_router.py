import numpy as np
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf

import re, pickle

from keras.src.utils import pad_sequences


router = APIRouter()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/lstm_model.keras"
TOKENIZER_PATH = "models/tokenizer.pickle"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Модель и токенизатор успешно загружены")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

# Словарь для обратного преобразования индекса в символ
index_to_char = {v: k for k, v in tokenizer.word_index.items()}

def clean_russian_text(text):
    """
    Очистка русского текста от лишних символов

    Параметры:
    - remove_stopwords: удалять ли стоп-слова
    - min_word_length: минимальная длина слова
    """
    if not isinstance(text, str):
        text = str(text)
    # 1. Замена спецсимволов на пробелы
    text = re.sub(r'\xa0', ' ', text)  # неразрывный пробел
    text = re.sub(r'\\[ntr]', ' ', text)  # \n, \t, \r
    text = re.sub(r'\\x[0-9a-f]{2}', ' ', text)  # hex-символы типа \x97
    # 2. Приведение к нижнему регистру и замена ё на е
    text = text.lower().replace('ё', 'е')
    # 3. Удаление всего, кроме русских букв и базовых знаков препинания
    # (если хотите сохранить знаки препинания для анализа тональности,
    # можно добавить !?. в разрешенные символы)
    text = re.sub(r'[^а-я\s]', ' ', text)
    # 4. Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
        if len(generated) < len(seed_text):
            input_seq = generated
        else:
            input_seq = generated[-len(seed_text):]

        x = tokenizer.texts_to_sequences([input_seq])[0]

        if len(x) < len(seed_text):
            x = [0] * (len(seed_text) - len(x)) + x

        x = np.array(x).reshape(1, len(seed_text))

        pred = model.predict(x, verbose=0)[0]

        # Применяем температуру
        pred = np.log(pred + 1e-10) / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))

        next_char_idx = np.random.choice(len(pred), p=pred)
        next_char = index_to_char.get(next_char_idx, ' ')

        generated += next_char

    return generated

@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("lstm_form.html", {
        "request": request,
        "title_app_name": "ArthicsuLSTM",
        "title": "Генерация текста с параметрами"
    })

detection_history = []
@router.post("/predict", response_class=HTMLResponse)
async def predict_display(
        request: Request,
        requested_length: int = Form(25),
        requested_temp: float = Form(0.5),
        requested_seed: str = Form(""),
    ):
    try:
        cleaned_seed = clean_russian_text(requested_seed)

        return templates.TemplateResponse("lstm_result.html", {
            "request": request,
            "generated_text": generate_with_temperature(cleaned_seed, length=requested_length, temperature=requested_temp),
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})