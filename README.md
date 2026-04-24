# ArthicsuLSTM - Генерация текста с параметрами
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-red)]()
<br>
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119%2B-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Краткое описание
ArthicsuLSTM — это высокопроизводительное веб-приложение для генерации текста на основе параметров пользователя.

## Особенности
- **В модели есть слой Long Short-Term Memory**
- **Ответ модели полностью зависит от параметров, введённых пользователем**

## Установка
> Для работы приложения требуется Python 3.12
- Клонируйте репозиторий в вашу среду разработки:
	```
	git clone https://github.com/Arthicsu/ArthicsuLSTM.git
	```
- Установите все необходимые модули и библиотеки:
	```
	pip install -r requirements.txt
	```
- Скачайте последнюю версию SSD модели. Выше ссылка в разделе **Особенности**

- Запустите приложение:
	```
	uvicorn src.main:app --reload
	```

## Использование
1. Перейдите по адресу `http://localhost:8000`.
2. Напишите начальный текст, установите параметры и нажмите "Сгенерировать текст".

## Лицензия
- Этот проект распространяется под лицензией MIT.