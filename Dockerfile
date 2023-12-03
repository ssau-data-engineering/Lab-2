# Используем базовый образ с поддержкой Python 3.9.13
FROM python:3.9.13

# Устанавливаем необходимые библиотеки
RUN pip install pandas numpy matplotlib tqdm torch transformers nltk pymorphy2

# Создаем директорию для хранения данных
RUN mkdir -p /usr/app/src/data

# Копируем все файлы в текущем каталоге в контейнер
COPY . /usr/app/src/

# Устанавливаем рабочую директорию
WORKDIR /usr/app/src

# Копируем файл raz_metka.csv внутрь контейнера в директорию /usr/app/src/data
COPY razmetka/raz_metka.csv /usr/app/src/data/

# Добавляем метки к образу
LABEL maintainer="Bogdann"
LABEL version="1.0"
LABEL description="This is my first Docker image for my lab."