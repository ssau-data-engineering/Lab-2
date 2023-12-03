import pandas as pd
import re

# чтение датафрейма (ограничим датасет тк на CPU обучение полного датасета будет слишком долго идти)
df = pd.read_csv('/data/input/raz_metka.csv', sep='\t')[0:1000]

df.head()

# названия категорий которые будем оставлять в датафрейме
CATEGORY_NAMES = df.category.unique()

# оставить в датасете только выбранные категории
df = df[df.category.isin(CATEGORY_NAMES)]

# удалить дубликаты
df = df.drop_duplicates(subset=['text'])

def preprocess_text(text):
    # привести в нижнему регистру
    text = text.lower()
    # удаление 0 или более цифр и знаков переноса строки в конце предложения
    text = re.sub('^\d*\n*|\n*\d*$', '', text)
    return text

df.text = df.text.apply(preprocess_text)

df.to_csv('/data/output/ml_data.csv', sep='\t')
