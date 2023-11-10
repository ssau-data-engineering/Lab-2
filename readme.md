# Отчет по лабораторной работе №2



## Пайплайн для обучения

Была выбрана [модель классификации текстов](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment). Модель была модифицирована: был убран класс neutral (не смог найти нормальных данных с твиттера с тремя метками, везде только negative и postitive). Был собран докер образ (директория train_model) со следующей структурой рабочей директории:

```
.
└── wd/
    ├── data/
    │   ├── data0.csv
    │   ├── data1.csv
    │   └── ...
    ├── model/
    │   ├── ...
    │   ├── config.json
    │   ├── model.bin
    │   └── ...
    ├── results/
    │   └── log.log
    └── scripts/
        ├── data.py
        ├── main.py
        └── train.py
```


Все директории маунтятся к диску (scripts тоже, что было удобно при разработке, но не совсем правильно для прода). При запуске контейнера запускается скрипт ``main.py``. В файле ```train.py``` содержатся функции для обучения моделей. Модель цикл обучения модели запускается для каждого файла в папке ```data```.

<p align="center">
  <img width="800" height="300" src="https://github.com/Anteii/Lab-2/blob/main/screenshots/train-log.png"/>
</p>

Для того, чтобы построить лог использовал небольшую выборку данных, поскольку обучение проходило на CPU, поэтому такие неплохие результаты.
Основная сложностью на данном этапе стали:

* Устновка модуля провайдера docker для ariflow (для локального инстанса)
* Разбирался как дать доступ к ГПУ (разобрался, но по техническим причинам не смог)
* Поиск данных для обучения (долго пытался найти данные с тремя метками: positive/neutral/negative)
* Разбирался как заставить докер ``dind`` видеть собранные образы.



## Пайплайн инференса

Для FileSensor в веб интерфейсе Airflow был создан Connection.

<p align="center">
  <img width="800" height="300" src="https://github.com/Anteii/Lab-2/blob/main/screenshots/airflow-connection.png"/>
</p>

Докер контейнеры для каждого этапа имеют такую же структуру как и контейнер для обучения из прошлого пункта. 

В качестве видео использовался этот [ролик](https://www.youtube.com/watch?v=nreoAJHMtFM&list=PLoWjlqRGkEhtJWnqOFnWwNAy28P5OfDnu&index=5).

Из видео при помощи ```ffmpeg``` была извлечена звуковая дорожка.

Из неё был извлечен следующий текст (оформление было добавлено вручную, знаки препинания проставила модель):

    Here we go..  
    You're just like an angel.  
    Your skin makes me cry.  
    You flow like a feather.  
    In a beautiful world.  
    You're so very special.  
    I wish I was special.  

    But I'm a creep.  
    And I'm always a fool.  
    What in the hell am I doing here?.  
    I don't belong here.  

    I don't care the hurts.  
    I wanna have control.  
    I ain't want a perfect body.  
    I want a perfect soul.  
    I want you to notice.  
    When I'm not around. 
    You're so special.  
    I wish I was special. 

    But I'm a creep.  
    And I'm always a fool.  
    What in the hell am I doing here?.  
    I don't belong here.  
    
    Oh, oh, now she's.  
    Running out the door.  
    She's running out.  
    She'll run, run, run, run.  
    Run, run.  
    
    Whatever makes you happy.  
    Whatever you want.  
    You're so special.  
    I wish I was special.  
    
    But I'm a creep.  
    And I'm always a fool.  
    What in the hell am I doing here?.  
    I don't belong here.  
    I don't belong here

Модель суммаризации выдала следующий результат:

```
A short story of a love story about the life of a man who's got stuck in the room atthe end of the year, and it was a "creep" and I'm a creep.
```

Он был сохранен в пдф (```inference_data/reports```).

Каждому этапу соответсвует докер контейнер.
Вся кодовая база помещалась в директорию airflow/data (поскольку код маунтится в контейнеры).
Собранные образы пушились в регистри docker hub.

Из сложностей:
* Большой объем работы
