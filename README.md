# fincubator_hacksai

## Структура проекта

```
├─ configs/
├─ data/
├─ models/
├─ objects/
│  └─ encoders/
├─ results/
└─ src
   ├─ app
   │  ├─ server.py
   │  └─ templates
   │     └─ index.html
   ├─ notebooks
   │  ├─ baseline.ipynb
   │  └─ eda.ipynb
   └─ utils
      ├─ model.py
      ├─ predict.py
      └─ transforms.py
```
- ```src/app/``` - микросервис
- ```src/notebooks``` - .ipynb ноутбуки с экспериментами
- ```src/utils/```:
    - ```model.py``` - модуль для обучения модели
    - ```predict.py``` - модуль для инференса
    - ```transforms.py``` - вспомогательные трансформации данных


## Установка библиотек
```
python3 -m pip install -r requirements.txt
```

## Подзадача 1

### Локальный деплой

#### Запуск сервиса на локальном хосте

Реализован web-сервис на ```flask```, запуск сервиса:
```
python3 -m src.app.server
```
Сервис доступен по адресу http://127.0.0.1:5000

#### Интерфейс сервиса

![Screenshot](interface.png)

После загрузки файла и клика "Получить результат переклассификации" начинает скачиваться файл с полями ```"id", "Тип переклассификации", "Тип финального запроса"```.

### Автономная работа

Получение результата возможно непосредственно с использование командной строки:

```
python3 -m src.utils.predict data/test.csv
```

Результат сохраняется в папку ```results/```