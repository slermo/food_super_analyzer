# Приветствие 
Нейросеть для предсказания калорийности блюд


Так как данные включают как изображения, так и текстовые признаки, была выбрана мультимодальная архитектура, объединяющая:
- "bert-base-uncased"
- "resnet50"

# Запуск
Создание окружения: 
```python
poetry install
```

Создание виртуального окружения с названием food_analyzer:
```python
pip install ipykernel 

python -m ipykernel install --user --name food_analyzer --display-name "Python (food_analyzer)" 
```

# Структура проекта
## Папки:
- `/data` - данные модели, [ссылка](https://disk.yandex.ru/d/kz9g5msVqtahiw)
- `/src` - скрипты проекта

## Файлы:
- `pypoetry.toml` - необходимые зависимости для запуска проекта
- `notebook.ipynb` - ноутбук с описанием задачи, обучением модели и выводами по проделанной работе
- `conf.py` - конфиг проекта в формате py-класса

## Структура папки `/data`:
- `dish.csv` - информация о блюдах. Содержит поля: `dish_id`,`total_calories`, `total_mass` , `ingredients` , `split`
- `ingredients.csv` - информация о названиях ингредиентов в формате `id`,`ingr`
- `/images` - папка с фотографиями блюд в соответствии с `dish_id`