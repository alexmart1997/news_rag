# News RAG

Учебный проект по анализу новостей: сбор корпуса, хранение в базе, тематическая аналитика, поиск похожих материалов и простой QA-интерфейс по новостям.

## Что умеет проект

- загружать новости из `Lenta.ru`;
- сохранять статьи в `SQLite`;
- показывать новости в `Streamlit`-интерфейсе;
- строить темы по новостям через `TF-IDF + KMeans`;
- показывать динамику тем по времени;
- выделять ключевые слова;
- искать похожие статьи;
- отвечать на вопросы по текущей выборке новостей через локальный retrieval.

## Архитектура

```text
news_rag/
  app/
    db/         # подключение к БД
    models/     # ORM-модели
    parsers/    # парсеры источников
    services/   # ingestion и orchestration
    ml/         # аналитика, темы, similarity, QA
    ui/         # Streamlit-интерфейс
  scripts/      # служебные скрипты запуска
  alembic/      # заготовка под миграции
  .streamlit/   # конфиг Streamlit
  streamlit_app.py
  requirements.txt
  README.md
```

## Стек

- `Python`
- `SQLite`
- `SQLAlchemy`
- `pandas`
- `requests + BeautifulSoup`
- `scikit-learn`
- `Streamlit`

## Как запустить локально

### 1. Создать окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Создать `.env`

```powershell
Copy-Item .env.example .env
```

### 3. Инициализировать базу

```powershell
python -m scripts.init_db
```

### 4. Загрузить новости

Быстрый тест:

```powershell
python -m scripts.run_ingestion --date-from 2024-01-01 --date-to 2024-03-31 --size 500 --bloc 4 --chunk-days 37 --no-text
```

Большая историческая загрузка:

```powershell
python -m scripts.load_lenta_history --date-from 2024-01-01 --date-to 2024-12-31 --size 200 --chunk-days 7
```

Если нужны полные тексты:

```powershell
python -m scripts.load_lenta_history --date-from 2024-01-01 --date-to 2024-12-31 --size 200 --chunk-days 7 --with-text
```

### 5. Запустить интерфейс

```powershell
streamlit run streamlit_app.py
```

## Основные скрипты

- `python -m scripts.init_db` — создать таблицы в БД
- `python -m scripts.run_ingestion ...` — загрузить новости за период
- `python -m scripts.load_lenta_history ...` — загрузить большой исторический корпус
- `python -m scripts.show_articles --limit 5` — посмотреть последние статьи в консоли
- `streamlit run streamlit_app.py` — открыть интерфейс

## Что находится в интерфейсе

- фильтры по дате, источнику и ключевому слову;
- таблица новостей;
- ключевые слова;
- темы и их размеры;
- динамика тем;
- похожие статьи;
- блок вопросов по новостям.

## Данные

Локально проект использует файл:

- `news_rag.db`

В этой базе хранятся статьи, источники и все данные, которые нужны интерфейсу.

## Почему SQLite

Для учебного проекта был выбран `SQLite`, потому что:

- не нужен отдельный сервер БД;
- база хранится в одном файле;
- проект легко запускать локально и деплоить;
- для MVP этого достаточно.

При необходимости проект можно перенести на `PostgreSQL`.

## Деплой

Проект можно деплоить через `Streamlit Community Cloud`.

Для этого в репозитории уже есть:

- `streamlit_app.py`
- `.streamlit/config.toml`

Важно: если нужен готовый корпус новостей в облаке, файл `news_rag.db` тоже должен быть в репозитории.

## Идея развития

Дальше проект можно усиливать в любую из сторон:

- сохранять темы и аналитику в БД;
- улучшать naming тем;
- добавить эмбеддинги и более сильный similarity;
- перейти к полноценному RAG с внешней LLM;
- добавить отдельный API.
