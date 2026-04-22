from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


RUSSIAN_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "но",
    "по",
    "с",
    "со",
    "для",
    "что",
    "это",
    "как",
    "так",
    "тоже",
    "после",
    "при",
    "из",
    "от",
    "до",
    "об",
    "о",
    "у",
    "а",
    "или",
    "же",
    "ли",
    "бы",
    "был",
    "была",
    "были",
    "будет",
    "будут",
    "эти",
    "этот",
    "этой",
    "этом",
    "этому",
    "этого",
    "только",
    "уже",
    "еще",
    "между",
    "без",
    "их",
    "его",
    "ее",
    "им",
    "ими",
    "мы",
    "он",
    "она",
    "они",
    "все",
    "всех",
}

DOMAIN_STOPWORDS = {
    "фото",
    "видео",
    "репортаж",
    "репортер",
    "фоторепортаж",
    "тасс",
    "риа",
    "новости",
    "новость",
    "лента",
    "ру",
    "иа",
    "reuters",
    "bloomberg",
    "агентство",
    "корреспондент",
    "источник",
    "источника",
    "источники",
    "ссылка",
    "ссылки",
    "ссылке",
    "ссылку",
    "ссылкой",
    "ссылок",
    "собеседник",
    "собеседника",
    "собеседники",
    "пресс",
    "служба",
    "прессслужба",
    "кадр",
    "кадры",
    "автор",
    "авторы",
    "редакция",
    "комментарий",
    "комментарии",
    "цитата",
    "цитаты",
    "читайте",
    "читать",
    "также",
    "сообщил",
    "сообщила",
    "сообщили",
    "сообщает",
    "сообщал",
    "сообщала",
    "сообщали",
    "показал",
    "показала",
    "показали",
    "рассказал",
    "рассказала",
    "рассказали",
    "стало",
    "стал",
    "стала",
    "заявил",
    "заявила",
    "заявили",
    "назвал",
    "назвала",
    "назвали",
    "отметил",
    "отметила",
    "отметили",
    "добавил",
    "добавила",
    "добавили",
    "пишет",
    "писал",
    "писала",
    "писали",
    "москва",
    "россия",
    "россии",
    "россиян",
    "россияне",
    "россиянин",
    "российский",
    "российская",
    "российские",
    "рублей",
    "рубля",
    "рубли",
    "рейтерс",
    "рейтер",
    "интерфакс",
    "прайм",
    "блумберг",
    "блумберга",
    "коммерсант",
    "коммерсанта",
    "коммерсантъ",
    "ведомости",
    "известия",
    "рбк",
    "форбс",
    "форбса",
    "газета",
    "издание",
    "издания",
    "канал",
    "канала",
    "канале",
    "ассосиэйтед",
    "асcосиэйтедпресс",
    "пресс",
    "афп",
    "дпа",
    "efe",
    "эфе",
    "анадолу",
    "синьхуа",
    "синьхуань",
    "слово",
    "слова",
    "слов",
}

PERSON_NAME_STOPWORDS = {
    "александр",
    "алексей",
    "андрей",
    "анна",
    "артем",
    "вадим",
    "валерий",
    "виктор",
    "виктория",
    "владимир",
    "вячеслав",
    "галина",
    "геннадий",
    "дарья",
    "денис",
    "дмитрий",
    "евгений",
    "егор",
    "елена",
    "игорь",
    "иван",
    "ирина",
    "кирилл",
    "константин",
    "лариса",
    "леонид",
    "людмила",
    "максим",
    "мария",
    "михаил",
    "наталья",
    "никита",
    "николай",
    "олег",
    "ольга",
    "павел",
    "петр",
    "роман",
    "светлана",
    "сергей",
    "станислав",
    "татьяна",
    "федор",
    "юлия",
    "юрий",
}

ALL_STOPWORDS = RUSSIAN_STOPWORDS | DOMAIN_STOPWORDS | PERSON_NAME_STOPWORDS

NOISE_PATTERNS = [
    r"https?://\S+",
    r"www\.\S+",
    r"фото\s*:\s*[^.;,\n]+",
    r"видео\s*:\s*[^.;,\n]+",
    r"источник\s*:\s*[^.;,\n]+",
    r"по\s+ссылк\w*[^.;,\n]*",
    r"сообщает\s+[^.;,\n]+",
    r"пишет\s+[^.;,\n]+",
]


def clean_text_for_analysis(text: str) -> str:
    cleaned = text.lower()
    cleaned = cleaned.replace("ё", "е")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[/|]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def build_analysis_text(articles_df: pd.DataFrame) -> pd.Series:
    if articles_df.empty:
        return pd.Series(dtype="object")

    title = articles_df["title"].fillna("").astype(str)
    overview = articles_df["overview"].fillna("").astype(str)
    text = articles_df["text"].fillna("").astype(str)

    weighted_text = (
        title + " " + title + " "
        + overview + " " + overview + " "
        + text
    )
    return weighted_text.map(clean_text_for_analysis)


def tokenize_for_analysis(text: str, min_word_length: int = 4) -> list[str]:
    tokens = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    return [
        token
        for token in tokens
        if len(token) >= min_word_length and token not in ALL_STOPWORDS
    ]


def build_daily_counts(articles_df: pd.DataFrame) -> pd.DataFrame:
    if articles_df.empty or "published_at" not in articles_df.columns:
        return pd.DataFrame(columns=["date", "articles"])

    daily_df = articles_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["published_at"], errors="coerce").dt.date
    daily_df = daily_df.dropna(subset=["date"])
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "articles"])

    return (
        daily_df.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "articles"})
        .sort_values("date", ignore_index=True)
    )


def build_top_keywords(
    articles_df: pd.DataFrame,
    top_n: int = 15,
    min_word_length: int = 4,
) -> pd.DataFrame:
    if articles_df.empty:
        return pd.DataFrame(columns=["keyword", "count"])

    text_series = (
        articles_df["title"].fillna("").astype(str) + " " + articles_df["overview"].fillna("").astype(str)
    ).map(clean_text_for_analysis)

    documents = text_series[text_series.str.strip().astype(bool)].tolist()
    if not documents:
        return pd.DataFrame(columns=["keyword", "count"])

    vectorizer = TfidfVectorizer(
        tokenizer=lambda text: tokenize_for_analysis(text, min_word_length=min_word_length),
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        min_df=2 if len(documents) >= 10 else 1,
        max_df=0.35 if len(documents) >= 20 else 0.8,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(documents)
    if matrix.shape[1] == 0:
        return pd.DataFrame(columns=["keyword", "count"])

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = matrix.mean(axis=0).A1
    doc_frequency = (matrix > 0).sum(axis=0).A1

    keywords_df = pd.DataFrame(
        {
            "keyword": feature_names,
            "score": mean_scores,
            "count": doc_frequency,
        }
    )
    keywords_df = keywords_df.sort_values(["score", "count"], ascending=[False, False], ignore_index=True)
    return keywords_df[["keyword", "count"]].head(top_n)


def build_topic_dynamics(articles_df: pd.DataFrame) -> pd.DataFrame:
    if articles_df.empty or "published_at" not in articles_df.columns or "topic" not in articles_df.columns:
        return pd.DataFrame()

    dynamics_df = articles_df.copy()
    dynamics_df["date"] = pd.to_datetime(dynamics_df["published_at"], errors="coerce").dt.date
    dynamics_df = dynamics_df.dropna(subset=["date", "topic"])
    if dynamics_df.empty:
        return pd.DataFrame()

    pivot_df = (
        dynamics_df.groupby(["date", "topic"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    return pivot_df
