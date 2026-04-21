from app.db.base import Base
from app.db.session import engine
from app.models import Article, ArticleTopic, Source, Topic


def init_db() -> None:
    # Importing models above is enough for SQLAlchemy metadata registration.
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
