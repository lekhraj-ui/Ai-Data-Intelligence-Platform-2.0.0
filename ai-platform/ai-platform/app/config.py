import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'app.db'}")
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    docs_dir: Path = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "docs")))
    eval_file: Path = Path(os.getenv("EVAL_FILE", str(BASE_DIR / "data" / "eval" / "rag_eval.json")))
    models_dir: Path = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
    fraud_threshold: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.05"))


settings = Settings()
