from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "aiops"

    llm_provider: str = "claude"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Training mode
    demo_mode: bool = True

    # Demo-mode hyperparameters (~90 s, good for presentations)
    demo_epochs: int             = 8
    demo_kpi_stream_seconds: int = 600
    demo_max_samples: int        = 200
    demo_batch_size: int         = 128
    demo_learning_rate: float    = 0.01

    # Full-mode hyperparameters (production-quality, longer run)
    full_epochs: int             = 100
    full_kpi_stream_seconds: int = 6000
    full_max_samples: int        = 5000
    full_batch_size: int         = 32
    full_learning_rate: float    = 0.001


settings = Settings()
