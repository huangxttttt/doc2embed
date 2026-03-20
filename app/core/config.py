from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "doc2embed"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    database_url: str
    dify_base_url: str = "http://172.23.27.133:8680"
    dify_dataset_api_key: str = ""
    dify_timeout_seconds: float = 30.0
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 120.0
    llm_batch_size: int = 3
    llm_max_document_chars: int = 12000
    document_export_dir: str = "storage/exports"
    conversation_storage_dir: str = "storage/conversations"
    conversation_recent_turns: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url.startswith("postgresql+"):
            return self.database_url
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace(
                "postgresql://",
                "postgresql+psycopg://",
                1,
            )
        return self.database_url


settings = Settings()
