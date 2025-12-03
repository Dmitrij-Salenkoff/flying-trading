from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    clickhouse_url: str

    # StarRocks connection settings
    starrocks_host: str = "localhost"
    starrocks_port: int = 9030
    starrocks_user: str = "root"
    starrocks_password: str = ""
    starrocks_database: str = "dbt"

    bybit_api_key: str
    bybit_api_secret: str
    bybit_testnet: bool = True

    telegram_bot_token: str
    telegram_chat_id: str


config = Config()
