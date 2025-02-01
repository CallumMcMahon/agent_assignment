
from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    openai_api_key: SecretStr
    serper_api_key: SecretStr

