from pydantic import BaseModel

class Reference(BaseModel):
    statement: str
    source: str


class State(BaseModel):
    question: str | None = None
    plan: str | None = None
    current_query: str | None = None
    selected_link: str | None = None
    extracted_content: str | None = None
    references: list[Reference] = []
    final_answer: str | None = None
    messages: list[dict] = []
    retry_count: int = 0
    max_retries: int = 20