from enum import Enum
import json
import logging
import hishel
from pydantic import BaseModel, Field
from openai import OpenAI
import httpx

from agent_assignment.config import Settings
from agent_assignment.orchestrator import END, START, Command, Graph
from agent_assignment.serper_models import SerperResponse, fetch_and_convert_to_markdown
from markdownify import markdownify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logging.getLogger("httpx").setLevel(logging.WARNING)

controller = hishel.Controller(force_cache=True, cacheable_methods=["GET", "POST"])
storage = hishel.FileStorage()
openai_http_client = hishel.CacheClient(controller=controller, storage=storage)
client = OpenAI(
    api_key=Settings().openai_api_key.get_secret_value(), http_client=openai_http_client
)
SERPER_URL = "https://google.serper.dev/search"
http_client = httpx.Client(
    headers={
        "X-API-KEY": Settings().serper_api_key.get_secret_value(),
        "Content-Type": "application/json",
    }
)

# response = client.chat.completions.create(messages=state.messages, model="gpt-4o")
# state.messages.append(response.choices[0].message.content)


class State(BaseModel):
    question: str | None = None
    plan: str | None = None
    current_query: str | None = None
    selected_link: str | None = None
    messages: list[dict] = []
    retry_count: int = 0
    max_retries: int = 20


def get_name(state: State):
    user_name = input("Enter your name: ")
    state.user_name = user_name
    response = client.chat.completions.create(messages=state.messages, model="gpt-4o")
    state.messages.extend(response.choices[0].message.content)
    state.retry_count += 1
    if state.retry_count < 3:
        return Command(state=state, next_node="get_name")
    else:
        return Command(state=state, next_node=END)


decision_prompt = """You are a planning assistant who must break down a complex task into smaller steps.
You have access to a web search tool, which you must use to find information to complete the task.
Given the available information, decide if the task has been completed successfully."""


class NextStep(str, Enum):
    PLAN = "plan"
    END = "end"


class Decision(BaseModel):
    """Decision on whether to continue planning or finish."""

    decision: NextStep


def end_decision(state: State):
    state.retry_count += 1
    if state.retry_count > state.max_retries:
        logger.info("Max retries reached. Ending conversation.")
        return Command(state=state, next_node=END)

    messages = [{"role": "system", "content": decision_prompt}] + state.messages
    response = client.beta.chat.completions.parse(
        messages=messages, model="gpt-4o", response_format=Decision
    )
    decision = response.choices[0].message.parsed
    command = Command(state=state, next_node=decision.decision)
    return command


plan_prompt = """You are a planning assistant who must break down a complex task into smaller steps.
You have access to a web search tool, which you must use to find information to complete the task.
Given the work you have done so far, write or update the high-level plan, and provide the next search query."""


class Plan(BaseModel):
    """Plan and next search query."""

    plan: str
    query: str


def plan(state: State):
    messages = [{"role": "system", "content": plan_prompt}] + state.messages
    if state.plan:
        messages.append({"role": "system", "content": f"Current plan: {state.plan}"})
    else:
        messages.append({"role": "system", "content": "No plan has been made yet."})
    response = client.beta.chat.completions.parse(
        messages=messages, model="gpt-4o", response_format=Plan
    )
    plan = response.choices[0].message.parsed
    state.plan = plan.plan
    state.messages.append(
        {"role": "system", "content": f"Next search query: {plan.query}"}
    )
    state.current_query = plan.query
    return state


def dummy_search(state: State):
    state.messages.append(
        {"role": "system", "content": "Search API is down, please end the session."}
    )
    return state

link_picker_prompt = """You are a search assistant who must find the most relevant information to complete a task.
Given the plan and the latest search results, select the most promising link to expand the aquired sources."""

class LinkPicker(BaseModel):
    """The webpage to select"""

    reasoning: str
    link: int = Field(..., description="The number of the most promising link to select.")

def search(state: State):
    response = http_client.post(url=SERPER_URL, data=json.dumps({"q": state.current_query}))
    logger.info(response.text)
    search_out = response.json()
    logger.info(f"Search results: {search_out}")
    response = SerperResponse(**search_out)
    results = []
    for i, webpage in enumerate(response.organic, start=1):
        formatted = f"{i}. {webpage.title} - {webpage.snippet} - {webpage.link} - {webpage.date}"
        results.append(formatted)
    formatted_results = "\n".join(results)

    state.messages.append(
        {"role": "system", "content": f"Search results:\n{formatted_results}"}
    )
    llm_response = client.beta.chat.completions.parse(
        messages=state.messages, model="gpt-4o", response_format=LinkPicker
    )
    link_picker = llm_response.choices[0].message.parsed
    logger.info(f"Reasoning for picking link: {link_picker.reasoning}")
    if link_picker.link < 1 or link_picker.link > len(results):
        state.messages.append(
            {"role": "system", "content": "Invalid link number selected."}
        )
        return Command(state=state, next_node="search")
    selected_link = response.organic[link_picker.link - 1]
    state.selected_link = selected_link.link
    return state

extract_prompt = """You are an information assistant who must extract relevant information from a webpage.
Given the question and current plan, extract the relevant information from the selected webpage."""
def extract_claim(state: State):
    content = fetch_and_convert_to_markdown(state.selected_link)
    messages = [{"role": "system", "content": extract_prompt}] + state.messages



def main(question: str):
    state = State(messages=[{"role": "user", "content": question}], max_retries=1)
    graph = Graph()
    # graph.add_node("get_name", get_name)
    # graph.add_edge(START, "get_name")
    # graph.add_edge("get_name", END)
    graph.add_node("end_decision", end_decision)
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_edge(START, "end_decision")
    graph.add_edge("end_decision", "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "end_decision")
    graph.add_edge("end_decision", END)
    out = graph.run(state)


if __name__ == "__main__":
    # question = input("Enter your question: ")
    question = (
        "Compile a list of 10 statements made by Joe Biden regarding US-China relations. "
        "Each statement must have been made on a separate occasion. Provide a source for each statement."
    )
    main(question=question)
