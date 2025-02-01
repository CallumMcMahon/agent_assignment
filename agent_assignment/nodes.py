"""Set of Agent nodes that call LLMs and tools to complete a task."""

from enum import Enum
import json
import logging

import hishel
from openai import OpenAI
from pydantic import BaseModel, Field

from agent_assignment.config import Settings
from agent_assignment.data_models import Reference, State
from agent_assignment.orchestrator import END, START, Command, Graph
from agent_assignment.serper_models import SerperResponse, fetch_and_convert_to_markdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

controller = hishel.Controller(force_cache=True, cacheable_methods=["GET", "POST"])
storage = hishel.FileStorage()
openai_http_client = hishel.CacheClient(controller=controller, storage=storage)
client = OpenAI(
    api_key=Settings().openai_api_key.get_secret_value(), http_client=openai_http_client
)
SERPER_URL = "https://google.serper.dev/search"
http_client = hishel.CacheClient(
    controller=controller,
    storage=storage,
    headers={
        "X-API-KEY": Settings().serper_api_key.get_secret_value(),
        "Content-Type": "application/json",
    },
)

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
        return Command(state=state, next_node="summarise")

    messages = [{"role": "system", "content": decision_prompt}] + state.messages
    response = client.beta.chat.completions.parse(
        messages=messages, model="gpt-4o", response_format=Decision
    )
    decision = response.choices[0].message.parsed
    decision_mapping = {NextStep.PLAN: "plan", NextStep.END: "summarise"}
    command = Command(state=state, next_node=decision_mapping[decision.decision])
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
    link: int = Field(
        ..., description="The number of the most promising link to select."
    )


def search(state: State):
    response = http_client.post(
        url=SERPER_URL, data=json.dumps({"q": state.current_query})
    )
    search_out = response.json()
    logger.debug(f"Search results: {search_out}")
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


class ExtractedContent(BaseModel):
    """Extracted content. If multiple distinct statements are needed, provide a list of strings.
    If not, provide a list with a single string.
    """

    content: list[str]


def extract_claim(state: State):
    try:
        content = fetch_and_convert_to_markdown(state.selected_link)
    except Exception as e:
        logger.error(f"Error fetching content: {e}")
        state.messages.append(
            {
                "role": "system",
                "content": f"Error fetching content from {state.selected_link}.",
            }
        )
        return Command(state=state, next_node="search")
    logger.info(f"Extracted content: {content}")
    messages = [{"role": "system", "content": extract_prompt}]
    # add plan
    messages.append({"role": "system", "content": f"Current plan: {state.plan}"})
    messages.append(
        {"role": "system", "content": f"Retrieved content from the webpage: {content}"}
    )
    response = client.beta.chat.completions.parse(
        messages=messages, model="gpt-4o", response_format=ExtractedContent
    )
    extracted = response.choices[0].message.parsed
    logger.info(f"Extracted content: {extracted.content}")
    state.extracted_content = extracted.content
    return state


filter_prompt = """You are an information assistant who must filter the extracted information.
Given the extracted information and existing references, decide if the information should be incorporated into the references."""


def filter_statements(state: State):
    for statement in state.extracted_content:
        state.references.append(
            Reference(statement=statement, source=state.selected_link)
        )
    return state


summary_prompt = """You are a summarization assistant who must summarize the retrieved information.
Given the extracted information and references, combine the information into a coherent summary.
use inline citations to reference the sources of the information. For example, "According to [1], Joe Biden said...".
All direct quotes must be copied verbatim from the source. Do not write a bibliography.
"""


class SummarisedContent(BaseModel):
    """Summarised content."""

    content: str


def summarise(state: State):
    logger.info("Summarising conversation...")
    messages = [{"role": "system", "content": summary_prompt}]
    messages.append({"role": "user", "content": state.question})
    references = []
    for i, reference in enumerate(state.references, start=1):
        references.append(f"[{i}] {reference.statement}")
    formatted_references = "\n".join(references)
    messages.append(
        {"role": "system", "content": f"References:\n{formatted_references}"}
    )
    response = client.beta.chat.completions.parse(
        messages=messages, model="gpt-4o", response_format=SummarisedContent
    )
    summary = response.choices[0].message.parsed
    logger.info(f"Summarised content: {summary.content}")
    state.final_answer = summary.content
    return state


def build_graph():
    graph = Graph()
    graph.add_node("end_decision", end_decision)
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_node("extract_claim", extract_claim)
    graph.add_node("filter_statements", filter_statements)
    graph.add_node("summarise", summarise)
    graph.add_edge(START, "end_decision")
    graph.add_edge("end_decision", "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "extract_claim")
    graph.add_edge("extract_claim", "filter_statements")
    graph.add_edge("filter_statements", "end_decision")
    graph.add_edge("end_decision", "summarise")
    graph.add_edge("summarise", END)
    return graph
