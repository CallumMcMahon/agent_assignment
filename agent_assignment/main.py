from enum import Enum
import json
import logging
import hishel
from pydantic import BaseModel, Field
from openai import OpenAI

from agent_assignment.config import Settings
from agent_assignment.data_models import State
from agent_assignment.nodes import build_graph
from agent_assignment.orchestrator import END, START, Command, Graph
from agent_assignment.serper_models import SerperResponse, fetch_and_convert_to_markdown
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logging.getLogger("httpx").setLevel(logging.WARNING)


def run_question(question: str):
    state = State(messages=[{"role": "user", "content": question}], max_retries=3)
    graph = build_graph()
    out: State = graph.run(state)
    formatted_sources = "\n".join(
        [f"{i + 1}. {ref.source}" for i, ref in enumerate(out.references)]
    )
    formatted_answer = f"Final Answer: {out.final_answer}\nReferences: {formatted_sources}"
    return formatted_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a question for the assistant.")
    parser.add_argument("filepath", type=str, help="Path to the file containing the question.")
    args = parser.parse_args()

    with open(args.filepath, "r") as file:
        # one line per question
        questions = file.readlines()
    
    questions = questions[:1]

    for question in questions:
        print(run_question(question))
