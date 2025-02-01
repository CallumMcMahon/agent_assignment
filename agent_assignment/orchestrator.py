import logging
from collections import defaultdict
from typing import Any
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START = "start"
END = "end"


class Command(BaseModel):
    state: Any
    next_node: str


class Graph:
    def __init__(self):
        self.name_to_node = {}
        self.edges = defaultdict(list)

    def add_node(self, name: str, node):
        self.name_to_node[name] = node

    def add_edge(self, from_node, to_node):
        self.edges[from_node].append(to_node)

    def run(self, state: BaseModel):
        if START not in self.edges:
            raise ValueError("Graph must define an edge starting from 'start' node")
        current_node = START
        next_node = self.edges[START][0]
        while True:
            logger.info(
                f"About to execute node {next_node} with state: {state.model_dump_json(indent=2)}"
            )
            if next_node == END:
                break
            output = self.name_to_node[next_node](state)
            if isinstance(output, Command):
                state = output.state
                next_node = output.next_node
            else:
                state = output
                next_nodes = self.edges[next_node]
                if len(next_nodes) != 1:
                    raise ValueError(
                        "If next_node is ambiguous, should return a Command object "
                        f"from node {current_node}"
                    )
                next_node = next_nodes[0]
            # current_node = next_node
        logger.info(
            f"Finished executing graph. Final state: {state.model_dump_json(indent=2)}"
        )
        return state
