"""Search tree for TraceAAD V10.3, unchanged from V10.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

INIT = "Init"
OPERATORS = ("Refine", "Pivot", "Fuse")


def normalize_code(text: str) -> str:
    """Normalize newlines and surrounding whitespace for storage."""
    return "\n".join(text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


@dataclass
class Node:
    id: int
    code: str
    idea: str
    fitness: float
    evaluation_id: int | None = None
    parent_id: int | None = None
    operator: str = "Init"
    donor_id: int | None = None


class SearchTree:
    """Single-parent program tree; children indexed by ``parent_id``."""

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.children: dict[int, list[int]] = {}
        self.roots: list[int] = []
        self.next_id = 0

    def ancestors(self, node_id: int) -> list[Node]:
        """Ancestors of a node, nearest parent first."""
        chain: list[Node] = []
        current = self.nodes[node_id].parent_id
        while current is not None:
            chain.append(self.nodes[current])
            current = self.nodes[current].parent_id
        return chain

    def descendants(self, node_id: int) -> set[int]:
        found: set[int] = set()
        stack = list(self.children.get(node_id, []))
        while stack:
            child = stack.pop()
            if child in found:
                continue
            found.add(child)
            stack.extend(self.children.get(child, []))
        return found

    def add(
        self,
        *,
        code: str,
        idea: str,
        fitness: float,
        evaluation_id: int,
        parent_id: int | None,
        operator: str,
        donor_id: int | None = None,
    ) -> Node:
        """Insert a node into the tree."""
        node = Node(
            id=self.next_id,
            code=code,
            idea=idea,
            fitness=fitness,
            evaluation_id=evaluation_id,
            parent_id=parent_id,
            operator=operator,
            donor_id=donor_id,
        )
        self.next_id += 1
        self.nodes[node.id] = node
        if parent_id is None:
            self.roots.append(node.id)
        else:
            self.children.setdefault(parent_id, []).append(node.id)
        return node

    def add_raw(self, node: Node) -> None:
        """Restore a serialized node."""
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.roots.append(node.id)
        else:
            self.children.setdefault(node.parent_id, []).append(node.id)
        self.next_id = max(self.next_id, node.id + 1)

    def all_nodes(self) -> list[Node]:
        return list(self.nodes.values())

    def best(self) -> Node:
        return max(self.nodes.values(), key=lambda n: n.fitness)

    def to_state(self) -> list[dict]:
        return [asdict(n) for n in self.nodes.values()]
