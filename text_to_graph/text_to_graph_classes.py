from typing import List, Union
from langchain_core.pydantic_v1 import Field
from langchain_core.documents import Document
from langchain_core.load.serializable import Serializable

class Node(Serializable):
    """Represents a node in a graph with associated properties.

    Attributes:
        id (str): A unique identifier for the node. Use lowercase and underscores instead of spaces. Examples: 'professor', 'mercedes_benz', 'nobel_prize'
        label (str): The label of the node. Examples: 'Car', 'Winner', 'Stone', 'Professor'. Don't combine, keep it simple. E.g. 'President Trump' should be 3 nodes, one of label 'Donald Trump', one of label 'Person', one of label 'President'.
    """
    id: str
    label: str
      

class Relationship(Serializable):
    """Represents a directed relationship between two nodes in a graph.

    Attributes:
        source (str): The source node of the relationship. It must be from the 'id' field of the node list. Use exactly the same name.
        target (str): The target node of the relationship. It must be from the 'id' field of the node list. Use exactly the same name.
        type (str): The type of the relationship. Often a verb or an adjective e.g. 'greater than', 'drinks', 'has signed' etc.
    """
    source: Node
    target: Node
    type: str


class GraphDocument(Serializable):
    """Represents a graph document consisting of nodes and relationships.

    Attributes:
        nodes (List[Node]): A list of nodes in the graph.
        relationships (List[Relationship]): A list of relationships in the graph.
        source (Document): The document from which the graph information is derived.
    """
    nodes: List[Node]
    relationships: List[Relationship]
    source: Document