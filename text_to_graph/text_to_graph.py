from typing import Optional, Sequence, List, Dict, Tuple
import json, logging, pickle, math
from textwrap import dedent
from logging import Logger

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# from langchain_community.graphs.graph_document import GraphDocument
from text_to_graph.text_to_graph_classes import Node, Relationship, GraphDocument
from text_to_graph.text_to_graph_utils import GraphCallBackHandler


# from langchain_community.graphs.graph_document import Node, Relationship

sys_prompt_initial = (dedent(
    """You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
    Try to capture as much information from the text as possible without sacrificing accuracy.
    Do not add any information that is not explicitly mentioned in the text.
    The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.""")
)

human_prompt_initial_nodes = (dedent(
    """Create a knowledge graph of a document. Only extract the nodes (vertices).
    1.Nodes (Vertices) IDs represent entities and concepts.
    The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
    Nodes are not relationships (edges). Don't extract relationships. E.g. a 'boss' is a 'position' and a node is always a noun and never an adjective, adverb or a verb.
    Don't combine different information into one node. For example 'a Professor Frank Miller' are two nodes, 'Professor' and 'Frank Miller'. Keep the nodes short, not more than 4 words. Better 1 word only.
    If they get too long break it into 2 nodes.
    2. Labeling Nodes
    Consistency: Ensure you re-use already assigned types. For example when you identify a person,
    always label it as 'person' Avoid using more specific terms like 'mathematician' or 'scientist'.
    3. Node IDs: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text. Make them lower-case without space. You can use underscores.
    4. Extraction of nodes:
    Use the document below. Only extract the nodes and report back as JSON. The JSON should be a list of nodes.
    Only give a valid JSON list of nodes as response. Nothing else.
    
    {format_instructions}
    
    {nodes}
       
    The document content you have to use is below:
    
    {content}
    """)
)

human_prompt_find_node = dedent(
    """You are trying to resolve a problem with a previous edge extraction. The edge extracted was {extracted_edge}.
    The node '{node_missing}' of this edge however is not part of the list of nodes. This could be because not all context was available in the previous step. You now get a wider context below.
    For example, if the missing node was 'he', 'him', 'she', 'her, 'it' and such look at below text and try to find out the person's name or object's name (e.g. 'Frank' or 'President' or 'Apple' or 'Dream') and provide the corrected edge based on the formatting instruction.
    In short: ONLY Repair the faulty edge {extracted_edge} - You can provide more than one edge as result set if it helps. But don't create edges not related to the faulty node. Double-check that before giving back the results.
    
    {format_instructions}
    
    The nodes you have to use are below - only use these:
    
    {nodes}
    
    The document content you have to use is below - at the end double-check that you have used only existing 'id' from the node list.
    
    {content}                                    
    """)

human_prompt_qc_nodes = (
    "I want to create a knowledge graph of a document. Now I'm only interested in the nodes (vertices). "
    "Each node consists out of a unique ID (nodeid) and a node description. Use lowercase text for the 'nodeid' and where needed underscores but no spaces."
    "You will get a document below. Only extract the vertices and report back as JSON. The JSON should be a list of vertices\n"
    "Make sure to only give a valid JSON list of nodes as response. Nothing else.\n"
    "{format_instructions}\n"
    "From a previous analysis you have this JSON - check if entries need to be augmented, removed or added. This is a quality check step.\n"
    "{first_run_nodes}\n"
    "The document content you have to use is below:\n\n{content}"
)

human_prompt_initial_edges = (dedent("""
    Create a knowledge graph of a document. Below the list of nodes you have to use. Only extract the relationships (edges of the graph)!
    1. Relationships: represent connections between entities or concepts.
    Ensure consistency and generality in relationship types when constructing knowledge graphs. 
    Instead of using specific and momentary types such as 'became_professor', use more general and timeless relationship types like 'professor'.
    Under all circumstances avoid 'he' or 'she' or 'it'. Replace it with the subject (e.g. 'queen' or 'mr_miller') or object ('apple', 'car' etc.). Get the information from the context.
    Also try to relate e.g. 'man' or 'girl' to a specific node id from the list like 'millers_daughter' or 'jonathan' etc.
    Make sure to use general and timeless relationship types!
    Use lowercase without spaces, you can use underscores to separate words if needed.
    2. Relationship structure:
    Each relationship consists out of a source and a target node ID which you get from the list of nodes. Both source as well as target must be in the node list and must be from the 'id' field!
    Don't use the 'type' field. Only 'id' field from the node list. Don't change the 'id'. Keep it exactly as it is.
    Also only extract the relationships between the nodes as well as the type. Report the relationship list as JSON. Nothing else than the JSON list.
    
    {format_instructions}
    
    The nodes you have to use are below - only use these:
    
    {nodes}
    
    The document content you have to use is below - at the end double-check that you have used only existing 'id' from the node list.
    
    {content}                                    
    """)
)

human_prompt_deduplicate_nodes = (dedent("""
    Below is a sorted list of nodes that got extracted from a text to build a knowledge graph. Remove duplicates from the JSON list and improve/ generalize the nodes.
    Don't remove nodes that are correct defined. If you discover a node combination, split it. For example 'day_in_april' into 'day' and 'april'.
    Return the same format of JSON list of the improved list of nodes. Don't invent nodes that don't relate to nodes in the list.
    
    {format_instructions}
    
    The list of nodes to be checked and improved:
    
    {nodes}                                         
    """)
)


class LLMDoc2GraphTransformer:
    """ Converts a text to a graph """
    llm: BaseLanguageModel
    pickle_folder: str
    logger: Logger
    nodes_list: List[Node]
    edges_list: List[Relationship]
    store_to_disk: bool
    chunk_size: int
    chunk_multiplier: float
    
    def __init__(self, llm: BaseLanguageModel, pickle_folder: str, docs_nodes: Sequence[Document], docs_edges: Sequence[Document], chunk_size: int, chunk_multiplier: float, store_to_disk: Optional[bool]=False)->None:
        """ Init method of the class """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting logging in LLMDoc2GraphTransformer.")
        self.llm = llm
        self.pickle_folder = pickle_folder
        self.store_to_disk = store_to_disk
        # Create the JsonOutputParser with the Pydantic model
        self.nodeparser = JsonOutputParser(pydantic_object=Node)
        # Instantiate empty Nodes and Relationships
        self.nodes_list = []
        self.edges_list = []
        self.docs_nodes = docs_nodes # All document data
        self.docs_edges = docs_edges
        self.chunk_size = chunk_size
        self.chunk_multiplier = chunk_multiplier
    
    def _get_prompt(self, sys_prompt: str, human_prompt: str, input_vars: List[str], parser: Optional[JsonOutputParser] = None)->ChatPromptTemplate:
        if not parser:
            chat_prompt = PromptTemplate(
                template = sys_prompt + "\n\n" + human_prompt,
                input_variables=input_vars,
            )
        else:
            format_instructions = parser.get_format_instructions()
            self.logger.debug(f"Format instructions:\n{format_instructions}")
            chat_prompt = PromptTemplate(
                template = sys_prompt + "\n\n" + human_prompt,
                input_variables=input_vars,
                partial_variables={"format_instructions": format_instructions},
            )
        return chat_prompt
    
        
    def _extract_nodes_from_document(self, text: str, chunk_no: int)->List[Node]:
        """ Builds the list of nodes and appends to it. """
        raw_response = None
        self.logger.info(f"Starting nodes (vertices) extraction.")
        parser = JsonOutputParser(pydantic_object=Node)
        prompt = self._get_prompt(sys_prompt = sys_prompt_initial, human_prompt = human_prompt_initial_nodes, input_vars=["content", "nodes"], parser=parser)
        self.chain = prompt | self.llm | parser
        self.nodes_list.sort(key=lambda node: node.id)
        try:
            text_nodes_list = ""
            raw_response = self.chain.invoke(input={"content": text, "nodes": text_nodes_list}, config={'callbacks': [GraphCallBackHandler(on_llm_start=False)]})
            self.logger.info(f"{len(raw_response)} nodes were extracted.")
            self.logger.info(f"Extracted nodes:\n{json.dumps(raw_response, indent=4)}")
            try:
                if raw_response.get("nodes", None):
                    raw_response = raw_response["nodes"] # sometimes LLM sets this hierarchy...
            except Exception as e:
                pass # Nothing needed
            # Convert the list of dictionaries to a list of Node objects
            new_nodes = []
            for node_dict in raw_response:                
               new_nodes.append(Node(id=self._format_nodeid(input_string=node_dict["id"]), label=node_dict["label"], chunk=[chunk_no]))                   
            # After extracting new nodes, deduplicate the entire list
            self.nodes_list.extend(new_nodes)
            self._deduplicate_nodes()            
        except Exception as e:
            self.logger.error(f"Error during nodes extraction. Error was: {e}")
            
            
    def _deduplicate_nodes_by_llm(self):
        raw_response = None
        self.logger.info(f"Starting node deduplication and improvement by LLM.")
        parser = JsonOutputParser(pydantic_object=Node)
        prompt = self._get_prompt(sys_prompt = sys_prompt_initial, human_prompt = human_prompt_deduplicate_nodes, input_vars=["nodes"], parser=parser)
        self.chain = prompt | self.llm | parser
        self.nodes_list.sort(key=lambda node: node.id)
        try:
            dict_nodes = self._nodes_to_dict_list(nodes=self.nodes_list)
            self.logger.info(f"{len(dict_nodes)} nodes were handed to the LLM.")
            raw_response = self.chain.invoke(input={"nodes": self.nodes_list}) #changed from dict_nodes
            self.logger.info(f"{len(raw_response)} nodes were improved.")
            self.logger.debug(f"Extracted nodes:\n{raw_response}")
            # Convert the list of dictionaries to a list of Node objects
            new_nodes = [Node(**node_dict) for node_dict in raw_response]                          
            # After extracting new nodes, deduplicate the entire list
            self.nodes_list = new_nodes
            self._deduplicate_nodes()
        except Exception as e:
            self.logger.error(f"Error during nodes extraction. Error was: {e}")    
            

    def _nodes_to_dict_list(self, nodes: List[Node], chunk_no: int) -> List[Dict]:
        """Converts a Nodes object into a list of dictionaries.
        Args:
            nodes (Nodes): The Nodes object containing a list of Node objects.
            chunk_no (int): The current edge chunk #
        Returns:
            Nodes: A list of dictionaries representing the nodes.
        """
        start_chunk = math.floor(chunk_no * self.chunk_multiplier - math.ceil(self.chunk_multiplier))
        if start_chunk < 1: start_chunk = 1
        end_chunk = math.ceil(chunk_no * self.chunk_multiplier + math.ceil(self.chunk_multiplier))
        highest_value = self._find_highest_chunk_value(self.nodes_list)
        if end_chunk > highest_value: end_chunk = highest_value
        filtered_nodes = self._find_nodes_by_chunk_range(nodes=self.nodes_list, start=start_chunk, end=end_chunk)        
        return [{"id": node.id, "label": node.label} for node in filtered_nodes]
    

    def _find_highest_chunk_value(self, nodes: List[Node]) -> int:
        """Finds the highest value in the 'chunk' list of all nodes.
        
        Args:
            nodes: A list of Node objects.
            
        Returns:
            The highest value found in any 'chunk' list, or 0 if no nodes are provided.
        """
        highest_value = 0
        for node in nodes:
            highest_value = max(highest_value, max(node.chunk, default=0))  # Handles empty chunks
        return highest_value
    

    def _find_nodes_by_chunk_range(self, nodes: List[Node], start: int, end: int) -> List[Node]:
        """Finds all nodes with a chunk value within the specified range (inclusive).

        Args:
            nodes: A list of Node objects.
            start: The lower bound of the range (inclusive).
            end: The upper bound of the range (inclusive).

        Returns:
            A list of Node objects that have at least one chunk value within the range.
        """
        filtered_nodes = []
        for node in nodes:
            # Check if any element in the chunk list is within the range
            if any(start <= value <= end for value in node.chunk):
                filtered_nodes.append(node)
        return filtered_nodes
    

    def _deduplicate_nodes(self):
        """Deduplicates the nodes in the nodes_list."""
        unique_nodes = {}
        for node in self.nodes_list:
            key = (node.id)
            if key not in unique_nodes:
                unique_nodes[key] = node
            else:
                # Node already exists, so we add the chunk values to the existing node
                unique_nodes[key].chunk.extend(node.chunk)
        self.nodes_list = list(unique_nodes.values())
        # Sorts the list in place
        self.nodes_list.sort(key=lambda node: node.id)
        
    def _deduplicate_edges(self): #TODO
        """Deduplicates the nodes in the edges_list."""
        unique_edges = {}
        # for edge in self.edges_list.relationships:
        #     if edge.source not in unique_nodes:
        #         unique_nodes[node.id] = node
        # self.nodes_list.nodes = list(unique_nodes.values())
        
        
    def _extract_edges_from_document(self, text: str, chunk_no: int)->List[Relationship]:
        """ Extracts the edges from a text and considers already determined nodes """
        raw_response = None
        self.logger.info(f"Starting relationships (edges) extraction.")
        parser = JsonOutputParser(pydantic_object=Relationship)
        prompt = self._get_prompt(sys_prompt = sys_prompt_initial, human_prompt = human_prompt_initial_edges, input_vars=["content", "nodes"], parser=parser)
        self.chain = prompt | self.llm | parser
        try:
            dict_nodes = self._nodes_to_dict_list(nodes=self.nodes_list, chunk_no=chunk_no)
            raw_response = self.chain.invoke(input={"content": text, "nodes": dict_nodes}, config={'callbacks': [GraphCallBackHandler(on_llm_start=True)]})
            # Create a mapping from node IDs to Node objects for quick lookup
            node_mapping = {node.id: node for node in self.nodes_list}
            # Initialize an empty list to store valid Relationship objects
            valid_edges = []
            # Iterate over each dictionary in raw_response
            try:
                if raw_response.get("relationships", None):
                    raw_response = raw_response["relationships"] # sometimes LLM sets this hierarchy...
            except Exception as e:
                pass # not needed to do anything, it just means there is no hierarchy on top.
            self.logger.info(f"{len(raw_response)} relationships were extracted.")
            self.logger.debug(f"Extracted relationships:\n{raw_response}")
            for edge_dict in raw_response:
                source_id = self._format_nodeid(edge_dict['source'])
                target_id = self._format_nodeid(edge_dict['target'])
                # Check if both source and target are in the node_mapping
                if source_id in node_mapping and target_id in node_mapping:
                    # Both source and target exist, create a Relationship object
                    valid_edges.append(Relationship(
                        source=node_mapping[source_id],
                        target=node_mapping[target_id],
                        type=edge_dict['type'],
                        text=text #TODO: Might be good to extend the window for a later RAG?
                    ))
                else:
                    # Handle the case where source or target does not exist
                    result = self._find_node(source_id=source_id, target_id=target_id, edge_type=edge_dict['type'], chunk_no=chunk_no)
                    for edge_dict in result:
                        source_id = self._format_nodeid(edge_dict['source'])
                        target_id = self._format_nodeid(edge_dict['target'])
                        # Check if both source and target are in the node_mapping
                        if source_id in node_mapping and target_id in node_mapping:
                            # Both source and target exist, create a Relationship object
                            valid_edges.append(Relationship(
                                source=node_mapping[source_id],
                                target=node_mapping[target_id],
                                type=edge_dict['type'],
                                text=text #TODO: Might be good to extend the window for a later RAG?
                            ))
                        # Give up
                        if source_id not in node_mapping:
                            self.logger.warning(f"Source '{source_id}' does not exist in the nodes list.")
                        if target_id not in node_mapping:
                            self.logger.warning(f"Target '{target_id}' does not exist in the nodes list.")
            # Now valid_edges contains only the valid Relationship objects            
            self.edges_list.extend(valid_edges)
        except Exception as e:
            self.logger.error(f"Error during relationships extraction. Error was: {e}")


    def _find_node(self, source_id: str, target_id: str, edge_type: str, chunk_no: int)->Tuple[bool, str]:
        """ Tries to find a suitable node_id if matching doesn't work by LLM """
        node_mapping = {node.id: node for node in self.nodes_list}
        raw_response = None
        if source_id not in node_mapping:
            direction = "source"
            missing_node = source_id
        else:
            direction = "target"
            missing_node = target_id
        faulty_edge = {"source": source_id, "target": target_id, "type": edge_type}
        self.logger.info(f"Starting recovery on faulty edge: '{faulty_edge}' with missing {direction}-node '{missing_node}'")
        parser = JsonOutputParser(pydantic_object=Relationship)
        prompt = self._get_prompt(sys_prompt = sys_prompt_initial, human_prompt = human_prompt_find_node, input_vars=["content", "nodes", "extracted_edge", "node_missing"], parser=parser)
        self.chain = prompt | self.llm | parser
        try:
            dict_nodes = self._nodes_to_dict_list(nodes=self.nodes_list, chunk_no=chunk_no)
            start_chunk_no = max(0, chunk_no - 2)
            end_chunk_no = min(len(self.docs_edges) - 1, start_chunk_no + 2)
            text = ""
            for doc in self.docs_edges[start_chunk_no:end_chunk_no + 1]: # Slicing is exclusive at the end, therefore add 1
                text += doc.page_content
            raw_response = self.chain.invoke(input={"content": text, "nodes": dict_nodes, "extracted_edge": edge_type, "node_missing": missing_node}, config={'callbacks': [GraphCallBackHandler(on_llm_start=True)]})
        except Exception as e:
            self.logger.error(f"Error during relationship recovery. Error was: {e}")           
        if isinstance(raw_response, dict):
        # If the result is a dictionary, turn it into a list of one dictionary
            return [raw_response]
        elif isinstance(raw_response, list) and all(isinstance(elem, dict) for elem in raw_response):
        # If the result is already a list of dictionaries, return it as is
            return raw_response
        else:
            # If the result is neither a dictionary nor a list of dictionaries, raise an error
            self.logger.error(f"The result is neither a dictionary nor a list of dictionaries. Value is:\n{raw_response}")
            return []

        
        
        
    def _format_nodeid(self, input_string: str)->str:
        """ to ensure naming convention of node ids """
        # Convert the string to lowercase
        lower_case_string = input_string.lower()
        # Replace spaces with underscores and ' completely
        formatted_string = lower_case_string.replace(" ", "_")
        formatted_string = formatted_string.replace("'", "")
        formatted_string = formatted_string.replace("-", "_")
        return formatted_string
              
        
    def process_response(self, document: Document)->GraphDocument:
        """ Processes one document at a time and turns it into a graph (hopefully) """
        # 1. Step: Build a list of vertices (nodes)
        text = document.page_content
        self._extract_nodes_from_document(text=text)
        # 2. Step: Build a list of edges (relationships)
        edges_list = self._extract_edges_from_document(text=text)
        # 3. Step: Check for consistency
        # TODO if there's a good idea!
        # 4. Step: Send the graph of the document back to the caller
        self.edges_list.extend(edges_list)
        return GraphDocument(nodes=self.nodes_list, relationships=self.edges_list, source=document)
    
    
    def convert_to_graph_documents(self)->List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        filename = self.docs_nodes[0].metadata.get("source", "unknown")
        file_found = True
        # 1. Extract nodes or load from disk if already extracted for that file
        if self.store_to_disk:
            self.logger.info(f"Trying to load nodes pickle from file {self.pickle_folder}/{filename}.")
            try:
                with open(f'{self.pickle_folder}/{filename}_nodes.pkl', mode="rb") as f:
                    self.nodes_list = pickle.load(f)
                self.logger.info(f"Loaded {len(self.nodes_list)} nodes from file.")
            except Exception as e:
                self.logger.info(f"No file found. Extracting with LLM. Error was {e}")
                file_found = False
        if not (self.store_to_disk and file_found):
            for index, document in enumerate(self.docs_nodes, start=1):
                self.logger.info(f"Extracting nodes ({index}/{len(self.docs_nodes)}) documents.")
                self._extract_nodes_from_document(text=document.page_content, chunk_no=index)                
            if self.store_to_disk:
                self.logger.info(f"Storing nodes object for file {filename} to disk.")
                with open(f'{self.pickle_folder}/{filename}_nodes.pkl', 'wb') as f:
                    pickle.dump(self.nodes_list, f)
        # 2. Extract edges
        if self.store_to_disk:
            self.logger.info(f"Trying to load edges pickle from file {self.pickle_folder}/{filename}.")
            try:
                with open(f'{self.pickle_folder}/{filename}_edges.pkl', mode="rb") as f:
                    self.edges_list = pickle.load(f)
                self.logger.info(f"Loaded {len(self.edges_list)} relationships from file.")
            except Exception as e:
                self.logger.info(f"No file found. Extracting with LLM. Error was {e}")
                file_found = False
        if not (self.store_to_disk and file_found):
            for index, document in enumerate(self.docs_edges, start=1):
                self.logger.info(f"Extracting edges ({index}/{len(self.docs_edges)}) documents.")
                self._extract_edges_from_document(text=document.page_content, chunk_no=index)
            if self.store_to_disk:
                self.logger.info(f"Storing edges object for file {filename} to disk.")
                with open(f'{self.pickle_folder}/{filename}_edges.pkl', 'wb') as f:
                    pickle.dump(self.edges_list, f)
        doc_text = ""
        for document in self.docs_edges: # Keep it separate from the edge extraction
            doc_text += document.page_content
        summary_doc = Document(page_content=doc_text, metadata={"source": filename})        
        return [GraphDocument(nodes=self.nodes_list, relationships=self.edges_list, source=summary_doc)]
        
    