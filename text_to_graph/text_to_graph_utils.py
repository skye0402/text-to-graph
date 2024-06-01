from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction

from langchain_community.graphs.graph_document import GraphDocument

from hana_ml import ConnectionContext
from hdbcli.dbapi import Cursor

from typing import Union, List, Dict, Any, Optional
import logging, json, pickle
from logging import Logger
from textwrap import dedent

class GraphCallBackHandler(BaseCallbackHandler):
    """ Handling of callbacks for Graph use case """
    
    def __init__(self, 
                 on_llm_start: Optional[bool]=False, 
                 on_chain_start: Optional[bool]=False, 
                 on_tool_start: Optional[bool]=False, 
                 on_agent_action: Optional[bool]=False, 
                 on_llm_new_token: Optional[bool]=False,
                 on_llm_error: Optional[bool]=True):
        self.logger = logging.getLogger(__name__)
        self.flag_on_llm_start = on_llm_start
        self.flag_on_chain_start = on_chain_start
        self.flag_on_tool_start = on_tool_start
        self.flag_on_agent_action = on_agent_action
        self.flag_on_llm_new_token = on_llm_new_token
        self.flag_on_llm_error = on_llm_error
        self.logger.info("Logging for Graph Callbacks active.")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        if self.flag_on_llm_start:
            self.logger.info(f"on_llm_start {serialized['name']}\nPrompts:\n{json.dumps(prompts, indent=4, )}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        if self.flag_on_llm_new_token:
            self.logger.info(f"on_new_token {token}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when LLM errors."""
        if self.flag_on_llm_error:
            self.logger.error(f"on_llm_error {error}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        if self.flag_on_chain_start:
            self.logger.info(f"on_chain_start {serialized['name']}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        if self.flag_on_tool_start:
            self.logger.info(f"on_tool_start {serialized['name']}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        if self.flag_on_agent_action:
            self.logger.info(f"on_agent_action {action}")
            
class DbHandlingForGraph:
    """ 
    Class to handle all kinds of stuff for HANA Graph 
    
    Attributes:    
        logger (Logger): To log all kinds of things.
        conn_params (dict): HANA Connection parameters.
        schema (str): The schema to be used. If nothing is given it's likely the user name and we don't set it. 
    
    """    
    logger: Logger
    conn_params: dict
    schema: str
    
    def __init__(self, logger: Logger, conn_params: dict, table_names: dict, text_length: Optional[int]=1000, schema: Optional[str]=None):
        self.logger = logger
        self.conn_params = conn_params
        self.conn_context: ConnectionContext = None
        self.db_cursor: Cursor = None
        self.t_names = table_names
        self.text_length = text_length
        if not schema:
            self.schema = ""
        else:
            self.schema = schema
    
    
    def get_hana_connection(self)->bool:
        """ Connect to HANA Cloud """
        try:
            self.conn_context = ConnectionContext(
                address = self.conn_params["host"],
                port = 443,
                user = self.conn_params["user"],
                password= self.conn_params["password"],
                encrypt= True
            )    
            self.logger.info(f"HANA Version: {self.conn_context.hana_version()}")
            self.logger.info(f"Current default schema: {self.conn_context.get_current_schema()}")
            self.db_cursor = self.conn_context.connection.cursor()
        except Exception as e:
            self.logger.error(f'Error when opening connection to HANA Cloud DB with host: {self.conn_params["host"]}, user {self.conn_params["user"]}. Error was:\n{e}')
            return False
        finally:    
            return True
        
        
    def create_graph_tables(self):
        if self.schema != "":
            schema_script = dedent(
                f"""
                DROP SCHEMA "{self.schema}" CASCADE;
                CREATE SCHEMA "{self.schema}";
                SET SCHEMA "{self.schema}";
                """
            )
        else:
            schema_script = ""
        script_vertex = dedent(f"""
            CREATE COLUMN TABLE "{self.t_names['v']}" (
                "ID" BIGINT PRIMARY KEY,
                "NAME" VARCHAR(100),
                "LABEL" VARCHAR(100)
            );""")
        script_edge = dedent(f"""
            CREATE COLUMN TABLE "{self.t_names['e']}" (
                "ID" BIGINT PRIMARY KEY,
                "SOURCE" BIGINT REFERENCES "{self.t_names['v']}"("ID") ON UPDATE CASCADE ON DELETE CASCADE NOT NULL,
                "SOURCE_LABEL" VARCHAR(100),
                "TARGET" BIGINT REFERENCES "{self.t_names['v']}"("ID") ON UPDATE CASCADE ON DELETE CASCADE NOT NULL,
                "TARGET_LABEL" VARCHAR(100),
                "EDGE_LABEL" VARCHAR(100),
                "TEXT_ARTIFACT" VARCHAR({self.text_length})
            );
        """)
        try:
            sql_drop = f'DROP GRAPH WORKSPACE "{self.t_names['g']}";'
            self.db_cursor.execute(sql_drop)
            self.logger.info(f"Dropped graph workspace '{self.t_names['g']}'.")
        except Exception as e:
            self.logger.info(f"Graph workspace {self.t_names["e"]} not dropped. Maybe didn't exist. Error was {e}")
        try:
            sql_drop = f'DROP TABLE "{self.t_names["e"]}";'
            self.db_cursor.execute(sql_drop)
            self.logger.info(f"Dropped table '{self.t_names['e']}'.")
        except Exception as e:
            self.logger.info(f"Table {self.t_names["e"]} not dropped. Maybe didn't exist. Error was {e}")
        try:
            sql_drop = f'DROP TABLE "{self.t_names["v"]}";'
            self.db_cursor.execute(sql_drop)
            self.logger.info(f"Dropped table '{self.t_names['v']}'.")
        except Exception as e:
            self.logger.info(f"Table {self.t_names["v"]} not dropped. Maybe didn't exist. Error was {e}")
        try:
            self.db_cursor.execute(script_vertex)
            self.db_cursor.execute(script_edge)
            self.logger.info(f"Created tables '{self.t_names['v']}', '{self.t_names['e']}'.")
            return True
        except Exception as e:
            self.logger.error(f"Error creating graph tables. Error was {e}")
            return False
    
    def load_graph_data_to_table(self, filename: str)->bool:
        """ Load document into HANA Cloud """       
        graph_docs = self._load_graph_documents(filename=filename)
        v_schema = ""
        if self.schema:
            v_schema = f'"{self.schema}".'
        for grdoc in graph_docs:
            try:
                sql = f"""INSERT INTO {v_schema}"{self.t_names["v"]}" ("ID", "NAME", "LABEL") VALUES (?, ?, ?)"""
                nodes_list = []
                for key, node in enumerate(grdoc.nodes):
                    node.id = node.id.replace("'", "´")
                    nodes_list.append((key, node.id, node.label))
                self.db_cursor.executemany(sql, tuple(nodes_list))
                sql = f"""INSERT INTO {v_schema}"{self.t_names["e"]}" ("ID", "SOURCE", "SOURCE_LABEL", "TARGET", "TARGET_LABEL", "EDGE_LABEL", "TEXT_ARTIFACT") VALUES (?, ?, ?, ?, ?, ?, ?);"""
                edges_list = []
                for key, edge in enumerate(grdoc.relationships):
                    key_source = self._find_node_index(nodes=grdoc.nodes, node_id=edge.source.id)
                    if key_source == -1:
                        self.logger.error(f"Source Key was not found: {key_source}")
                    key_target = self._find_node_index(nodes=grdoc.nodes, node_id=edge.target.id)
                    if key_target == -1:
                        self.logger.error(f"Target Key was not found: {key_target}")
                    escaped_text = edge.text
                    edges_list.append((key, key_source, edge.source.id, key_target, edge.target.id, edge.type, escaped_text))
                self.db_cursor.executemany(sql, tuple(edges_list))
                return True
            except Exception as e:
                self.logger.error(f"Error when inserting data. Statement was \n{sql}\nError was: {e}")
                return False
    
    
    def create_graph_workspace(self):
        """ Create Graph Workspace on HANA - this is a mandatory step to work with Graphs/ OpenCypher"""
        sql = dedent(f"""
            CREATE GRAPH WORKSPACE {self.schema}"{self.t_names['g']}"
                EDGE TABLE {self.schema}"{self.t_names['e']}"
                    SOURCE COLUMN "SOURCE"
                    TARGET COLUMN "TARGET"
                    KEY COLUMN "ID"
                VERTEX TABLE {self.schema}"{self.t_names['v']}" 
                    KEY COLUMN "ID";
        """
        )
        try:   
            self.db_cursor.execute(sql)
            self.logger.info(f"Graph workspace {self.t_names['g']} created.")
        except Exception as e:
            self.logger.error(f"Error creating graph workspace. Error was {e}")
        
    
    def _find_node_index(self, nodes, node_id):
        """ Find the index of a node in a list of nodes """
        for index, node in enumerate(nodes):
            if node.id == node_id:
                return index
        return -1  # Not found
    

    def _load_graph_documents(self, filename: str)->List[GraphDocument]:
        """ Load the graph_documents object from disk """
        with open(file=filename, mode="rb") as f:
            loaded_graph_documents = pickle.load(f)
        return loaded_graph_documents