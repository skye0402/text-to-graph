import logging, os
from text_to_graph.text_to_graph_utils import DbHandlingForGraph 
   
def main()->None:
    args = {}
    args["host"] = os.environ.get("HOSTNAME","0.0.0.0")
    args["port"] = os.environ.get("HOSTPORT",51030)
    log_level = int(os.environ.get("APPLOGLEVEL", logging.ERROR))    
    if log_level < 10: log_level = 20
    logging.basicConfig(level=log_level,)
    logger = logging.Logger(name=__name__)
    hana_cloud = {
        "host": os.getenv("HOST"),
        "user": os.getenv("USERNAME",""),
        "password": os.getenv("PASSWORD","") 
    }
    
    # Set table names
    graph_workspace = "GOLDEN_GOOSE_GWS"
    vertices_table_name = "GOLDEN_GOOSE_VERTICES"
    edges_table_name = "GOLDEN_GOOSE_EDGES"   
    
    do_drop_create_fill = False
    
    # Graph file to be loaded
    graph_filename = "The Golden Goose.txt_graph.pkl" 
    tables = {
        "v": vertices_table_name,
        "e": edges_table_name,
        "g": graph_workspace
    }
    
    gdb_handler = DbHandlingForGraph(logger=logger, conn_params=hana_cloud, table_names=tables)
    # Connect to HANA
    if not gdb_handler.get_hana_connection():
        exit(0)
    if do_drop_create_fill:
        # Create vertices and edges tables
        if not gdb_handler.create_graph_tables():
            exit(0)
        # Save data in tables
        if not gdb_handler.load_graph_data_to_table(filename=graph_filename):
            exit(0)
    # Create graph workspace
    if not gdb_handler.create_graph_workspace():
        exit(0)
        
    
if __name__ == "__main__":
    main()