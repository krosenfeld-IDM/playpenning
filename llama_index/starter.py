"""
https://docs.llamaindex.ai/en/latest/examples/callbacks/TokenCountingHandler.html#
"""
import os
import logging
import sys
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # level=logging.INFO for less verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # read in the data as a list of llama_index.scheme.Document types: https://docs.llamaindex.ai/en/latest/understanding/loading/loading.html
    documents = SimpleDirectoryReader("data").load_data()
    # create an index, query OpenAI: https://docs.llamaindex.ai/en/latest/understanding/indexing/indexing.html
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)