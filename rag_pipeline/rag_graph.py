import os
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
import warnings
warnings.filterwarnings("ignore")
from rag_pipeline.document import document_text

print('Initializing...')
# Load FAISS
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
FAISS_DB_PATH = "./faiss_db"
if os.path.exists(FAISS_DB_PATH):
    vectordb = FAISS.load_local(FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    documents = [Document(page_content=document_text),]
    docs = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20).split_documents(documents)
    vectordb = FAISS.from_documents(docs, embedding_model)
    vectordb.save_local(FAISS_DB_PATH)

model = init_chat_model("llama3-70b-8192", model_provider="groq")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a helpful and conversational customer support agent. "
            "Use the retrieved context to support your answer, but do not quote it directly. "
            "Address the user's question naturally, as if you're speaking to them directly in a call."
            "Be very brief and concise when explaining anything to the customer."
        )),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Here is background information you can use (do not restate it):\n{context}")
    ]
)


class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

def retrieve_context(state: RAGState) -> dict:
    last_message = state["messages"][-1]
    query = last_message.content
    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def call_model(state: RAGState) -> dict:
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

# Build the graph
graph = StateGraph(state_schema=RAGState)
graph.add_node("retriever", retrieve_context)
graph.add_node("model", call_model)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "model")
memory = MemorySaver()
rag_app = graph.compile(checkpointer=memory)
print('Initialization Completed.')