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
from rag_pipeline.session import SessionState


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

model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful and polite voice-based customer support assistant for SmileBright Dental Clinic. "
        "Your responses must be brief, clear, and sound like a natural conversation on a phone call.\n\n"

        "You have access to the following session state for the current user:\n"
        "- identity_verified = {identity_verified}\n"
        "- awaiting_identity = {awaiting_identity}\n\n"

        "Follow these rules:\n"
        "1. **Your TOP PRIORITY is to directly answer the user's questions** using the provided {context}. If the user asks for information (about services, hours, doctors, etc.), give a clear and concise answer. Do not ask for clarification if the question is straightforward.\n"
        
        "2. **Only clarify if necessary.** If the user's request is a single, ambiguous word (like 'appointment' or 'billing'), offer specific choices to help them.\n"

        "3. **Follow the application's state.** If `awaiting_identity` is 'True', your only job is to ask the user for their name and year of birth.\n"
        
        "4. **Be concise.** Never describe your own thought process. Do not say things like 'Let's get more specific.' Just answer the question or ask for clarification naturally.\n"
    )),
    MessagesPlaceholder(variable_name="messages"),
    ("user", (
        "Internal Knowledge Base Context:\n{context}"
    ))
])


class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    session_state: SessionState


def retrieve_context(state: RAGState) -> dict:

    last_message = state["messages"][-1]
    query = last_message.content

    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}


def call_model(state: RAGState) -> dict:
    session: SessionState = state["session_state"]
    last_msg = state["messages"][-1].content.lower()

    # Update session if user provides ID
    if not session.identity_verified and "my id is" in last_msg:
        session.identity_verified = True
        session.awaiting_identity = False

    # Prepare input for prompt
    prompt_input = {
        "messages": state["messages"],
        "context": state["context"],
        "identity_verified": str(session.identity_verified),       # Convert to string if needed
        "awaiting_identity": str(session.awaiting_identity),
        "user": session.user or "unknown",
        "pending_intent": session.pending_intent or "unknown"
    }

    # Build prompt and get model response
    prompt = prompt_template.invoke(prompt_input)
    response = model.invoke(prompt)

    return {
        "messages": [response],
        "session_state": session
    }



# Build the graph
graph = StateGraph(state_schema=RAGState)
graph.add_node("retriever", retrieve_context)
graph.add_node("model", call_model)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "model")
memory = MemorySaver()
rag_app = graph.compile(checkpointer=memory)
print('Initialization Completed.')