from rag_pipeline.session import SessionState
from langchain_core.messages import HumanMessage, SystemMessage
from rag_pipeline.rag_graph import rag_app, FAISS_DB_PATH, vectordb, model
from langchain_core.documents import Document
from datetime import datetime
import uuid

async def store_summary_to_faiss(session: SessionState):
    conversation_text = "\n".join(
        f"{'User' if msg.type == 'human' else 'AI'}: {msg.content.strip()}"
        for msg in session.chat_history
    )

    summary_response = rag_app.invoke(
        {
            "messages": [
                SystemMessage(content="Summarize the following conversation briefly and clearly."),
                HumanMessage(content=conversation_text)
            ],
            "session_state": session # <-- Add this line
        },
        config={"configurable": {"thread_id": "summary_" + session.thread_id}}
    )
    summary = summary_response["messages"][-1].content.strip()
    doc = Document(
        page_content=summary,
        metadata={"timestamp": datetime.utcnow().isoformat(), "thread_id": session.thread_id}
    )

    vectordb.add_documents([doc], ids=[str(uuid.uuid4())])
    vectordb.save_local(FAISS_DB_PATH)

def validate_time_with_rag(details: dict) -> str:
    """
    Uses RAG to validate if a requested time is within the clinic's open hours.
    """
    # Use an LLM to validate the requested time against the hours
    validation_prompt = f"""
    Context: The clinic's hours are: Monday to Saturay from 9 AM to 6 PM."
    User's requested appointment time: {details.get('day')} at {details.get('time')}.
    
    Based ONLY on the provided context, is the requested time within the clinic's opening hours? 
    Answer with a single word: YES or NO.
    """
    validation_response = model.invoke(validation_prompt)
    return validation_response.content.strip().upper()