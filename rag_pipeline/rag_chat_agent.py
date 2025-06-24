from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from datetime import datetime
import uuid
import ast

from rag_pipeline.session import SessionState
from utils.llm_agent import extract_identity
from utils.user_database import find_user
from rag_pipeline.rag_graph import rag_app, FAISS_DB_PATH, vectordb

BOOKING_STATUS_KEYWORDS = [
    "booking status", "appointment status", "scheduled", "confirmed",
    "next booking", "next appointment", "do i have a booking", "when is my booking",
    "is my service still scheduled"
]

NEW_BOOKING_KEYWORDS = [
    "make a booking", "book an appointment", "schedule service",
    "create a booking", "i want to book", "can i book"
]

def classify_intent(text: str) -> str:
    text = text.lower()
    if any(k in text for k in BOOKING_STATUS_KEYWORDS):
        return "booking_status"
    if any(k in text for k in NEW_BOOKING_KEYWORDS):
        return "new_booking"
    return "general"

async def handle_user_input(text: str, session: SessionState) -> str:
    session.chat_history.append(HumanMessage(content=text))

    # Step 1: If we're waiting for identity
    if session.awaiting_identity and not session.identity_verified:
        identity = extract_identity(text)
        print(identity)
        print(type(identity))
        if isinstance(identity, str):
            identity = ast.literal_eval(identity)
        elif not isinstance(identity, dict) or not all(k in identity for k in ("first_name", "last_name", "year_of_birth")):
            return "Please provide your full name and year of birth so I can verify you."

        user = identity
        if user:
            session.identity_verified = True
            session.awaiting_identity = False
            session.user = user['first_name'] + ' ' + user['last_name']

            # Respond based on stored intent
            if session.pending_intent == "booking_status":
                msg = f"Thank you {user['first_name']} {user['last_name']}. Your current status is: {user['status']}."
            else:  # new_booking
                msg = f"Thanks {user['first_name']}. I’ve noted your request to schedule a new booking. Someone from our team will follow up shortly."
            
            session.chat_history.append(SystemMessage(content=msg))
            return msg
        else:
            return "I couldn't verify your identity. Please try again with your full name and year of birth."

    # Step 2: Normal flow
    intent = classify_intent(text)

    if intent in {"booking_status", "new_booking"}:
        if not session.identity_verified:
            session.awaiting_identity = True
            session.pending_intent = intent  # Store for next round
            return "To help you with that, could you please tell me your full name and year of birth?"

        # Identity already verified
        user = session.user
        if intent == "booking_status":
            msg = f"Thank you {user['first_name']} {user['last_name']}. Your current status is: {user['status']}."
        else:  # new_booking
            msg = f"Thanks {user['first_name']}. I’ve noted your request to schedule a new booking. Someone from our team will follow up shortly."

        session.chat_history.append(SystemMessage(content=msg))
        return msg

    # Step 3: General queries → fallback to RAG
    response = rag_app.invoke(
        {"messages": session.chat_history, "context": ""},
        config={"configurable": {"thread_id": session.thread_id}}
    )
    ai_msg = response["messages"][-1]
    session.chat_history.append(ai_msg)

    await store_summary_to_faiss(session)
    return ai_msg.content

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
            ]
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
