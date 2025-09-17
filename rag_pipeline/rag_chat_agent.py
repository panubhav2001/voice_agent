import json
from langchain_core.messages import HumanMessage, SystemMessage

from rag_pipeline.session import SessionState
from src.llm_agent import extract_identity, extract_booking_details
from rag_pipeline.rag_graph import rag_app
from rag_pipeline.utils.intent_classifier import classify_intent
from rag_pipeline.utils.misc import store_summary_to_faiss, validate_time_with_rag

async def handle_user_input(text: str, session: SessionState) -> str:
    """
    Handles user input by managing a multi-turn conversation for different intents.
    """
    session.chat_history.append(HumanMessage(content=text))

    # --- BLOCK 1: Handles the turn AFTER the agent asks for identity ---
    # This block has the highest priority because it's a direct response to a question.
    if session.awaiting_identity and not session.identity_verified:
        identity_str = extract_identity(text)
        user_details = json.loads(identity_str)

        if user_details.get("first_name") and user_details.get("year_of_birth"):
            session.identity_verified = True
            session.awaiting_identity = False
            session.user = user_details

            # Check if we were in the middle of a booking
            if session.pending_booking_details:
                booking_info = session.pending_booking_details
                session.pending_booking_details = None # Clear the pending info
                # This is the final confirmation message for the booking flow
                return f"Perfect, thank you. You're all set for your appointment on {booking_info['day']} at {booking_info['time']}. We'll see you then!"
            else:
                # Handle other intents that required verification (e.g., booking_status)
                return "Thank you for verifying your identity. How can I now help you with your booking status?"
        else:
            return "I'm sorry, I didn't catch your full name and year of birth. Could you please provide them again?"

    # --- BLOCK 2: Handles the turn AFTER the agent asks for a time slot ---
    if session.awaiting_booking_details:

        details_str = extract_booking_details(text)
        details = json.loads(details_str)

        if not details.get("day") or not details.get("time"):
            return "I'm sorry, I didn't quite catch that. Could you please provide a day and time for the appointment?"

        # Validate the requested time against the RAG knowledge base
        is_open = validate_time_with_rag(details)
        if "YES" in is_open:
            session.awaiting_booking_details = False # Turn this flag off
            session.pending_booking_details = details # Store the validated time
            
            # Now, transition to asking for identity to finalize
            session.awaiting_identity = True 
            return "Great, that time is available. To finalize the booking for you, could you please tell me your full name and year of birth?"
        else:
            # Re-prompt the user if the time is invalid, keeping the flag on
            return "It looks like we're closed at that time. Please choose a different time."

    # --- BLOCK 3: This is the first entry point for classifying new intents ---
    intent = classify_intent(text)

    if intent == "new_booking":
        session.awaiting_booking_details = True
        return "Of course. What day and time would you like to schedule your appointment for?"
        
    if intent == "booking_status":
        session.awaiting_identity = True
        return "Certainly. To check your booking status, I'll need to verify your identity. What is your full name and year of birth?"

    # --- BLOCK 4: Fallback to general RAG for all other queries ---
    else:
        response = rag_app.invoke(
            {"messages": session.chat_history, "session_state": session},
            config={"configurable": {"thread_id": session.thread_id}}
        )
        ai_msg = response["messages"][-1]
        session.chat_history.append(ai_msg)
        await store_summary_to_faiss(session)
        return ai_msg.content