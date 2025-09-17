from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class SessionState(BaseModel):
    """
    Represents the state of a user session, including multi-turn logic for booking.
    """
    # Existing fields
    thread_id: str
    awaiting_identity: bool = False
    identity_verified: bool = False
    user: Optional[Any] = None
    pending_intent: Optional[str] = None
    chat_history: List[Any] = Field(default_factory=list)

    # New fields for the booking flow
    awaiting_booking_details: bool = False
    pending_booking_details: Optional[Dict[str, Any]] = None