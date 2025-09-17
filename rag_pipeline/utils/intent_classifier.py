import spacy

# Load the spacy model once when your application starts
nlp = spacy.load("en_core_web_sm")

# Use sets for very fast lookups. Keywords are in their root form (lemma).
NEGATIVE_KEYWORDS = {"not", "cancel", "reschedule", "change", "don't"}

STATUS_ACTIONS = {"status", "check", "when", "confirm", "where", "what's"}
STATUS_OBJECTS = {"appointment", "booking", "visit", "time", "schedule"}

NEW_BOOKING_ACTIONS = {"book", "schedule", "make", "create", "set", "get", "want", "need"}
NEW_BOOKING_OBJECTS = {"appointment", "booking", "visit", "time"}

def classify_intent(text: str) -> str:
    """
    Classifies intent using lemmatization and logical keyword groups.
    """
    # Process the text to get tokens and their lemmas
    doc = nlp(text.lower())
    lemmas = {token.lemma_ for token in doc}

    # 1. Check for negative keywords first to override other intents
    if NEGATIVE_KEYWORDS & lemmas: # '&' checks for intersection between sets
        # Could be a "cancellation" intent, but for now, we'll treat as general
        return "general"

    # 2. Check for booking status intent (requires an action and an object)
    if STATUS_ACTIONS & lemmas and STATUS_OBJECTS & lemmas:
        return "booking_status"

    # 3. Check for new booking intent (requires an action and an object)
    if NEW_BOOKING_ACTIONS & lemmas and NEW_BOOKING_OBJECTS & lemmas:
        return "new_booking"

    # 4. If no specific intent is found, default to general
    return "general"