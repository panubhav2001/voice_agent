class SessionState:
    def __init__(self, thread_id: str = "default_thread"):
        self.thread_id = thread_id
        self.awaiting_identity = False
        self.identity_verified = False
        self.user = None
        self.pending_intent = None
        self.chat_history = []
