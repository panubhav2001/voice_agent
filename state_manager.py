class SessionState:
    def __init__(self):
        self.awaiting_identity = True
        self.identity_verified = False
        self.user = None