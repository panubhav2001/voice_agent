MOCK_USERS = [
    {"first_name": "Jane", "last_name": "Doe", "year_of_birth": 1988, "status": "Your service is scheduled for Tuesday at 2 PM"},
    {"first_name": "Anubhav", "last_name": "Prasad", "year_of_birth": 2000, "status": "Your service is scheduled for Wednesday at 8 AM"}
]

def find_user(identity):
    for user in MOCK_USERS:
        if all(user.get(k) == identity.get(k) for k in ["first_name", "last_name", "year_of_birth"]):
            return user
    return None