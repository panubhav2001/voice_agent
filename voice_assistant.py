from voice_transcriber import start_live_transcription
from llm_agent import extract_identity
from user_database import find_user
from voice_synthesizer import synthesize_response
from state_manager import SessionState

async def run_voice_assistant():
    session = SessionState()
    max_attempts = 2
    attempts = 0

    # 1. Greet the user
    await synthesize_response(
        "Hi, this is customer service. To continue, please tell me your first name, last name, and year of birth."
    )
    session.awaiting_identity = True

    # 2. Now loop over transcriptions
    async for sentence, mic in start_live_transcription():
        # 3. As soon as we receive input, pause mic
        mic.pause()
        print(f"You said: {sentence}")

        # 4. Check for end-call
        if "end call" in sentence.lower() or "goodbye" in sentence.lower():
            await synthesize_response("Thank you for calling. Goodbye.")
            break

        if session.awaiting_identity:
            identity = extract_identity(sentence)

            try:
                identity_data = eval(identity) if isinstance(identity, str) else identity
                keys = identity_data.keys()
                if not all(k in keys for k in ("first_name", "last_name", "year_of_birth")):
                    raise ValueError
            except Exception:
                attempts += 1
                if attempts >= max_attempts:
                    await synthesize_response(
                        "I'm having trouble verifying your identity. Try calling later"
                    )
                    break
                else:
                    await synthesize_response(
                        "I couldn't understand that. Could you repeat your full name and year of birth?"
                    )
                mic.start()  # 5. Reactivate mic after agent reply
                continue

            user = find_user(identity_data)
            if user:
                session.identity_verified = True
                session.awaiting_identity = False
                session.user = user
                await synthesize_response(
                    f"Thank you {user['first_name']} {user['last_name']}. Your current status is: {user['status']}."
                )
                break  # Or continue the conversation here
            else:
                attempts += 1
                if attempts >= max_attempts:
                    await synthesize_response(
                        "I'm sorry, I still can't verify your identity. Let's try again. Please say your full name and year of birth clearly."
                    )
                    attempts = 0
                else:
                    await synthesize_response(
                        "I couldn't verify your identity. Could you repeat your first name, last name, and year of birth?"
                    )
                mic.start()  # 5. Turn mic back on for next attempt


