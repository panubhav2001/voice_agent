import asyncio
from src.voice_transcriber import start_live_transcription
from src.voice_synthesizer import synthesize_response
from rag_pipeline.session import SessionState
from rag_pipeline.rag_chat_agent import handle_user_input
import uuid 

async def run_voice_assistant():
    session_id = f"voice_thread_{uuid.uuid4()}"
    session = SessionState(thread_id=session_id)
    print(f"Starting new conversation with Thread ID: {session.thread_id}")
    # Initial greet
    await synthesize_response(
        "Hi, this is SmileBright Dental Clinic. How can I help?"
    )

    async for text, mic in start_live_transcription():
        mic.pause()
        print(f"User said: {text}")
        clean_text = text.strip().lower()
        if any(phrase in clean_text for phrase in ["end call", "goodbye", "bye"]):
            await synthesize_response("Thank you for calling. Goodbye.")
            break

        response = await handle_user_input(text, session)
        await synthesize_response(response)

        mic.start()

if __name__ == "__main__":
    asyncio.run(run_voice_assistant())
