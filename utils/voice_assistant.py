import asyncio
from utils.voice_transcriber import start_live_transcription
from utils.voice_synthesizer import synthesize_response
from rag_pipeline.session import SessionState
from rag_pipeline.rag_chat_agent import handle_user_input

async def run_voice_assistant():
    session = SessionState(thread_id="voice_thread_001")

    # Initial greet
    await synthesize_response(
        "Hi, this is SmileBright Dental Clinic. How can I help?"
    )

    async for text, mic in start_live_transcription():
        mic.pause()
        print(f"User said: {text}")

        if any(phrase in text.lower() for phrase in ["end call", "goodbye", "bye"]):
            await synthesize_response("Thank you for calling. Goodbye.")
            break

        response = await handle_user_input(text, session)
        await synthesize_response(response)

        mic.start()

if __name__ == "__main__":
    asyncio.run(run_voice_assistant())
