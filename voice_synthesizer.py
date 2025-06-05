import asyncio
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    SpeakWebSocketEvents,
    SpeakWSOptions
)
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


async def synthesize_response(text: str):
    try:
        # Create Deepgram client inside the async function (important!)
        config = DeepgramClientOptions(
            options={
                "speaker_playback": "true"
            }
        )
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)

        # Create async WebSocket connection
        dg_connection = deepgram.speak.asyncwebsocket.v("1")

        # Register error handler only
        async def on_error(self, error, **kwargs):
            print(f"❌ Error: {error}")

        dg_connection.on(SpeakWebSocketEvents.Error, on_error)

        # Setup WebSocket TTS options
        options = SpeakWSOptions(
            model="aura-2-thalia-en",
            encoding="linear16",
            sample_rate=16000
        )

        # Start the TTS WebSocket connection
        started = await dg_connection.start(options)
        if not started:
            print("❌ Failed to start the TTS connection.")
            return

        print(f"Agent: {text}")  # ✅ Display agent statement

        await dg_connection.send_text(text)
        await dg_connection.flush()
        await dg_connection.wait_for_complete()
        await dg_connection.finish()


    except Exception as e:
        print(f"Error during TTS synthesis or playback: {e}")
