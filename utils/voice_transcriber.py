import asyncio
import os
import logging
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

# Suppress verbose asyncio debug logs
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Transcript manager
class TranscriptCollector:
    def __init__(self):
        self.transcript_parts = []

    def add_part(self, part: str):
        self.transcript_parts.append(part)

    def get_full_transcript(self) -> str:
        return ' '.join(self.transcript_parts)

    def reset(self):
        self.transcript_parts = []

class ControlledMicrophone:
    def __init__(self, send_function):
        self._mic = Microphone(send_function)
        self._is_active = False

    def start(self):
        if not self._is_active:
            self._mic.start()
            self._is_active = True

    def pause(self):
        if self._is_active:
            self._mic.finish()
            self._is_active = False

    def is_active(self):
        return self._is_active


async def start_live_transcription():
    transcript_collector = TranscriptCollector()
    queue = asyncio.Queue()

    config = DeepgramClientOptions(options={"keepalive": "true"})
    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)
    dg_connection = deepgram.listen.asynclive.v("1")

    async def on_transcript(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if sentence:
            transcript_collector.add_part(sentence)
            if result.speech_final:
                full_sentence = transcript_collector.get_full_transcript()
                await queue.put(full_sentence)
                transcript_collector.reset()

    async def on_error(self, error, **kwargs):
        print(f"Error: {error}")

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)

    options = LiveOptions(
        model="nova-3-general",
        punctuate=True,
        language="en-US",
        encoding="linear16",
        channels=1,
        sample_rate=16000,
        endpointing=2,
        vad_events=True,
        numerals=True
    )
    await dg_connection.start(options)

    mic = ControlledMicrophone(dg_connection.send)
    mic.start()

    try:
        while True:
            sentence = await queue.get()
            yield sentence, mic  # also yield mic so it can be paused externally
    finally:
        mic.pause()
        await dg_connection.finish()


