import asyncio
from src.voice_assistant import run_voice_assistant

if __name__ == "__main__":
    while True:
        # The 'try' block allows us to attempt running the agent
        try:
            print("\n--- Waiting for next call... ---")
            asyncio.run(run_voice_assistant())

        # The 'except' block catches the Ctrl+C command
        except KeyboardInterrupt:
            print("\nExiting application.")
            break # This breaks the 'while True' loop and ends the script cleanly

        except Exception as e:
            print(f"An error occurred: {e}. Restarting conversation loop...")