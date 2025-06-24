# 🗣️ Voice Assistant for Customer Service

A real-time voice-enabled assistant that understands speech, verifies identity, handles booking-related queries, and speaks natural responses using AI—ideal for modern customer service scenarios.

---

## ✅ Features

- 👋 Friendly, human-like greeting and tone
- 🔐 Extracts and verifies identity from name and year of birth
- 📅 Checks current booking status if asked
- 📝 Handles new booking requests
- 💬 Answers general service-related queries using AI
- 🧠 Uses lightweight intent detection to decide what action to take
- 🧾 Summarizes and stores each session to FAISS DB
- 🔊 Speaks out responses with realistic voice synthesis
- 🎙️ Pauses and resumes mic automatically for smooth flow

---

## 🛠️ Tech Stack

| Component            | Tech Used                      |
|----------------------|--------------------------------|
| Speech to Text       | Deepgram Live `listen` API     |
| Text to Speech       | Deepgram `speak` API           |
| LLM Processing       | Groq LLaMA 3.1                 |
| Session Logic        | Python + `SessionState` object |
| RAG Contextual QA    | LangGraph + FAISS DB           |
| Identity Extraction  | Groq Prompt + JSON parsing     |
| Intent Detection     | Custom keyword-based logic     |
| Storage              | FAISS vectorDB                 |

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone 
cd voice-assistant
```
### 2. Install Dependencies
Requires Python 3.9 or higher:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a .env file in the root directory:
```bash
DEEPGRAM_API_KEY=your_deepgram_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. How to RUN
Start the assistant with:
```bash
python voice_assistant.py
```

## Example Conversation
👤 User: Hi, I want to book a service.
🤖 Assistant: I can help with that. Please tell me your first name, last name, and year of birth.

👤 User: Jane Doe, born in 1990.
🤖 Assistant: Thanks Jane. I’ve noted your request to schedule a new booking. Someone from our team will follow up shortly.

👤 User: When is my next appointment?
🤖 Assistant: Thank you Jane Doe. Your current status is: Your service is scheduled for Wednesday at 10 AM.

👤 User: What services do you offer?
🤖 Assistant: We provide maintenance, inspections, oil changes, and more. How can I assist you today?
