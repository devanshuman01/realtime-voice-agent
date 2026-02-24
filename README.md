Real-Time Conversational AI Voice Agent 🎙️🤖
An ultra-low latency, voice-to-voice AI assistant built in Python. This system features local voice activity detection (VAD), cloud-based speech-to-text, LLM processing, and text-to-speech generation.

Built to minimize conversational delays and mimic natural human interaction.

🏗️ System Architecture
The pipeline is split into three modular components (Ears, Brain, and Mouth) managed by a central orchestrator:

VAD (Silence Detection): silero-vad continuously monitors local microphone input. It buffers audio until a natural pause in speech is detected.

Ears (STT): Deepgram's Nova-3 model processes the audio buffer into text instantly.

Brain (LLM): Groq's API utilizing the llama-3.1-8b-instant model generates conversational responses with sub-second token delivery.

Mouth (TTS): Microsoft edge-tts synthesizes the text back into human-like speech.

⚙️ Features
Real-Time Voice Activity Detection: Never sends continuous background noise to the cloud; only processes intentional speech.

Auto-Muting: System state management ignores microphone input while the AI is speaking to prevent echo-loops.

Modular Design: Clean separation of hardware processing, API connections, and the main loop.