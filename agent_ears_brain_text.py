#Deepgram takes our full audio file through API and
#sends a transcript text to Groq for prompting.abs

#Groq takes the transcribed text and sends the responsive text
#to microsoft Edge -TTS.

#Microsoft Edge -TTS generates an audio file of that text.

#At last our pc plays the audio through the speakers.

import os
import io
import asyncio
import edge_tts
from deepgram import DeepgramClient 
from groq import Groq
import scipy.io.wavfile as wav
import soundfile as sf
import sounddevice as sd
from dotenv import load_dotenv 

load_dotenv() 

deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ears_transcribe(audio_buffer):
    print("Transcribing audio...")
    buffer_bytes = io.BytesIO()
    wav.write(buffer_bytes, 16000, audio_buffer)
    buffer_bytes.seek(0)
    
    response = deepgram.listen.v1.media.transcribe_file(
        request=buffer_bytes.read(),
        model="nova-3",
        smart_format=True
    )
    
    transcript = response.results.channels[0].alternatives[0].transcript
    return transcript

def brain_think(user_text):
    print(f"Prompting LLM: {user_text}")
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful data science assistant. Keep answers brief, conversational, and under 2 sentences."},
            {"role": "user", "content": user_text},
        ],
        model="llama-3.1-8b-instant", 
    )
    return chat_completion.choices[0].message.content

async def mouth_speak(text):
    print(f"Generating Speech: {text}")
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    
    await communicate.save("response.mp3")
    
    data, fs = sf.read("response.mp3")
    sd.play(data, fs)
    sd.wait()