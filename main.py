import asyncio
import queue
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

from audio_vad_model import load_vad_model, prepare_audio_tensor
from agent_ears_brain_text import ears_transcribe, brain_think, mouth_speak

def main():
    print("--- Real-Time Voice Agent Initializing ---")
    
    vad_model = load_vad_model()
    audio_queue = queue.Queue()
    
    state = {"is_ai_speaking": False}
    
    user_is_speaking = False
    silence_counter = 0
    buffer = []
    
    def callback(indata, frames, time, status):
        if not state["is_ai_speaking"]:
            audio_queue.put(indata.copy())

    print("System Ready. Listening... (Press Ctrl+C to stop)")
    
    with sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=512, dtype='int16'):
        while True:
            indata = audio_queue.get()
            
            tensor = prepare_audio_tensor(indata)
            speech_prob = vad_model(tensor, 16000).item()
            
            if speech_prob > 0.5:
                user_is_speaking = True
                silence_counter = 0
                buffer.append(indata)
                print("🔹", end='', flush=True) 
            else:
                if user_is_speaking:
                    silence_counter += 1
                    buffer.append(indata)
                    
                    if silence_counter > 30:
                        print("\nProcessing pipeline...")
                        full_audio = np.concatenate(buffer)
                        
                        transcript = ears_transcribe(full_audio)
                        
                        print(f"I heard: '{transcript}'")
                        # ------------------------------------------------

                        if transcript.strip():
                            
                            clean_text = transcript.lower()
                            if "goodbye" in clean_text or "good bye" in clean_text or "stop" in clean_text or "exit" in clean_text:
                                print("🛑 Exit command received. Shutting down...")
                                
                                state["is_ai_speaking"] = True 
                                asyncio.run(mouth_speak("Shutting down. Have a great day!"))
                                
                                break  

                            response = brain_think(transcript)
                            
                           
                            state["is_ai_speaking"] = True
                            
                            asyncio.run(mouth_speak(response))
                            
                            while not audio_queue.empty():
                                audio_queue.get()
                                
                            state["is_ai_speaking"] = False
                        
                        buffer = []
                        user_is_speaking = False
                        silence_counter = 0
                        print("✅ System Ready. Listening...")

if __name__ == "__main__":
    main()