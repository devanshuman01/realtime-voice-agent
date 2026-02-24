#Microphone → Raw bytes
#Raw bytes → int16
#int16 → normalized float32
#float32 → torch tensor
#Tensor → VAD model/


import torch
import numpy as np

def load_vad_model():
    print("Loading local VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)
    return model

def prepare_audio_tensor(indata):
    audio_int16 = np.frombuffer(indata, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return torch.from_numpy(audio_float32)

#silero vad is a voice activity detection model.
#Detects: speech input, silence, background noise.

