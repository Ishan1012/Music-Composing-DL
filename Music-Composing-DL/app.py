import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pretty_midi
import io
from scipy.io import wavfile
from music21 import stream, chord

class CasualConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size-1)*dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding
        )

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class Model(nn.Module):
    def __init__(self, num_notes):
        super().__init__()

        self.embedding = nn.Embedding(num_notes, 20)

        self.conv1 = CasualConv1D(4*5, 32, kernel_size=2, dilation=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = CasualConv1D(32, 48, kernel_size=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(48)

        self.conv3 = CasualConv1D(48, 64, kernel_size=2, dilation=4)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = CasualConv1D(64, 96, kernel_size=2, dilation=8)
        self.bn4 = nn.BatchNorm1d(96)

        self.conv5 = CasualConv1D(96, 128, kernel_size=2, dilation=16)
        self.bn5 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.05)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Linear(256, num_notes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)   

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        x = self.fc(x)
        return x

@st.cache_resource
def load_model(model_path, num_notes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(num_notes=num_notes)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    return model, device

def sample_next_note(probs):
    probabilities = np.asarray(probs, dtype=float)
    
    probs_sum = probabilities.sum()

    if probs_sum <= 0 or not np.isfinite(probs_sum):
        return int(np.argmax(probabilities))

    probabilities /= probs_sum
    return np.random.choice(len(probabilities), p=probabilities)

def generate_chorale(model, seed_chords, length, device="cpu"):
    token_sequence = np.array(seed_chords, dtype=int)
    token_sequence = np.where(
        token_sequence == 0,
        token_sequence,
        token_sequence - 36 + 1
    )

    token_sequence = torch.LongTensor(token_sequence).to(device)
    token_sequence = token_sequence.view(1, -1) 
    progress_bar = st.progress(0)

    for i in range(length * 4):
        with torch.no_grad():
            logits = model(token_sequence)          
            
            last_logits = logits[:, -1, :]          
            probs = F.softmax(last_logits, dim=-1)
            probs = probs.squeeze(0).cpu().numpy()

        next_token_idx = sample_next_note(probs)
        next_token_tensor = torch.LongTensor([[next_token_idx]]).to(device)
        token_sequence = torch.cat([token_sequence, next_token_tensor], dim=1)
        progress_bar.progress((i + 1) / (length * 4))

    token_sequence = token_sequence.cpu().numpy()
    token_sequence = np.where(
        token_sequence == 0,
        token_sequence,
        token_sequence + 36 - 1
    )

    return token_sequence.reshape(-1, 4)

st.title("AI Bach Chorale Generator")
st.subheader("Generate music using a Conv1D-LSTM Model")

st.sidebar.header("Generation Settings")
gen_length = st.sidebar.slider("Length (Chords)", min_value=16, max_value=128, value=56)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.5, 1.5, 1.0)

seed_chords = [[73, 68, 61, 53]] * 8 

if st.button("Generate New Music"):
    model, device = load_model("./Music-Composing-DL/weights.pth", num_notes=47)
    
    with st.spinner("Writing the notes..."):
        generated_notes = generate_chorale(model, seed_chords, gen_length, device)
    
    st.success("Generation Complete!")

    with st.spinner("Preparing MIDI..."):
        s = stream.Stream()
        for row in generated_notes.tolist():
            pitches = [n for n in row if n > 0]
            if pitches:
                s.append(chord.Chord(pitches, quarterLength=1))

        import tempfile, os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
            midi_path = tmp.name

        s.write("midi", fp=midi_path)

        with open(midi_path, "rb") as f:
            midi_bytes = f.read()

        os.remove(midi_path)

    with st.spinner("Synthesizing Audio..."):
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        
        audio_data = midi_data.synthesize(fs=44100)
        
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 44100, audio_data.astype(np.float32))
        
        st.write("### Listen to Bach (Synthesized)")
        st.audio(wav_buffer.getvalue(), format="audio/wav")
        
        st.download_button(
            label="Download Audio (.wav)",
            data=wav_buffer.getvalue(),
            file_name="bach_chorale.wav",
            mime="audio/wav"
        )

    st.write("### Generated Notes (S, A, T, B)")
    st.dataframe(generated_notes)