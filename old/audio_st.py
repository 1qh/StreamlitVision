from math import log2

import streamlit as st
import torchaudio
from pydub import AudioSegment
from streamlit import sidebar as sb
from torchaudio.functional import pitch_shift, speed

file = sb.file_uploader('Upload an audio file')

if file and 'audio' in file.type:
    sb.audio(file)

    if 'wav' not in file.type:
        s = f'up_{file.name}'
        with open(s, 'wb') as f:
            f.write(file.read())
        file = s

        if file.endswith('.mp3'):
            out = f"{'.'.join(file.split('.')[:-1])}.wav"
            AudioSegment.from_mp3(file).export(out, format='wav')
            file = out

    au, sr = torchaudio.load(file, normalize=True)

    factor = sb.slider('Speed', 0.75, 3.0, 1.0, 0.25)
    shift = sb.number_input('Pitch Shift', -24, 24, 0, 1)

    # Certain pitch shifts might not be possible and app will crash
    if sb.button('Export'):
        if factor == 1.0 and shift == 0:
            sb.warning('No changes made')
        else:
            out = au
            if shift != 0:
                pitch = shift - 12 * log2(factor)
                out = pitch_shift(out, n_steps=pitch, sample_rate=sr)
            if factor != 1.0:
                out = speed(out, orig_freq=sr, factor=factor)[0]

            new = 'out.wav'
            torchaudio.save(new, out, sr)
            st.audio(new)
