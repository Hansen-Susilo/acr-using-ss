import os
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile
import torchaudio

mp3_file = './random-test-songs/Radiohead - Creep.mp3'

# Load Demucs model
model = get_model(name="htdemucs")

# Baca file audio
wav = AudioFile(mp3_file).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
ref = wav
if ref.dim() == 1:
    ref = ref.unsqueeze(0)  # dari [time] -> [1, time]

print('Successfully Read the Audio File')

# Apply model
sources = apply_model(model, ref.unsqueeze(0), split=True, overlap=0.25)[0]
stem_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

print('successfully apply model')

# Simpan semua stem ke folder
os.makedirs("separated", exist_ok=True)
for i, name in enumerate(stem_names):
    path = f"separated/{name}.wav"
    torchaudio.save(path, sources[i].cpu(), model.samplerate)

print("✅ Stem separation selesai.")

# Ambil sumber 'bass' dan 'other'
bass = sources[stem_names.index('bass')]
other = sources[stem_names.index('other')]

# Samakan panjang (safety)
min_len = min(bass.shape[-1], other.shape[-1])
bass = bass[..., :min_len]
other = other[..., :min_len]

# Gabungkan
combined = bass + other

# Simpan sebagai WAV sementara
combined_path = "combined_bass_other.wav"
torchaudio.save(combined_path, combined.cpu(), model.samplerate)

print("✅ Gabungan bass + other disimpan (WAV sementara).")

# from pydub import AudioSegment

# # Konversi WAV ke MP3
# audio = AudioSegment.from_wav(combined_path)
# mp3_output = "combined_bass_other.mp3"
# audio.export(mp3_output, format="mp3", bitrate="192k")
# files.download(mp3_output)

# print("✅ File akhir disimpan sebagai MP3:", mp3_output)