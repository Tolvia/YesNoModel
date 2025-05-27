from elevenlabs.client import ElevenLabs
import os
import librosa
import numpy as np
import soundfile as sf
import time

# Initialize ElevenLabs client (assumes API key is set in environment variables)
client = ElevenLabs(api_key="sk_79c09722936f496175457f8f164a916dfc7e14ac659895a9")

# List of voice IDs to use
voice_ids = [
    "KHCvMklQZZo0O30ERnVn",
    "gbTn1bmCvNgk0QEAVyfM",
    "Ir1QNHvhaJXbAGhT50w3",
    "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "AZnzlk1XvdvUeBnXmlld",  # Domi
    "EXAVITQu4vr4xnSDxMaL",  # Bella
    "ErXwobaYiN019PkySvjV",  # Antoni
    "TxGEqnHWrfWFTfGW9XjX"   # Josh
]

# Text to synthesize (Spanish "yes")
texts = ["sí", "no", "ok", "va", "vale", "ya", "lo", "me", "te", "di", "ve", "sol", "mar", "luz"]

# Directory to save audio files
output_dir = "synthesized_voices"
os.makedirs(output_dir, exist_ok=True)

def generate_and_save_audio(voice_id, text, output_path):
    try:
        # Generate audio using Multilingual v2 model
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # Save the audio to file
        with open(output_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        
        print(f"Successfully generated audio for voice {voice_id} at {output_path}")
        
    except Exception as e:
        print(f"Error generating audio for voice {voice_id}: {str(e)}")

# Función para aplicar data augmentation
def apply_augmentation(audio_path, output_dir, text, voice_id):
    try:
        # Cargar el archivo de audio
        y, sr = librosa.load(audio_path, sr=None)

        # Aplicar transformaciones
        augmentations = {
            "speed_up": librosa.effects.time_stretch(y, rate=1.5),
            "slow_down": librosa.effects.time_stretch(y, rate=0.75),
            "pitch_up": librosa.effects.pitch_shift(y, sr=sr, n_steps=4),
            "pitch_down": librosa.effects.pitch_shift(y, sr=sr, n_steps=-4),
            "add_noise": y + 0.005 * np.random.normal(0, 1, len(y))
        }

        # Guardar cada transformación
        for aug_name, augmented_audio in augmentations.items():
            augmented_output_path = os.path.join(output_dir, f"{text}_{voice_id}_{aug_name}_{int(time.time())}.wav")
            sf.write(augmented_output_path, augmented_audio, sr)
            print(f"Generated augmented audio: {augmented_output_path}")

    except Exception as e:
        print(f"Error applying augmentation to {audio_path}: {str(e)}")

# Generar audio para cada texto y voz, y aplicar data augmentation
for text in texts:
    for voice_id in voice_ids:
        # Crear un nombre de archivo único
        output_file = os.path.join(output_dir, f"{text}_{voice_id}_{int(time.time())}.mp3")
        # Generar y guardar el audio original
        generate_and_save_audio(voice_id, text, output_file)

        # Aplicar data augmentation al archivo generado
        apply_augmentation(output_file, output_dir, text, voice_id)

print("Audio generation and augmentation complete!")
