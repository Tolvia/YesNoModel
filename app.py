from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import librosa
import tensorflow as tf
import audioop
import logging
import sys


# Configurar logging para stdout sin buffering
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Indica cuántas muestras hay por segundo. Ejemplo: 8000 Hz = 8000 muestras en 1 segundo.
SAMPLE_RATE = 8000  # Ejemplo: 1 segundo de audio = 8000 muestras
# Número de coeficientes MFCC por cada ventana (frame).
# Es la cantidad de características acústicas que se extraen por bloque de audio.
N_MFCC = 13  # Ejemplo: cada frame tendrá un vector de 13 valores

# Avance entre frames, en número de muestras.
# Determina cada cuánto se calcula un nuevo MFCC.
HOP_LENGTH = 20  # Ejemplo: 20 muestras a 8000 Hz = 20 / 8000 = 0.0025 s = 2.5 ms por frame
# Tamaño de cada ventana para calcular FFT y MFCC, en muestras.
# Define la resolución espectral del análisis.
FRAME_SIZE = 512  # Ejemplo: 512 / 8000 = 0.064 s = 64 ms por ventana

# Número total de frames MFCC que se usarán como entrada para el modelo.
# Es el tamaño fijo que espera la red como input (se recorta o rellena).
TARGET_FRAMES = 400  
# Ejemplo: 400 frames × 20 muestras = 8000 muestras = 1 segundo exacto de audio

# Lista de clases que el modelo puede predecir.
CLASSES = ["No", "Sí", "nothing"]  
app = FastAPI()

try:
    logger.info("Cargando modelo...")
    model = tf.keras.models.load_model("/app/monosyllables_model_v3.keras")
    _ = model.predict(np.zeros((1, TARGET_FRAMES, N_MFCC)))  # Arrancamos motores
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise

# Nomalize audio and extract MFCC features
def extract_mfcc(audio: np.ndarray, target_frames: int = TARGET_FRAMES):
    audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0.0001 else audio
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=FRAME_SIZE,
        hop_length=HOP_LENGTH,
        win_length=FRAME_SIZE
    ).T

    if mfcc.shape[0] < target_frames:
        pad = target_frames - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    else:
        mfcc = mfcc[:target_frames]

    return mfcc[np.newaxis, ...]

# Predict class from MFCC features
def predict_class(mfcc: np.ndarray) -> str:
    prediction = model.predict(mfcc)
    predicted_class = CLASSES[np.argmax(prediction[0])]
    logger.debug(f"Predicción: {predicted_class} | Confianza: {prediction[0]}")
    return predicted_class

# Endpoint to handle audio file upload and prediction
@app.post("/predict")
async def predict_audio(file: UploadFile):
    logger.info(f"Recibiendo solicitud para archivo: {file.filename}")
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise ValueError("El buffer recibido está vacío")

        # A-law → PCM16 → float32
        pcm = audioop.alaw2lin(audio_bytes, 2)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        # Extract MFCC features
        mfcc = extract_mfcc(audio)
        pred = predict_class(mfcc)
        logger.info(f"Predicción completada: {pred}")
        return {"prediction": pred}

    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando el servidor Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8125)