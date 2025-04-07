from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import librosa
import tensorflow as tf
import logging
import sys
import json

# Configurar logging para stdout sin buffering
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()

CLASSES = ["si", "nothing", "no", "ya", "que", "yo"]
NMFCC = 13
HOP_LENGTH = 60
SAMPLE_RATE = 8000 
N_FFT = 512
FRAME_SIZE = 512
MAX_FRAMES = 300

logger.info("Iniciando la carga del modelo...")
try:
    model = tf.keras.models.load_model("/app/monosyllables_model_v0.keras")
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise

def alaw_to_linear(audio_alaw):
    """Convertir audio ALAW a formato lineal"""
    logger.debug(f"Convirtiendo ALAW a lineal, longitud: {len(audio_alaw)}")
    audio_normalized = (audio_alaw.astype(np.float32) - 128) / 128.0
    alaw_decoded = librosa.mu_expand(audio_normalized, mu=255, quantize=False)
    logger.debug(f"Conversión completada, longitud: {len(alaw_decoded)}")
    return alaw_decoded

def normalize(audio_data):
    """Normalizar el audio basado en el pico máximo"""
    max_peak = np.max(np.abs(audio_data))
    if max_peak > 0.0001:
        ratio = 1 / max_peak
        return audio_data * ratio
    return audio_data

def extract_mfcc(audio_data, sr=SAMPLE_RATE):
    """Extraer características MFCC del audio"""
    logger.debug(f"Extrayendo MFCC, longitud: {len(audio_data)}")
    audio_data = normalize(audio_data)
    logger.debug(f"Audio normalizado, longitud: {len(audio_data)}")

    mfcc = librosa.feature.mfcc(
        y=audio_data, sr=sr, n_mfcc=NMFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=FRAME_SIZE
    )
    mfcc = mfcc.T  
    logger.debug(f"MFCC calculado, forma inicial: {mfcc.shape}")
    if mfcc.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
        logger.debug(f"Rellenado a {MAX_FRAMES} bloques")
    elif mfcc.shape[0] > MAX_FRAMES:
        mfcc = mfcc[:MAX_FRAMES, :]
        logger.debug(f"Recortado a {MAX_FRAMES} bloques")

    mfcc = mfcc[np.newaxis, ...] 
    logger.debug(f"Forma final de MFCC: {mfcc.shape}")
    return mfcc

def predict_class(mfcc_features):
    """Realizar predicción con el modelo"""
    logger.debug("Realizando predicción...")
    predictions = model.predict(mfcc_features)
    logger.debug(f"Predicciones: {predictions}")
    class_idx = np.argmax(predictions[0])
    predicted_class = CLASSES[class_idx]
    logger.debug(f"Clase predicha: {predicted_class}")
    return predicted_class

@app.post("/predict")
async def predict_audio(file: UploadFile):
    logger.info(f"Recibiendo solicitud para archivo: {file.filename}")
    try:
        audio_bytes = await file.read()
        logger.debug(f"Bytes leídos: {len(audio_bytes)}")

        if len(audio_bytes) == 0:
            raise ValueError("El buffer recibido está vacío")

        audio_array = np.frombuffer(audio_bytes, dtype=np.uint8)
        logger.debug(f"Array numpy creado, forma: {audio_array.shape}")

        audio_linear = alaw_to_linear(audio_array)
        mfcc_features = extract_mfcc(audio_linear)
        prediction = predict_class(mfcc_features)

        logger.info(f"Predicción completada: {prediction}")
        return json.dumps({"prediction": prediction}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando el servidor Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8125)