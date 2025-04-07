from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import librosa
import io
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

CLASSES = ["si","nothing", "no", "ya", "que", "yo"]

logger.info("Iniciando la carga del modelo...")
try:
    model = tf.keras.models.load_model("/app/monosyllables_model_v0.keras")
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise

def alaw_to_linear(audio_alaw):
    logger.debug(f"Convirtiendo ALAW a lineal, longitud: {len(audio_alaw)}")
    alaw_decoded = librosa.mu_expand(audio_alaw, mu=255, quantize=False)
    logger.debug(f"Conversión completada, longitud: {len(alaw_decoded)}")
    return alaw_decoded

def extract_mfcc(audio_data, sr=8000):
    """Extraer características MFCC del audio"""
    logger.debug(f"Extrayendo MFCC, longitud: {len(audio_data)}")
    n_mfcc = 24  # Alineado con el modelo
    n_fft = 512
    frame_size = 512
    frame_step = 40  # Ajustado para coincidir con 260 frames
    num_blocks = 260  # Alineado con el modelo

    mfcc = librosa.feature.mfcc(
        y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=frame_step, win_length=frame_size
    )
    mfcc = mfcc.T  # (frames, n_mfcc)

    logger.debug(f"MFCC calculado, forma inicial: {mfcc.shape}")
    if mfcc.shape[0] < num_blocks:
        pad_width = num_blocks - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
        logger.debug(f"Rellenado a {num_blocks} bloques")
    elif mfcc.shape[0] > num_blocks:
        mfcc = mfcc[:num_blocks, :]
        logger.debug(f"Recortado a {num_blocks} bloques")

    mfcc = mfcc[np.newaxis, ...]  # (1, num_blocks, n_mfcc)
    logger.debug(f"Forma final de MFCC: {mfcc.shape}")
    return mfcc

def predict_class(mfcc_features):
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

        audio_array = np.frombuffer(audio_bytes, dtype=np.uint8)
        logger.debug(f"Array numpy creado, forma: {audio_array.shape}")

        audio_linear = alaw_to_linear(audio_array)
        mfcc_features = extract_mfcc(audio_linear)
        prediction = predict_class(mfcc_features)

        logger.info(f"Predicción completada: {prediction}")
        return json.dumps({"prediction": prediction},ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando el servidor Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8125)