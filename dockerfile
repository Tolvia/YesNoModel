FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    tensorflow==2.14.0 \
    fastapi==0.115.3 \
    uvicorn==0.32.0 \
    librosa==0.10.1 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    pyaudio==0.2.14\
    python-multipart

COPY app.py .

COPY monosyllables_model_v0.keras .

EXPOSE 8125

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8125", "--log-level", "debug"]