from tensorflow import keras

# Load the model
model = keras.models.load_model("./monosyllables_model_v0.keras")

# Run inference
print(model.summary())

#model.predict( data )
