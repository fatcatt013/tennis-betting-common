import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

# Define paths to the model files
model_path_keras = 'path_to_save_model/final_model.keras'
# model_path_h5 = 'final_model.h5'  # Uncomment if you prefer to use .h5 model

# Check if model files exist
if not os.path.isfile(model_path_keras):
    raise FileNotFoundError(f"Model file not found at {model_path_keras}")

# Load the Keras model
model = tf.keras.models.load_model(model_path_keras)
# Alternatively, if using .h5 file:
# model = tf.keras.models.load_model(model_path_h5)

# Load new data
new_data = pd.read_csv('data/new_tennis_matches.csv')

# Define columns used for training (ensure they match the model input features)
training_columns = ['tourney_id', 'surface', 'winner_hand', 'loser_hand', 'winner_ioc', 'loser_ioc',
                    'tourney_level', 'round', 'winner_age', 'loser_age', 'winner_seed', 'loser_seed',
                    'winner_ht', 'loser_ht', 'w_ace', 'l_ace']

# Ensure the new data has the same columns
new_data = new_data[training_columns]

# Print the new_data shape and columns to verify
print("New data shape:", new_data.shape)
print("New data columns:", new_data.columns.tolist())

# Load or initialize encoders used during training
encoders = {
    'tourney_id': LabelEncoder(),
    'surface': LabelEncoder(),
    'winner_hand': LabelEncoder(),
    'loser_hand': LabelEncoder(),
    'winner_ioc': LabelEncoder(),
    'loser_ioc': LabelEncoder(),
    'tourney_level': LabelEncoder(),
    'round': LabelEncoder()
}

# Dummy fit for encoders; replace this with loading from a file if available
for col, encoder in encoders.items():
    if col in new_data.columns:
        # Fit encoders with some sample data if you don't have pre-trained encoders
        sample_data = new_data[col].dropna().astype(str)
        encoder.fit(sample_data)
        new_data[col] = encoder.transform(new_data[col].astype(str))

# Ensure that the features in new_data match those expected by the model
# Convert all columns to numeric and handle NaN values
new_data = new_data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Print new_data again to verify changes
print("Transformed new data shape:", new_data.shape)

# Prepare data for prediction
X_new = new_data.values.astype('float32')  # Ensure data type is float32

# Make predictions
predictions = model.predict(X_new)

# Output predictions
output_df = new_data.copy()
output_df['predictions'] = predictions

# Save predictions to a CSV file
output_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")
