import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

# Load data
df = pd.read_csv('data/tennis_matches.csv')

# Replace empty strings with NaN
df.replace('', pd.NA, inplace=True)

# Drop rows with missing values in critical columns
df.dropna(subset=['winner_id', 'winner_age', 'loser_age', 'w_ace', 'l_ace'], inplace=True)

# Fill missing values for numerical columns
df['winner_seed'] = df['winner_seed'].fillna(0)
df['loser_seed'] = df['loser_seed'].fillna(0)
df['winner_ht'] = df['winner_ht'].fillna(df['winner_ht'].mean())
df['loser_ht'] = df['loser_ht'].fillna(df['loser_ht'].mean())

# Ensure numerical columns are of correct type
numerical_cols = ['winner_age', 'loser_age', 'w_ace', 'l_ace', 'winner_seed', 'loser_seed', 'winner_ht', 'loser_ht']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaNs that resulted from coercion
df[numerical_cols] = df[numerical_cols].fillna(0)

# Convert categorical columns to numerical
label_cols = ['surface', 'winner_hand', 'loser_hand', 'winner_ioc', 'loser_ioc', 'tourney_level', 'round']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Calculate 'OddsRatio'
df['OddsRatio'] = df['w_ace'] / (df['l_ace'] + 1)

# Save label encoders
joblib.dump(label_encoders, 'path_to_save_model/label_encoders.pkl')

# Debugging: Print columns to ensure 'OddsRatio' is created
print("DataFrame Columns:", df.columns)

# Feature selection
features = df[['surface', 'winner_hand', 'loser_hand', 'winner_age', 'loser_age', 'winner_seed', 'loser_seed', 'w_ace', 'l_ace', 'OddsRatio']]
target = df['winner_id'].apply(lambda x: 1 if x == df['winner_id'].mode()[0] else 0)  # Ensure this is 0 or 1 for binary classification

# Check if the features DataFrame is empty
if features.empty:
    raise ValueError("Features DataFrame is empty. Check the data processing steps.")

# Normalize features
scaler = StandardScaler()
features[['winner_age', 'loser_age', 'w_ace', 'l_ace', 'OddsRatio']] = scaler.fit_transform(features[['winner_age', 'loser_age', 'w_ace', 'l_ace', 'OddsRatio']])

# Save scaler
joblib.dump(scaler, 'path_to_save_model/scaler.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define and compile the model
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define checkpoint callback to save the best model
checkpoint = ModelCheckpoint('path_to_save_model/best_model.keras', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the final model
model.save('path_to_save_model/final_model.keras')
