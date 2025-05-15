import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ======================
# 1. LOAD DATA FUNCTION
# ======================

def load_features_and_labels(features_path, labels_csv_path, data_dir):
    """
    Loads features and labels from the dataset and ensures alignment.

    Parameters:
    - features_path: Path to the extracted features .npy file.
    - labels_csv_path: Path to the CSV file containing IDs and labels.
    - data_dir: Directory containing image files.

    Returns:
    - features: NumPy array of features (flattened if necessary).
    - labels: NumPy array of cleaned numeric labels.
    - image_ids: List of image IDs corresponding to the loaded features.
    """
    # Load extracted features
    try:
        features = np.load(features_path).astype(np.float32)
        print(f"Features loaded successfully with shape: {features.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Features file not found at {features_path}")

    # Load labels from CSV
    try:
        # Reading CSV without header, and naming columns manually
        labels_df = pd.read_csv(labels_csv_path, header=None, names=['ID', 'BoneAge'])
        print(f"Labels CSV loaded successfully with shape: {labels_df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels CSV file not found at {labels_csv_path}")

    # Get a list of image filenames and their IDs
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Extract image IDs from filenames (without extension)
    image_ids = [os.path.splitext(f)[0] for f in image_files]

    print(f"Found {len(image_files)} images in the directory")

    # Create a mapping from image ID to feature index
    feature_id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}

    # Filter labels and features to ensure alignment
    aligned_features = []
    aligned_labels = []
    aligned_image_ids = []

    for index, row in labels_df.iterrows():
        img_id = row['ID']
        if img_id in feature_id_to_index:
            feature_index = feature_id_to_index[img_id]
            if feature_index < len(features):
                aligned_features.append(features[feature_index])
                aligned_labels.append(float(row['BoneAge']))
                aligned_image_ids.append(img_id)

    features = np.array(aligned_features, dtype=np.float32)
    labels = np.array(aligned_labels, dtype=np.float32)

    # Final check
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("No aligned samples were found. Check your ID formats.")

    print(f"Successfully aligned {len(features)} samples")
    return features, labels, aligned_image_ids

# ======================
# 2. MODEL BUILDING FUNCTION
# ======================
def build_linear_regression_model(input_shape):
    """
    Build a robust linear regression model using TensorFlow/Keras.

    Parameters:
    - input_shape: Tuple representing the shape of the input features.

    Returns:
    - model: A compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Input layer matching feature dimensions
        tf.keras.layers.Flatten(),                  # Flatten the input if it has multiple dimensions
        tf.keras.layers.Dense(128, activation='relu'),  # Add a hidden layer
        tf.keras.layers.Dropout(0.3),                   # Add dropout for regularization
        tf.keras.layers.Dense(64, activation='relu'),   # Add another hidden layer
        tf.keras.layers.Dropout(0.3),                   # Add dropout for regularization
        tf.keras.layers.Dense(1)                        # Single output for regression
    ])

    # Compile the model with Mean Squared Error (MSE) loss and Adam optimizer
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

# ======================
# 4. PREDICTION REPORT GENERATOR
# ======================
def generate_prediction_report(image_ids, predictions, output_csv='prediction_report.csv'):
    """
    Generates a CSV report containing image IDs and model predictions for the test set.

    Args:
        image_ids (list): List of image IDs for the test set.
        predictions (np.array): Array of model predictions for the test set.
        output_csv (str): Path to save the CSV report.
    """
    report_df = pd.DataFrame({'ID': image_ids, 'prediction': predictions.flatten()})
    report_df.to_csv(output_csv, index=False)
    print(f"Prediction report for test set saved to: {output_csv}")

# ======================
# 3. MAIN WORKFLOW
# ======================

def main():
    # Paths to data
    FEATURES_PATH = "/home/madhavr/Yantragya/Yantragya_pramukh_vatavaran/Feature_Extracted_output/Features_Extractions_output.npy"
    LABELS_CSV_PATH = "/home/madhavr/Yantragya/Yantragya_pramukh_vatavaran/Bone_Age_data/BoneAge_Dataset/BoneAge_train.csv"
    TRAINING_DATA_DIR = "/home/madhavr/Yantragya/Yantragya_pramukh_vatavaran/Bone_Age_data/BoneAge_Dataset/Training_dataset_BoneAge"

    try:
        # Step 1: Load the data
        print("Step 1: Loading data...")
        features, labels, all_image_ids = load_features_and_labels(FEATURES_PATH, LABELS_CSV_PATH, TRAINING_DATA_DIR)

        # Step 2: Verify data is not empty
        if len(features) == 0 or len(labels) == 0:
            print("Error: Features or labels array is empty!")
            return

        print(f"Loaded {len(features)} samples successfully")

        # Step 3: Split into training and testing sets
        print("Step 3: Splitting data...")
        X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
            features.reshape(features.shape[0], -1),  # Flatten features for Dense layer compatibility
            labels,
            all_image_ids,
            test_size=0.2,
            random_state=42
        )

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, train_ids shape: {len(train_ids)}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, test_ids shape: {len(test_ids)}")

        # Step 4: Build the model
        print("Step 4: Building model...")
        input_shape = X_train.shape[1:]  # Automatically infer input shape from training data

        model = build_linear_regression_model(input_shape)

        # Print model summary
        model.summary()

        # Step 5: Train the model
        print("Step 5: Training model...")
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,  # More epochs with early stopping
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )

        # Step 6: Evaluate the model
        print("Step 6: Evaluating model...")
        loss, mae = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Mean Absolute Error: {mae:.4f}")

        # Step 7: Make predictions on the test set
        print("Step 7: Making predictions on the test set...")
        y_pred = model.predict(X_test).flatten()

        # Step 8: Generate the prediction report CSV for the test set
        print("Step 8: Generating the prediction report for the test set...")
        generate_prediction_report(test_ids, y_pred, output_csv='test_predictions.csv')

        # Step 9: Analyze errors (MAE on the test set)
        mae_test = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error on the Test Set: {mae_test:.2f} months")

        # Save the model
        model.save('bone_age_model.h5')
        print("Model saved as 'bone_age_model.h5'")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging

if __name__ == "__main__":
    main()