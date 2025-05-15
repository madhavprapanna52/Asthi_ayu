import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ======================
# 1. LOAD DATA FUNCTION
# ======================

def load_features_and_labels(features_path, labels_csv_path, data_dir):
    """
    Loads features and labels from the dataset and ensures alignment.
    In this dataset, the 'ID' column is used as the label.

    Parameters:
    - features_path: Path to the extracted features .npy file.
    - labels_csv_path: Path to the CSV file containing IDs and labels.
    - data_dir: Directory containing image files.

    Returns:
    - features: NumPy array of features (flattened if necessary).
    - labels: NumPy array of cleaned numeric labels.
    """
    # Load extracted features
    try:
        features = np.load(features_path).astype(np.float32)
        print(f"Features loaded successfully with shape: {features.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Features file not found at {features_path}")

    # Load labels from CSV
    try:
        labels_df = pd.read_csv(labels_csv_path)
        print(f"Labels CSV loaded successfully with shape: {labels_df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels CSV file not found at {labels_csv_path}")

    # Ensure 'ID' column exists
    if 'ID' not in labels_df.columns:
        raise KeyError("'ID' column not found in labels CSV")
    
    # Convert 'ID' to numeric values, handling any non-numeric values
    labels_df['ID'] = pd.to_numeric(labels_df['ID'], errors='coerce')
    
    # Remove any rows with non-numeric ID values
    labels_df = labels_df.dropna(subset=['ID'])
    
    # Get a list of image filenames and their IDs
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_ids = [os.path.splitext(f)[0] for f in image_files]
    
    print(f"Found {len(image_files)} images in the directory")
    
    # Create a mapping from image ID to feature index
    feature_id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
    
    # Make sure image IDs and labels_df IDs are in the same format for comparison
    # Convert numerical IDs to strings
    labels_df['ID_str'] = labels_df['ID'].astype(str)
    
    # Filter labels to only include IDs that exist in the feature set
    relevant_labels_df = labels_df[labels_df['ID_str'].isin(image_ids)]
    
    print(f"Found {len(relevant_labels_df)} matching records in labels CSV")
    
    if len(relevant_labels_df) == 0:
        # Debug information if no matches were found
        print("Sample image IDs:", image_ids[:5])
        print("Sample CSV IDs:", labels_df['ID_str'].values[:5])
        raise ValueError("No matching IDs found between images and labels CSV")
    
    # If there's still a mismatch, we need to extract the subset of features that match the labels
    if len(features) != len(relevant_labels_df):
        print(f"Mismatch detected: Features count = {len(features)}, Labels count = {len(relevant_labels_df)}")
        print("Extracting matching subset of features...")
        
        # Two possible scenarios:
        # 1. The features array corresponds directly to the image files in the same order
        if len(features) == len(image_files):
            # Create a mask of indices to keep
            indices_to_keep = [i for i, img_id in enumerate(image_ids) if img_id in relevant_labels_df['ID_str'].values]
            features = features[indices_to_keep]
            
            # Reorder labels to match the features
            ordered_ids = [image_ids[i] for i in indices_to_keep]
            relevant_labels_df = relevant_labels_df.set_index('ID_str').loc[ordered_ids].reset_index()
        # 2. We need to completely rebuild the alignment based on matching IDs
        else:
            # Create new arrays with aligned data
            aligned_features = []
            aligned_labels = []
            
            # For each ID in the relevant labels
            for _, row in relevant_labels_df.iterrows():
                img_id = row['ID_str']
                if img_id in feature_id_to_index:
                    feature_idx = feature_id_to_index[img_id]
                    if feature_idx < len(features):
                        aligned_features.append(features[feature_idx])
                        aligned_labels.append(float(row['ID']))  # The ID column is used as the label
            
            features = np.array(aligned_features, dtype=np.float32)
            relevant_labels_df = pd.DataFrame({
                'ID': aligned_labels,
                'ID_str': relevant_labels_df['ID_str'][:len(aligned_labels)]
            })

    # Final check
    if len(features) != len(relevant_labels_df):
        raise ValueError(
            f"Unable to resolve mismatch: Features count = {len(features)}, Labels count = {len(relevant_labels_df)}"
        )
    
    print(f"Successfully aligned {len(features)} samples")
    
    # Extract the actual labels (ID column), ensuring they are float values
    labels = relevant_labels_df['ID'].astype(np.float32).values

    return features, labels

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
        tf.keras.layers.Dense(1)                    # Single output for regression
    ])

    # Compile the model with Mean Squared Error (MSE) loss and Adam optimizer
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


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
        features, labels = load_features_and_labels(FEATURES_PATH, LABELS_CSV_PATH, TRAINING_DATA_DIR)

        # Step 2: Verify data is not empty
        if len(features) == 0 or len(labels) == 0:
            print("Error: Features or labels array is empty!")
            return

        print(f"Loaded {len(features)} samples successfully")
        
        # Debug: Check data types
        print(f"Features dtype: {features.dtype}")
        print(f"Labels dtype: {labels.dtype}")
        
        # Ensure labels are floating point numbers
        labels = labels.astype(np.float32)

        # Step 3: Split into training and testing sets
        print("Step 3: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            features.reshape(features.shape[0], -1),  # Flatten features for Dense layer compatibility
            labels,
            test_size=0.2,
            random_state=42
        )

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Step 4: Build the model
        print("Step 4: Building model...")
        input_shape = X_train.shape[1:]  # Automatically infer input shape from training data

        model = build_linear_regression_model(input_shape)

        # Step 5: Train the model
        print("Step 5: Training model...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

        # Step 6: Evaluate the model
        print("Step 6: Evaluating model...")
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Mean Absolute Error: {mae:.4f}")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging

if __name__ == "__main__":
    main()