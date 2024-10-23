import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.stats import skew, kurtosis
import os
from tqdm import tqdm
from tensorflow.keras import layers, Model, utils

# Custom layers and model definitions
@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

@tf.keras.utils.register_keras_serializable()
class TransformerClassifier(Model):
    def __init__(self, num_classes, embed_dim, num_heads, ff_dim, num_layers, **kwargs):
        super(TransformerClassifier, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dense_input = layers.Dense(embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense_input(inputs)
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        x = self.global_average_pooling(x)
        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers
        })
        return config

# Data loading and processing functions
def load_single_svc_file(file_path):
    """Load and process a single .svc file"""
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        df = pd.DataFrame([line.split() for line in lines])
        total_rows = int(df.iloc[0, 0])
        data = df.iloc[1:, :].values

        reshaped_data = []
        for row in data:
            if len(row) == 7:
                reshaped_data.append(row)

        if reshaped_data:
            reshaped_data = np.array(reshaped_data, dtype=float)
            return reshaped_data

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def extract_time_domain_features(data):
    """Extract time domain features from the data"""
    time_features = []
    
    for sample in data.T:
        feature_stats = [
            np.mean(sample),
            np.std(sample),
            np.min(sample),
            np.max(sample),
            np.percentile(sample, 25),
            np.percentile(sample, 50),
            np.percentile(sample, 75),
        ]
        time_features.extend(feature_stats)

    return np.array(time_features)

def extract_frequency_domain_features(data):
    """Extract frequency domain features from the data"""
    freq_features = []
    
    for sample in data.T:
        freq_feature = np.fft.fft(sample)
        freq_magnitude = np.abs(freq_feature)
        dominant_freq = np.argmax(freq_magnitude)
        freq_energy = np.sum(freq_magnitude)
        freq_features.extend([dominant_freq, freq_energy])

    return np.array(freq_features)

def extract_statistical_features(data):
    """Extract statistical features from the data"""
    statistical_features = []
    
    skew_value = skew(data.T)
    kurtosis_value = kurtosis(data.T)
    
    for feature in data.T:
        mean = np.mean(feature)
        median = np.median(feature)
        variance = np.var(feature)
        feature_stats = [mean, median, variance, skew_value, kurtosis_value]
        statistical_features.extend(feature_stats)
    
    return np.array(statistical_features)

def classify_svc_file(model_path, file_path):
    """Classify a single .svc file using the trained model"""
    # Class labels
    class_labels = ['Normal', 'Depression', 'Anxiety', 'Stress']
    
    # Load the trained model with custom objects
    model = tf.keras.models.load_model(model_path, 
                                     custom_objects={
                                         'TransformerClassifier': TransformerClassifier,
                                         'TransformerBlock': TransformerBlock,
                                         'MultiHeadSelfAttention': MultiHeadSelfAttention
                                     })
    
    # Load and process the input file
    data = load_single_svc_file(file_path)
    
    if data is None:
        return "Error: Could not process the input file"
    
    # Extract features
    time_features = extract_time_domain_features(data)
    freq_features = extract_frequency_domain_features(data)
    stat_features = extract_statistical_features(data)
    
    # Combine all features
    combined_features = np.concatenate([time_features, freq_features, stat_features])
    
    # Reshape for model input (add batch dimension)
    features = combined_features.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction[0])
    
    # Get confidence score
    confidence = prediction[0][predicted_class] * 100
    
    return {
        'predicted_class': class_labels[predicted_class],
        'confidence': f"{confidence:.2f}%",
        'probabilities': {
            class_labels[i]: f"{prob * 100:.2f}%" 
            for i, prob in enumerate(prediction[0])
        }
    }

def process_directory(model_path, directory_path):
    """Process all .svc files in a directory"""
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
        return
    
    svc_files = [f for f in os.listdir(directory_path) if f.endswith('.svc')]
    
    if not svc_files:
        print("No .svc files found in the directory")
        return
    
    for file_name in svc_files:
        file_path = os.path.join(directory_path, file_name)
        print(f"\nProcessing: {file_name}")
        
        result = classify_svc_file(model_path, file_path)
        
        if isinstance(result, dict):
            print(f"Prediction: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']}")
            print("\nProbabilities for each class:")
            for class_name, prob in result['probabilities'].items():
                print(f"{class_name}: {prob}")
        else:
            print(result)

# Example usage:
def get_absolute_path(relative_path):
    """Get absolute path to ensure correct file paths regardless of the current working directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
    return os.path.join(script_dir, relative_path)

if __name__ == "__main__":
    # Use the helper function to set the correct paths
    MODEL_PATH = get_absolute_path("newmodels/newworkinghigh.model.keras")
    INPUT_DIRECTORY = get_absolute_path("input_files/")

    print("Starting classification process...")
    process_directory(MODEL_PATH, INPUT_DIRECTORY)