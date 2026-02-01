# Automatic Modulation Recognition for Intelligent Radio Systems

**Project Analysis & Architecture Suggestions**
*Comprehensive Study and Implementation Guide*

**Intelligent Systems Engineering Department**
**The National School of Artificial Intelligence**
**4th Year, Semester 1 — 2025/2026**

**Wireless Communication Networks and Systems**

**Prepared by:**
Lyes HADJAR
Moulay Mohamed BOUABDELLI

---

## Executive Summary

This document provides a comprehensive analysis of the Automatic Modulation Recognition (AMR) mini-project, including detailed architectural suggestions, implementation strategies, and critical analysis frameworks. The project focuses on developing intelligent receivers capable of autonomously identifying modulation schemes in wireless communication systems, a crucial capability for 5G/6G networks and cognitive radio applications.

### Key Project Objectives
*   Understand digital modulation schemes and IQ representation
*   Analyze channel effects (Rayleigh fading, AWGN) on received signals
*   Generate synthetic datasets with multiple modulation types and SNR levels
*   Implement equalization techniques (ZF, MMSE)
*   Design deep learning models for automatic modulation classification
*   Evaluate performance across SNR ranges and equalization strategies

---

## Installation & Usage

### Requirements
Ensure you have the following dependencies installed (see `requirements.txt`):
```bash
pip install -r requirements.txt
```
Key libraries: `numpy`, `matplotlib`, `tensorflow`, `scikit-learn`, `pandas`.

### Usage
The project is organized into Jupyter notebooks located in the `notebooks/` directory:

1.  **`notebooks/01_baseline_amr.ipynb`**:
    *   **Run this first.**
    *   Generates the synthetic datasets (Raw, Zero-Forcing, MMSE) for various SNRs.
    *   Implements and evaluates the Baseline CNN model.
    *   Saves the datasets to `dataset/`, `dataset_ZF/`, and `dataset_MMSE/`.

2.  **`notebooks/02_advanced_architectures.ipynb`**:
    *   **Run this second.**
    *   Implements advanced architectures (ResNet, LSTM, CNN-Transformer, etc.).
    *   Comparatively evaluates these models using the datasets generated in step 1.

---

## Project Overview and Analysis

### Problem Statement

Automatic Modulation Recognition addresses the fundamental challenge of identifying modulation schemes without prior transmitter knowledge. This capability is essential for:

*   **Adaptive Communication Systems**: Dynamic adjustment to channel conditions
*   **Cognitive Radio**: Spectrum sensing and dynamic spectrum access
*   **Signal Intelligence**: Monitoring and identifying unknown transmissions
*   **Link Adaptation**: Optimizing spectral efficiency in 5G/6G networks

### Dataset Characteristics

#### Modulation Schemes
The project includes five modulation types covering different dimensions:

| Modulation | Type | Complexity |
| :--- | :--- | :--- |
| BPSK | Phase | Binary |
| 4-PAM | Amplitude | 4-level |
| 8-PSK | Phase | 8-symbol |
| 16-QAM | Quadrature | 16-symbol |
| 2-FSK | Frequency | Binary |

#### Channel Model
*   **Fading**: Flat Rayleigh fading (frequency-nonselective)
*   **Noise**: Additive White Gaussian Noise (AWGN)
*   **SNR Range**: -5 dB to 25 dB (7 levels)
*   **Signal Length**: 1024 IQ samples per signal
*   **Dataset Size**: 1000 signals per modulation per SNR = 35,000 total signals

#### Equalization Strategies
Three dataset versions enable comparative analysis:
1.  **Raw**: No equalization (baseline performance)
2.  **Zero-Forcing (ZF)**: $\hat{s} = \frac{y}{h}$ — eliminates ISI but amplifies noise
3.  **MMSE**: $\hat{s} = \frac{h^* y}{|h|^2 + \sigma_n^2}$ — balances ISI and noise

### Baseline CNN Architecture Analysis

The provided baseline model uses a conventional 1D convolutional architecture:

*   **Input**: (1024, 2) — 1024 time steps, 2 features (I/Q)
*   **Conv1D Layers**: 3 layers with increasing filters (64→128→256)
*   **Pooling**: MaxPooling1D after each convolution
*   **Dense Layers**: 256-unit fully connected layer with dropout
*   **Output**: 5-class softmax for modulation classification

**Strengths**:
*   Simple and interpretable architecture
*   Captures temporal patterns through 1D convolutions
*   Efficient training with moderate parameter count
*   Good baseline for comparison

**Limitations**:
*   Limited receptive field may miss long-range dependencies
*   No explicit attention to phase-frequency relationships
*   Lacks specialized mechanisms for temporal sequences
*   May struggle with subtle differences at low SNR

---

## Proposed Alternative Architectures

### Architecture 1: ResNet-Inspired Deep CNN

#### Rationale
Residual connections enable training deeper networks and help capture multi-scale features critical for distinguishing subtle modulation differences.

#### Architecture Design
```python
from tensorflow.keras.layers import Add, BatchNormalization

def residual_block(x, filters, kernel_size=3):
    """Residual block with skip connection"""
    shortcut = x

    # First conv
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second conv
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    # Skip connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet_amr():
    inputs = Input(shape=(1024, 2))

    # Initial conv
    x = Conv1D(64, 7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPooling1D(2)(x)

    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = MaxPooling1D(2)(x)

    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = GlobalAveragePooling1D()(x)

    # Classification head
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
```

### Architecture 2: LSTM-Based Temporal Model

#### Rationale
LSTMs excel at capturing long-term temporal dependencies, crucial for identifying patterns in time-series modulation data, particularly phase evolution and frequency transitions.

#### Architecture Design
```python
from tensorflow.keras.layers import LSTM, Bidirectional

def build_lstm_amr():
    model = Sequential([
        Input(shape=(1024, 2)),

        # Bidirectional LSTMs capture forward/backward patterns
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64)),
        Dropout(0.3),

        # Dense classification
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### Architecture 3: Hybrid CNN-LSTM

#### Rationale
Combines CNN's local feature extraction with LSTM's temporal modeling, providing the best of both worlds for modulation recognition.

#### Architecture Design
```python
def build_cnn_lstm():
    model = Sequential([
        Input(shape=(1024, 2)),

        # CNN for local feature extraction
        Conv1D(64, 7, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        # LSTM for temporal dependencies
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),

        # Classification
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### Architecture 4: Attention-Based Transformer

#### Rationale
Transformers with self-attention mechanisms can identify relevant signal features regardless of temporal distance, potentially superior for complex modulations.

#### Architecture Design
```python
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)

    # Skip connection and normalization
    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-forward network
    ff_output = Dense(ff_dim, activation='relu')(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)

    # Skip connection and normalization
    return LayerNormalization(epsilon=1e-6)(x + ff_output)

def build_transformer_amr():
    inputs = Input(shape=(1024, 2))

    # Positional embedding
    x = Dense(128)(inputs)

    # Transformer blocks
    for _ in range(4):
        x = transformer_encoder(x, head_size=32, num_heads=4,
                                ff_dim=256, dropout=0.1)

    # Global pooling and classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### Architecture 5: CNN-Transformer Hybrid with Latent Attention

#### Rationale
This novel architecture combines the local feature extraction capabilities of CNNs with the global context modeling of Transformers. The CNN acts as a feature extractor to reduce dimensionality and capture local patterns, while the Transformer processes these features with attention mechanisms to identify relationships across the entire signal. A latent representation layer enables the model to learn a compressed, discriminative embedding space for modulation classification.

#### Architecture Design
```python
from tensorflow.keras.layers import (MultiHeadAttention, LayerNormalization,
                                     GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.models import Model

def positional_encoding(length, depth):
    """Generate positional encodings for transformer"""
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate([
        np.sin(angle_rads[:, 0::2]),
        np.cos(angle_rads[:, 1::2])
    ], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

class AddPositionalEncoding(tf.keras.layers.Layer):
    """Add positional encoding to inputs"""
    def __init__(self, **kwargs):
        super(AddPositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        depth = tf.shape(inputs)[2]
        pos_enc = positional_encoding(length, depth)
        return inputs + pos_enc[tf.newaxis, :, :]

def transformer_block(x, num_heads, key_dim, ff_dim, dropout=0.1):
    """Single transformer encoder block"""
    # Multi-head attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(x, x)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(x.shape[-1])(ffn_output)

    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def build_cnn_transformer_latent():
    """
    CNN-Transformer architecture with latent representation
    """
    inputs = Input(shape=(1024, 2), name='input_signal')

    # CNN Feature Extractor
    x = Conv1D(64, 7, padding='same', activation='relu', name='conv1')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)  # 1024 -> 512

    x = Conv1D(128, 5, padding='same', activation='relu', name='conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)  # 512 -> 256

    x = Conv1D(256, 3, padding='same', activation='relu', name='conv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)  # 256 -> 128

    # Project to transformer dimension
    cnn_features = Conv1D(256, 1, activation='relu',
                          name='feature_projection')(x)

    # Add Positional Encoding
    x = AddPositionalEncoding()(cnn_features)

    # Transformer Encoder Blocks
    for i in range(3):
        x = transformer_block(
            x,
            num_heads=8,
            key_dim=32,
            ff_dim=512,
            dropout=0.1
        )

    transformer_features = x

    # Latent Representation Layer
    latent_dim = 128
    global_features = GlobalAveragePooling1D()(transformer_features)
    latent_repr = Dense(latent_dim, activation='relu',
                       name='latent_representation')(global_features)
    latent_repr = BatchNormalization()(latent_repr)
    latent_repr = Dropout(0.3)(latent_repr)

    # Attention-based Weighted Pooling
    attention_weights = Dense(1, activation='tanh')(transformer_features)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)

    attended_features = tf.keras.layers.Multiply()(
        [transformer_features, attention_weights]
    )
    attended_pooled = tf.reduce_sum(attended_features, axis=1)

    # Feature Fusion
    fused_features = Concatenate()([latent_repr, attended_pooled])

    # Classification Head (FFN)
    x = Dense(256, activation='relu', name='ffn_layer1')(fused_features)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', name='ffn_layer2')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(5, activation='softmax', name='classification')(x)

    model = Model(inputs=inputs, outputs=outputs,
                 name='CNN_Transformer_Latent')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

### Architecture 6: Dual-Path Network (I/Q Separation)

#### Rationale
Processes I and Q components separately before fusion, allowing the model to learn channel-specific features.

#### Architecture Design
```python
from tensorflow.keras.layers import Concatenate

def build_dual_path_network():
    # Input
    inputs = Input(shape=(1024, 2))

    # Separate I and Q channels
    i_channel = Lambda(lambda x: x[:, :, 0:1])(inputs)
    q_channel = Lambda(lambda x: x[:, :, 1:2])(inputs)

    # Parallel processing paths
    def process_channel(x):
        x = Conv1D(64, 7, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(256, 3, activation='relu', padding='same')(x)
        x = GlobalMaxPooling1D()(x)
        return x

    i_features = process_channel(i_channel)
    q_features = process_channel(q_channel)

    # Fusion
    merged = Concatenate()([i_features, q_features])
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### Architecture 7: Ensemble Model

#### Rationale
Combines predictions from multiple architectures to leverage their complementary strengths.

#### Implementation Strategy
```python
def build_ensemble():
    """Train multiple models and combine predictions"""
    models = {
        'cnn': build_cnn(),
        'resnet': build_resnet_amr(),
        'lstm': build_lstm_amr(),
        'cnn_lstm': build_cnn_lstm()
    }

    return models

def ensemble_predict(models, X_test):
    """Weighted voting or averaging"""
    predictions = []
    weights = [0.25, 0.30, 0.20, 0.25]  # Adjust based on validation

    for model in models.values():
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)

    # Weighted average
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    return np.argmax(ensemble_pred, axis=1)
```

---

## Advanced Preprocessing Techniques

### Constellation Normalization
```python
def normalize_constellation(X):
    """Normalize to unit average power"""
    power = np.mean(np.abs(X)**2, axis=1, keepdims=True)
    return X / np.sqrt(power + 1e-8)

def phase_rotation_augmentation(X, y):
    """Data augmentation through phase rotation"""
    angles = np.random.uniform(0, 2*np.pi, size=(len(X), 1, 1))
    rotation = np.exp(1j * angles)

    # Apply to complex signal
    complex_signal = X[:, :, 0] + 1j * X[:, :, 1]
    rotated = complex_signal * rotation

    # Convert back to I/Q
    X_aug = np.stack([rotated.real, rotated.imag], axis=-1)
    return X_aug, y
```

### Feature Engineering
*   **Higher-Order Statistics**: Extract 2nd and 4th order cumulants.
*   **Cyclic Spectrum Features**: Compute cyclic autocorrelation features.

---

## Experimental Design and Evaluation

### Training Strategy
*   **Cross-SNR Training**: Train on mixed SNR datasets (e.g., 0-20 dB) to improve robustness.
*   **Transfer Learning**: Pre-train on high SNR, fine-tune on low SNR.

### Performance Metrics
*   Classification Report (Precision, Recall, F1-score)
*   Cohen's Kappa
*   ROC-AUC for multi-class

---

## Critical Analysis Framework

### Expected Results by Architecture
| Architecture | High SNR | Low SNR | Training Time | Complexity |
| :--- | :---: | :---: | :---: | :---: |
| Baseline CNN | 95-98% | 60-70% | Fast | Low |
| ResNet CNN | 96-99% | 65-75% | Medium | Medium |
| LSTM | 94-97% | 70-80% | Slow | Medium |
| CNN-LSTM | 97-99.5% | 75-85% | Slow | High |
| Transformer | 97-99% | 72-82% | Medium | High |
| **CNN-Transformer** | **98-99.5%** | **80-90%** | Medium-Slow | High |
| Dual-Path | 96-98% | 68-78% | Medium | Medium |
| Ensemble | 98-99.8% | 78-88% | Very Slow | Very High |

### Equalization Impact Analysis
*   **Raw Data**: Baseline performance, channel distortion present.
*   **ZF Equalization**: Better at high SNR, noise amplification at low SNR.
*   **MMSE Equalization**: Best overall, balances ISI and noise.

---

## Implementation Roadmap

### Phase 1: Baseline Reproduction (Week 1)
*   Generate datasets using provided code.
*   Train baseline CNN on all three dataset versions.
*   Reproduce baseline results and document performance.

### Phase 2: Architecture Exploration (Week 2-3)
*   Implement ResNet architecture.
*   Implement LSTM or CNN-LSTM hybrid.
*   Compare performance with baseline.

### Phase 3: Advanced Techniques (Week 3-4)
*   Implement Transformer or another advanced architecture.
*   Apply data augmentation techniques.
*   Ensemble multiple models.

### Phase 4: Analysis and Reporting (Week 4)
*   Generate comprehensive visualizations.
*   Perform confusion matrix analysis.
*   Compare equalization strategies.
*   Write technical report with findings.

---

## Expected Challenges and Solutions

### Overfitting at High SNR
*   **Solutions**: Increase dropout rates, apply data augmentation (phase rotation), use early stopping.

### Poor Low SNR Performance
*   **Solutions**: Weighted loss for low SNR samples, denoising autoencoders, ensemble models.

### Similar Modulation Confusion
*   **Solutions**: Extract discriminative features (higher-order statistics), use attention mechanisms, apply focal loss.

---

## Conclusion and Recommendations

Based on the analysis, the **CNN-Transformer with Latent Attention** (Architecture 5) is recommended as the primary architecture. It offers the best balance of:
*   **Performance**: Expected 98-99.5% high SNR, 80-90% low SNR.
*   **Interpretability**: Attention visualization.
*   **Efficiency**: CNN reduces sequence length before attention operations.

Key takeaways:
*   Deep learning enables robust blind modulation classification.
*   Architecture choice significantly impacts low-SNR performance.
*   MMSE equalization provides consistent benefits across SNR ranges.
*   Ensemble methods achieve highest accuracy at cost of complexity.
