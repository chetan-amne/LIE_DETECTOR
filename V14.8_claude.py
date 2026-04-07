# ==============================================================================
# DECEPTION DETECTION — CLEAN TRAINING PIPELINE
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention,
    GlobalAveragePooling1D, MultiHeadAttention,
    LayerNormalization, Add, Conv1D, MaxPooling1D,
    Bidirectional, Flatten, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==============================================================================
# 0. CONFIG
# ==============================================================================

SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 1
RANDOM_STATE = 110
TEST_SIZE = 0.2
MAX_VIDEO_DURATION_S = 10  # filter threshold

FEATURE_COLS = ['emotion_conf', 'deception_score', 'au_velocity','brow_asymmetry', 'lid_asymmetry', 'mouth_asymmetry', 'total_asymmetry','AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU12', 'AU14R','AU15', 'AU17', 'AU20', 'AU25', 'AU26','emotion_Contempt', 'emotion_Fear', 'emotion_Surprise','risk_level_HIGH', 'risk_level_LOW', 'risk_level_MODERATE','micro_expression_Contempt', 'micro_expression_Fear',       'micro_expression_Surprise', 'micro_expression_a', 'masking_Sadness','masking_a', 'leakage_a']
  # 35 numeric features — 'video name' excluded


N_FEATURES = len(FEATURE_COLS)

# ==============================================================================
# 1. LOAD & FILTER DATA
# ==============================================================================

df = pd.read_csv('Preprocessed.csv')

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', inplace=True, axis=1)

# --- FIX: get actual video names (not nunique count) ---
df_grouped = df.groupby('video name')['time_s'].max().reset_index()

# Filter out videos longer than MAX_VIDEO_DURATION_S
long_video_names = df_grouped[df_grouped['time_s'] > MAX_VIDEO_DURATION_S]['video name']
df = df[~df['video name'].isin(long_video_names)].reset_index(drop=True)

print(f"Videos remaining after duration filter: {df['video name'].nunique()}")

# ==============================================================================
# 2. LABEL ENCODING
# ==============================================================================

# Label: 1 if video name contains 'lie', else 0
df['False'] = df['video name'].apply(lambda x: 1 if 'lie' in x else 0)

print(f"Class distribution:\n{df['False'].value_counts()}")

# ==============================================================================
# 3. VIDEO-LEVEL TRAIN / TEST SPLIT  (stratified by class)
# ==============================================================================

videos = pd.DataFrame({
    'video name': df['video name'].unique()
})
videos['False'] = videos['video name'].apply(lambda x: 1 if 'lie' in x else 0)

train_videos, test_videos = train_test_split(
    videos,
    test_size=TEST_SIZE,
    stratify=videos['False'],
    random_state=RANDOM_STATE
)

train_df = df[df['video name'].isin(train_videos['video name'])].reset_index(drop=True)
test_df  = df[df['video name'].isin(test_videos['video name'])].reset_index(drop=True)

print(f"Train frames: {len(train_df)} | Test frames: {len(test_df)}")

# ==============================================================================
# 4. FEATURE NORMALIZATION  (fit on train only — prevents leakage)
# ==============================================================================

scaler = StandardScaler()
train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

# ==============================================================================
# 5. SEQUENCE CREATION
# ==============================================================================

def create_sequences(data: np.ndarray, labels: np.ndarray, seq_len: int):
    """
    Sliding window sequence builder.
    Returns:
        x: (n_windows, seq_len, n_features)
        y: (n_windows,)  — label at the last frame of each window
    """
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i : i + seq_len])
        y.append(labels[i + seq_len - 1])
    return np.array(x), np.array(y)


def build_sequences_from_df(dataframe: pd.DataFrame, seq_len: int = SEQ_LEN):
    """Build sequences per video to avoid cross-video window contamination."""
    all_x, all_y = [], []
    for vid, group in dataframe.groupby('video name'):
        group = group.sort_values('time_s')
        features = group[FEATURE_COLS].values
        labels   = group['False'].values
        x, y = create_sequences(features, labels, seq_len)
        if len(x) > 0:
            all_x.append(x)
            all_y.append(y)
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


# --- FIX: sequences built AFTER split (no leakage) ---
X_train, y_train = build_sequences_from_df(train_df)
X_test,  y_test  = build_sequences_from_df(test_df)

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

# ==============================================================================
# 6. CLASS WEIGHT (handles imbalanced classes)
# ==============================================================================

class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight = dict(enumerate(class_weights_arr))
print(f"Class weights: {class_weight}")

# ==============================================================================
# 7. MODEL DEFINITION — LSTM + MULTI-HEAD ATTENTION
# ==============================================================================

def build_lstm_attention_model(seq_len: int, n_features: int) -> Model:
    """
    LSTM + Multi-Head Self-Attention with residual connections.
    Improvements over original:
      - MultiHeadAttention (4 heads) instead of basic Attention
      - Residual connection + LayerNorm (transformer-style)
      - BatchNormalization for training stability
    """
    inputs = Input(shape=(seq_len, n_features), name='input')

    # Encoder stack
    x = LSTM(64, return_sequences=True, name='lstm_1')(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=True, name='lstm_2')(x)
    x = Dropout(0.3)(x)

    # Multi-head self-attention
    attn_out = MultiHeadAttention(num_heads=4, key_dim=8, name='mha')(x, x)
    x = Add()([x, attn_out])           # residual
    x = LayerNormalization()(x)

    # Aggregate sequence → vector
    x = GlobalAveragePooling1D()(x)

    # Classifier head
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs, name='LSTM_MultiHeadAttention')
    return model


model = build_lstm_attention_model(SEQ_LEN, N_FEATURES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# ==============================================================================
# 8. CALLBACKS
# ==============================================================================

callbacks = [
    EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]

# ==============================================================================
# 9. TRAINING
# ==============================================================================

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ==============================================================================
# 10. EVALUATION
# ==============================================================================

y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)

print("\n========== TEST SET RESULTS ==========")
print(classification_report(y_test, y_pred, target_names=['Truth (0)', 'Lie (1)']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
'''plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Truth', 'Lie'],
            yticklabels=['Truth', 'Lie'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(history.history['auc'], label='Train AUC')
axes[1].plot(history.history['val_auc'], label='Val AUC')
axes[1].set_title('AUC'); axes[1].legend()
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()'''


# ==============================================================================
# ALTERNATIVE ARCHITECTURES  (drop-in replacements for build_lstm_attention_model)
# ==============================================================================

# ------------------------------------------------------------------------------
# ARCH 1: Bidirectional LSTM
# Best for: capturing past AND future context within each sequence
# ------------------------------------------------------------------------------
def build_bilstm_model(seq_len, n_features):
    inputs = Input(shape=(seq_len, n_features))
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='BiLSTM')


# ------------------------------------------------------------------------------
# ARCH 2: Temporal CNN (TCN-style)
# Best for: fast training, excellent at local pattern detection
# No vanishing gradient problem
# ------------------------------------------------------------------------------
def build_tcn_model(seq_len, n_features):
    inputs = Input(shape=(seq_len, n_features))

    # Dilated causal convolutions
    x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, padding='causal', dilation_rate=4, activation='relu')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='TCN')


# ------------------------------------------------------------------------------
# ARCH 3: Pure Transformer Encoder
# Best for: global attention across the full sequence, state-of-the-art
# ------------------------------------------------------------------------------
def transformer_encoder_block(x, num_heads=4, key_dim=16, ff_dim=64, dropout=0.2):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = Dropout(dropout)(attn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(x.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    return x

def build_transformer_model(seq_len, n_features):
    inputs = Input(shape=(seq_len, n_features))

    x = Dense(64)(inputs)                          # project to d_model
    x = transformer_encoder_block(x, num_heads=4, key_dim=16, ff_dim=128)
    x = transformer_encoder_block(x, num_heads=4, key_dim=16, ff_dim=128)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='Transformer')


# ------------------------------------------------------------------------------
# ARCH 4: CNN + LSTM Hybrid
# Best for: local feature extraction (CNN) + sequential modeling (LSTM)
# Often outperforms pure LSTM on biosignal/AU data
# ------------------------------------------------------------------------------
def build_cnn_lstm_model(seq_len, n_features):
    inputs = Input(shape=(seq_len, n_features))

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='CNN_LSTM')


# ==============================================================================
# QUICK COMPARISON  (optional — trains all 4 archs and compares AUC)
# ==============================================================================

def compile_and_evaluate(model, X_tr, y_tr, X_te, y_te, cw, epochs=100):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    cb = [EarlyStopping(monitor='val_auc', mode='max', patience=20,
                        restore_best_weights=True, verbose=0)]
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
              validation_split=0.15, class_weight=cw, callbacks=cb, verbose=1)
    y_prob = model.predict(X_te, verbose=0).flatten()
    auc = roc_auc_score(y_te, y_prob)
    return auc


if __name__ == '__main__':
    print("\n===== ARCHITECTURE COMPARISON =====")
    archs = {
        'LSTM + MultiHeadAttention': build_lstm_attention_model,
        'BiLSTM':                    build_bilstm_model,
        'Temporal CNN':              build_tcn_model,
        'Transformer':               build_transformer_model,
        'CNN + LSTM':                build_cnn_lstm_model,
    }

    results = {}
    for name, builder in archs.items():
        print(f"Training {name}...")
        m = builder(SEQ_LEN, N_FEATURES)
        auc = compile_and_evaluate(m, X_train, y_train, X_test, y_test,class_weight, epochs=3)
        results[name] = auc
        print(f"  AUC = {auc:.4f}")

    print("\n--- Final Ranking ---")
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {auc:.4f}  {name}")