# train_my_app_model.py

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, callbacks, optimizers

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# GPU setup (use GPU if available)
# -----------------------------
try:
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		for gpu in gpus:
			try:
				tf.config.experimental.set_memory_growth(gpu, True)
			except Exception:
				pass
		print(f"GPUs detected: {len(gpus)} (using GPU acceleration)")
		# Optional: enable mixed precision for speed on modern GPUs
		try:
			from tensorflow.keras import mixed_precision
			mixed_precision.set_global_policy('mixed_float16')
		except Exception:
			pass
	else:
		print("No GPU detected. Training will use CPU.")
except Exception as e:
	print(f"GPU setup warning: {e}")

# -----------------------------
# Paths and output setup
# -----------------------------
PROJECT_ROOT = os.path.abspath('.')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
MODEL_NAME = 'My-App-Model.h5'
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_NAME)
CONF_MATRIX_PNG = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
CSV_LOG_PATH = os.path.join(LOGS_DIR, 'training_log.csv')

for d in [OUTPUT_DIR, PLOTS_DIR, LOGS_DIR]:
	os.makedirs(d, exist_ok=True)

# -----------------------------
# Data loading and preprocessing
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

class_names = [
	'airplane', 'automobile', 'bird', 'cat', 'deer',
	'dog', 'frog', 'horse', 'ship', 'truck'
]

# -----------------------------
# Model definition (CNN)
# -----------------------------

def build_model(input_shape=(32, 32, 3), num_classes=10, use_mixed_precision=False):
	inputs = layers.Input(shape=input_shape)

	# In-graph data augmentation
	x = layers.RandomFlip('horizontal')(inputs)
	x = layers.RandomRotation(0.1)(x)
	x = layers.RandomZoom(0.1)(x)

	# Conv blocks
	def conv_block(x, filters):
		x = layers.Conv2D(filters, 3, padding='same')(x)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU()(x)
		x = layers.Conv2D(filters, 3, padding='same')(x)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU()(x)
		x = layers.MaxPooling2D()(x)
		x = layers.Dropout(0.25)(x)
		return x

	x = conv_block(x, 64)
	x = conv_block(x, 128)
	x = conv_block(x, 256)

	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Dense(256)(x)
	x = layers.ReLU()(x)
	x = layers.Dropout(0.3)(x)

	# If mixed precision is enabled, use float32 for final layer for numerical stability
	final_dtype = 'float32'
	outputs = layers.Dense(num_classes, activation='softmax', dtype=final_dtype)(x)

	model = models.Model(inputs, outputs, name='MyAppCIFAR10')
	return model

use_mixed = False
try:
	from tensorflow.keras import mixed_precision
	use_mixed = (mixed_precision.global_policy().compute_dtype == 'float16')
except Exception:
	use_mixed = False

model = build_model(use_mixed_precision=use_mixed)

# Optimizer and compile
base_lr = 1e-3
opt = optimizers.Adam(learning_rate=base_lr)
model.compile(
	optimizer=opt,
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
cb = [
	callbacks.ModelCheckpoint(
		BEST_MODEL_PATH,
		monitor='val_accuracy',
		save_best_only=True,
		save_weights_only=False,
		mode='max',
		verbose=1
	),
	callbacks.ReduceLROnPlateau(
		monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
	),
	callbacks.EarlyStopping(
		monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1
	),
	callbacks.CSVLogger(CSV_LOG_PATH)
]

# -----------------------------
# Training
# -----------------------------
epochs = 50
batch_size = 128

history = model.fit(
	x_train, y_train_cat,
	validation_split=0.1,
	epochs=epochs,
	batch_size=batch_size,
	callbacks=cb,
	verbose=1
)

# Ensure best model is saved
model.save(BEST_MODEL_PATH)
print(f"Saved model to: {BEST_MODEL_PATH}")

# -----------------------------
# Evaluation
# -----------------------------
print("Evaluating on test set...")
loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# Predictions and reports
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CIFAR-10 Confusion Matrix - My-App-Model')
plt.tight_layout()
plt.savefig(CONF_MATRIX_PNG)
print(f"Confusion matrix plot saved to: {CONF_MATRIX_PNG}")
plt.close()

# -----------------------------
# Export label mapping for inference
# -----------------------------
with open(os.path.join(OUTPUT_DIR, 'class_names.txt'), 'w') as f:
	for name in class_names:
		f.write(name + '\n')
print(f"Class names saved to: {os.path.join(OUTPUT_DIR, 'class_names.txt')}")
