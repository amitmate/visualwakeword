#!/usr/bin/env python3
"""
Advanced model compression: pruning, clustering, and combined techniques.
These require a fine-tuning step (even a short one with random data) to recover accuracy,
but can achieve significantly smaller models.
"""
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_model_optimization as tfmot


H5_PATH = "modelVisualWakeWord.h5"
INPUT_SHAPE = (96, 96, 3)
ORIGINAL_TFLITE_SIZE = os.path.getsize("modelVisualWakeWord.tflite")

# Synthetic calibration/fine-tuning data (no COCO needed)
NUM_SAMPLES = 500


def make_synthetic_data():
    """Create synthetic data for calibration and short fine-tuning."""
    x = np.random.uniform(-1.0, 1.0, (NUM_SAMPLES, *INPUT_SHAPE)).astype(np.float32)
    y = np.random.randint(0, 2, (NUM_SAMPLES, 1)).astype(np.float32)
    return x, y


def representative_dataset():
    for _ in range(100):
        yield [np.random.uniform(-1.0, 1.0, (1, *INPUT_SHAPE)).astype(np.float32)]


def to_tflite_int8(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    return converter.convert()


def to_tflite_dynamic(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def save(name, tflite_bytes, output_dir="compressed_models"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"model_{name}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_bytes)
    return path, len(tflite_bytes)


def report(name, size):
    reduction = (1 - size / ORIGINAL_TFLITE_SIZE) * 100
    print(f"  {name:<40} {size:>8,} bytes ({size/1024:>5.1f} KB) | {reduction:>+6.1f}% vs original")
    return (name, size, reduction)


def try_pruning(model, x, y):
    """Apply magnitude-based weight pruning then quantize."""
    print("\n--- Pruning (50% sparsity) + Quantization ---")
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.50, begin_step=0)
    }

    try:
        pruned_model = prune_low_magnitude(model, **pruning_params)
        pruned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        pruned_model.fit(x, y, epochs=2, batch_size=32, callbacks=callbacks, verbose=0)

        stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)

        tflite_bytes = to_tflite_int8(stripped)
        path, size = save("pruned_50_int8", tflite_bytes)
        return report("pruned_50% + int8 quant", size)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def try_pruning_high(model, x, y):
    """Apply aggressive 75% pruning then quantize."""
    print("\n--- Pruning (75% sparsity) + Quantization ---")
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, begin_step=0)
    }

    try:
        pruned_model = prune_low_magnitude(model, **pruning_params)
        pruned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        pruned_model.fit(x, y, epochs=2, batch_size=32, callbacks=callbacks, verbose=0)

        stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)

        tflite_bytes = to_tflite_int8(stripped)
        path, size = save("pruned_75_int8", tflite_bytes)
        return report("pruned_75% + int8 quant", size)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def try_clustering(model, x, y):
    """Weight clustering: reduce unique weight values to 16 centroids per layer."""
    print("\n--- Weight Clustering (16 centroids) + Quantization ---")
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': CentroidInitialization.LINEAR
    }

    try:
        clustered_model = cluster_weights(model, **clustering_params)
        clustered_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        clustered_model.fit(x, y, epochs=1, batch_size=32, verbose=0)

        stripped = tfmot.clustering.keras.strip_clustering(clustered_model)

        tflite_bytes = to_tflite_int8(stripped)
        path, size = save("clustered_16_int8", tflite_bytes)
        return report("clustered_16 + int8 quant", size)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def try_clustering_8(model, x, y):
    """Weight clustering with only 8 centroids for maximum compression."""
    print("\n--- Weight Clustering (8 centroids) + Quantization ---")
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
        'number_of_clusters': 8,
        'cluster_centroids_init': CentroidInitialization.LINEAR
    }

    try:
        clustered_model = cluster_weights(model, **clustering_params)
        clustered_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        clustered_model.fit(x, y, epochs=1, batch_size=32, verbose=0)

        stripped = tfmot.clustering.keras.strip_clustering(clustered_model)

        tflite_bytes = to_tflite_int8(stripped)
        path, size = save("clustered_8_int8", tflite_bytes)
        return report("clustered_8 + int8 quant", size)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def try_gzip_comparison():
    """Show theoretical compression floor via gzip."""
    import gzip
    print("\n--- GZIP compression of existing .tflite (theoretical floor) ---")

    with open("modelVisualWakeWord.tflite", "rb") as f:
        original = f.read()

    for level in [6, 9]:
        compressed = gzip.compress(original, compresslevel=level)
        path = f"compressed_models/model_original_gzip{level}.tflite.gz"
        os.makedirs("compressed_models", exist_ok=True)
        with open(path, "wb") as f:
            f.write(compressed)
        report(f"original + gzip level {level}", len(compressed))

    # Also gzip each of the compressed models
    for fname in sorted(os.listdir("compressed_models")):
        if fname.endswith(".tflite"):
            fpath = os.path.join("compressed_models", fname)
            with open(fpath, "rb") as f:
                data = f.read()
            compressed = gzip.compress(data, compresslevel=9)
            gz_path = fpath + ".gz"
            with open(gz_path, "wb") as f:
                f.write(compressed)
            report(f"{fname} + gzip9", len(compressed))


def main():
    print(f"Loading {H5_PATH}...")
    model = tf.keras.models.load_model(H5_PATH, compile=False)
    params = model.count_params()
    print(f"Parameters: {params:,}")
    print(f"Original .tflite: {ORIGINAL_TFLITE_SIZE:,} bytes ({ORIGINAL_TFLITE_SIZE/1024:.1f} KB)")

    x, y = make_synthetic_data()

    results = []

    r = try_pruning(model, x, y)
    if r: results.append(r)

    # Reload model fresh for each technique
    model = tf.keras.models.load_model(H5_PATH, compile=False)
    r = try_pruning_high(model, x, y)
    if r: results.append(r)

    model = tf.keras.models.load_model(H5_PATH, compile=False)
    r = try_clustering(model, x, y)
    if r: results.append(r)

    model = tf.keras.models.load_model(H5_PATH, compile=False)
    r = try_clustering_8(model, x, y)
    if r: results.append(r)

    try_gzip_comparison()

    print(f"\n{'='*70}")
    print("SUMMARY (sorted by size)")
    print(f"{'='*70}")
    print(f"  {'existing .tflite':<40} {ORIGINAL_TFLITE_SIZE:>8,} bytes ({ORIGINAL_TFLITE_SIZE/1024:.1f} KB)")
    for name, size, reduction in sorted(results, key=lambda x: x[1]):
        print(f"  {name:<40} {size:>8,} bytes ({size/1024:.1f} KB) | {reduction:>+6.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
