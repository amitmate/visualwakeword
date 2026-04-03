#!/usr/bin/env python3
"""
Compress modelVisualWakeWord.h5 using multiple TFLite quantization strategies.
Outputs size comparisons and compressed .tflite files.
"""
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


H5_PATH = "modelVisualWakeWord.h5"
INPUT_SHAPE = (1, 96, 96, 3)


def representative_dataset():
    """Generate random calibration data (mimics normalized input range [-1, 1])."""
    for _ in range(100):
        yield [np.random.uniform(-1.0, 1.0, INPUT_SHAPE).astype(np.float32)]


def convert_baseline(model):
    """Baseline: default TFLite conversion (no quantization)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_dynamic_range(model):
    """Dynamic range quantization: weights → int8, activations float at runtime."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_float16(model):
    """Float16 quantization: weights → float16."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_full_int8(model):
    """Full integer quantization: weights + activations → int8."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def convert_full_int8_float_io(model):
    """Full integer quantization with float32 I/O (easier to use, same weight compression)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Keep float32 input/output for compatibility
    return converter.convert()


def save_and_report(name, tflite_bytes, output_dir="compressed_models"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"model_{name}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_bytes)
    size = len(tflite_bytes)
    return path, size


def main():
    print(f"Loading {H5_PATH}...")
    model = tf.keras.models.load_model(H5_PATH, compile=False)
    model.summary()

    original_size = os.path.getsize("modelVisualWakeWord.tflite")
    print(f"\n{'='*60}")
    print(f"Original .tflite size: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    print(f"{'='*60}\n")

    strategies = [
        ("baseline_no_quant", convert_baseline),
        ("dynamic_range_int8", convert_dynamic_range),
        ("float16", convert_float16),
        ("full_int8_float_io", convert_full_int8_float_io),
        ("full_int8", convert_full_int8),
    ]

    results = []
    for name, fn in strategies:
        print(f"Converting: {name}...")
        try:
            tflite_bytes = fn(model)
            path, size = save_and_report(name, tflite_bytes)
            reduction = (1 - size / original_size) * 100
            results.append((name, size, reduction, path))
            print(f"  → {size:,} bytes ({size/1024:.1f} KB) | {reduction:+.1f}% vs original\n")
        except Exception as e:
            print(f"  → FAILED: {e}\n")
            results.append((name, None, None, None))

    print(f"\n{'='*60}")
    print(f"{'Strategy':<30} {'Size':>10} {'KB':>8} {'vs Original':>12}")
    print(f"{'-'*60}")
    print(f"{'existing .tflite':<30} {original_size:>10,} {original_size/1024:>8.1f} {'baseline':>12}")
    for name, size, reduction, path in results:
        if size is not None:
            print(f"{name:<30} {size:>10,} {size/1024:>8.1f} {reduction:>+11.1f}%")
        else:
            print(f"{name:<30} {'FAILED':>10}")
    print(f"{'='*60}")

    # Pick the smallest successful result
    successful = [(n, s, r, p) for n, s, r, p in results if s is not None]
    if successful:
        best = min(successful, key=lambda x: x[1])
        print(f"\nSmallest model: {best[0]} at {best[1]:,} bytes ({best[1]/1024:.1f} KB)")
        print(f"Saved to: {best[3]}")


if __name__ == "__main__":
    main()
