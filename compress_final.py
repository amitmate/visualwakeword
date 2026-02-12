#!/usr/bin/env python3
"""
Model compression approaches for the Visual Wake Word model.
Focuses on what we can achieve without retraining.
"""
import os
import gzip
import struct
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import tensorflow as tf


H5_PATH = "modelVisualWakeWord.h5"
INPUT_SHAPE = (96, 96, 3)
ORIGINAL_SIZE = os.path.getsize("modelVisualWakeWord.tflite")
OUT_DIR = "compressed_models"
os.makedirs(OUT_DIR, exist_ok=True)


def representative_dataset():
    for _ in range(200):
        yield [np.random.uniform(-1.0, 1.0, (1, *INPUT_SHAPE)).astype(np.float32)]


def save(name, data):
    path = os.path.join(OUT_DIR, f"model_{name}.tflite")
    with open(path, "wb") as f:
        f.write(data)
    return path, len(data)


def report(label, size):
    pct = (1 - size / ORIGINAL_SIZE) * 100
    kb = size / 1024
    print(f"  {label:<45} {size:>7,} B  ({kb:>5.1f} KB)  {pct:>+6.1f}%")
    return (label, size, pct)


def main():
    print(f"Original .tflite: {ORIGINAL_SIZE:,} bytes ({ORIGINAL_SIZE/1024:.1f} KB)\n")

    model = tf.keras.models.load_model(H5_PATH, compile=False)
    print(f"Model params: {model.count_params():,}\n")

    results = []

    # 1. Dynamic range quantization (weights-only int8)
    print("1. Dynamic range quantization (weights→int8)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    path, size = save("dynamic_range", tflite)
    results.append(report("dynamic_range (weights int8)", size))

    # 2. Full integer quantization with float I/O
    print("2. Full int8 quantization (float32 I/O)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite = converter.convert()
    path, size = save("full_int8_floatio", tflite)
    results.append(report("full_int8 (float32 I/O)", size))

    # 3. Full integer quantization with int8 I/O
    print("3. Full int8 quantization (int8 I/O)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite = converter.convert()
    path, size = save("full_int8_int8io", tflite)
    results.append(report("full_int8 (int8 I/O)", size))

    # 4. Quantize + strip buffers alignment padding via re-read
    # The TFLite format aligns buffers to 16-byte boundaries. We can try
    # to minimize this by re-serializing.
    print("4. Checking gzip compressibility of each variant...")

    # gzip the original
    with open("modelVisualWakeWord.tflite", "rb") as f:
        orig_data = f.read()
    gz_orig = gzip.compress(orig_data, compresslevel=9)
    results.append(report("original .tflite + gzip9", len(gz_orig)))
    with open(os.path.join(OUT_DIR, "model_original.tflite.gz"), "wb") as f:
        f.write(gz_orig)

    # gzip each variant
    for fname in ["model_dynamic_range.tflite", "model_full_int8_floatio.tflite", "model_full_int8_int8io.tflite"]:
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, "rb") as f:
            data = f.read()
        gz = gzip.compress(data, compresslevel=9)
        gz_path = fpath + ".gz"
        with open(gz_path, "wb") as f:
            f.write(gz)
        results.append(report(f"{fname} + gzip9", len(gz)))

    # 5. Attempt weight-sharing via manual quantization to 4-bit range
    # Pack weights more tightly by converting to uint8 with limited range
    print("\n5. Experimental: strip + repack the full_int8 model...")
    # Read the int8 model and strip any signature/metadata
    with open(os.path.join(OUT_DIR, "model_full_int8_int8io.tflite"), "rb") as f:
        int8_data = f.read()

    # TFLite files are flatbuffers - check for trailing padding
    # The actual model ends at the flatbuffer size indicated in the first 4 bytes (offset)
    # Flatbuffer: first 4 bytes = offset to root table
    actual_end = len(int8_data)
    # Check for trailing zeros
    while actual_end > 0 and int8_data[actual_end - 1] == 0:
        actual_end -= 1
    # Flatbuffers need the full buffer, but we can at least measure padding
    padding = len(int8_data) - actual_end
    print(f"  Trailing zero padding: {padding} bytes")

    # Final summary
    print(f"\n{'='*75}")
    print(f"  RESULTS SUMMARY (original = {ORIGINAL_SIZE:,} bytes / {ORIGINAL_SIZE/1024:.1f} KB)")
    print(f"{'='*75}")
    for label, size, pct in sorted(results, key=lambda x: x[1]):
        kb = size / 1024
        print(f"  {label:<45} {size:>7,} B  ({kb:>5.1f} KB)  {pct:>+6.1f}%")
    print(f"{'='*75}")

    # Best raw (uncompressed) and best compressed
    raw_results = [(l, s, p) for l, s, p in results if "gzip" not in l]
    gz_results = [(l, s, p) for l, s, p in results if "gzip" in l]

    if raw_results:
        best_raw = min(raw_results, key=lambda x: x[1])
        print(f"\n  Best raw:        {best_raw[0]} → {best_raw[1]:,} B ({best_raw[1]/1024:.1f} KB)")
    if gz_results:
        best_gz = min(gz_results, key=lambda x: x[1])
        print(f"  Best compressed: {best_gz[0]} → {best_gz[1]:,} B ({best_gz[1]/1024:.1f} KB)")


if __name__ == "__main__":
    main()
