#!/usr/bin/env python3
"""
Evaluate original and FP4-quantized Visual Wake Word models on COCO minival.

Run this on a machine with COCO val2014 data available.

Usage:
    python eval_fp4_coco.py \
        --data-dir /path/to/coco/raw-data \
        --minival-ids mscocominival.txt

The data-dir should contain:
    annotations/instances_val2014.json
    val2014/COCO_val2014_*.jpg
"""
import os
import sys
import argparse
import struct
import gzip
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import cv2

# Try pycocotools, fall back to local coco module
try:
    from pycocotools.coco import COCO
except ImportError:
    try:
        from coco import COCO
    except ImportError:
        print("ERROR: Install pycocotools: pip install pycocotools")
        sys.exit(1)

IMAGESIZE = 96
PERSONTHR = 2048
BLOCK_SIZE = 32

# ─── FP4 E2M1 codec (same as compress_fp4.py) ──────────────────────────────

FP4_TABLE = np.zeros(16, dtype=np.float32)
for code in range(16):
    sign = -1.0 if (code >> 3) & 1 else 1.0
    exp_bits = (code >> 1) & 0x3
    mant_bit = code & 0x1
    if exp_bits == 0:
        value = 0.5 * mant_bit
    else:
        value = 2.0 ** (exp_bits - 1) * (1.0 + 0.5 * mant_bit)
    FP4_TABLE[code] = sign * value


def unpack_fp4(data, count):
    arr = np.frombuffer(data, dtype=np.uint8)
    hi = (arr >> 4) & 0xF
    lo = arr & 0xF
    codes = np.empty(len(arr) * 2, dtype=np.uint8)
    codes[0::2] = hi
    codes[1::2] = lo
    return codes[:count]


def fp4_to_float(codes):
    return FP4_TABLE[codes]


def dequantize_block_mx_fp4(scale, codes):
    return scale * fp4_to_float(codes)


def dequantize_entry(entry):
    n = entry['n_elements']
    shape = entry['shape']
    scales = entry['scales']
    codes = entry['codes']
    pad_len = (BLOCK_SIZE - n % BLOCK_SIZE) % BLOCK_SIZE
    codes_padded = np.concatenate([codes, np.zeros(pad_len, dtype=np.uint8)])
    n_blocks = len(codes_padded) // BLOCK_SIZE
    result = np.empty(len(codes_padded), dtype=np.float32)
    for b in range(n_blocks):
        s = scales[b]
        c = codes_padded[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
        result[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE] = dequantize_block_mx_fp4(s, c)
    return result[:n].reshape(shape)


MAGIC = b'MXF4'

def load_fp4_binary(path):
    entries = []
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        n_entries = struct.unpack('<I', f.read(4))[0]
        for _ in range(n_entries):
            name_len = struct.unpack('<H', f.read(2))[0]
            name = f.read(name_len).decode('utf-8')
            ndims = struct.unpack('<B', f.read(1))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]
            n_elements = struct.unpack('<I', f.read(4))[0]
            n_scales = struct.unpack('<I', f.read(4))[0]
            scales = np.frombuffer(f.read(n_scales * 4), dtype=np.float32).copy()
            packed_len = struct.unpack('<I', f.read(4))[0]
            packed_data = f.read(packed_len)
            codes = unpack_fp4(packed_data, n_elements)
            entries.append({
                'name': name, 'shape': shape, 'n_elements': n_elements,
                'scales': scales, 'codes': codes,
            })
    return entries


def apply_fp4_weights(model, entries):
    new_weights = []
    for idx in range(len(model.weights)):
        new_weights.append(dequantize_entry(entries[idx]))
    model.set_weights(new_weights)


# ─── COCO data loading (adapted from coco_minieval88.py) ────────────────────

def resize2SquareKeepingAspectRatio(img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    dif = max(h, w)
    x_pos = int((dif - w) / 2.)
    y_pos = int((dif - h) / 2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)


def checkPersonThreshold(anns):
    for ann in anns:
        if ann['area'] > PERSONTHR:
            return 1
    return 0


def read_integers(filename):
    with open(filename) as f:
        return [int(x) for x in f]


def load_minival(data_dir, minival_path):
    """Load COCO minival set, matching coco_minieval88.py logic."""
    dataType = 'val2014'
    annFile = os.path.join(data_dir, 'annotations', f'instances_{dataType}.json')

    if not os.path.exists(annFile):
        print(f"ERROR: Annotations not found at {annFile}")
        sys.exit(1)

    print(f"Loading COCO annotations from {annFile}...")
    coco = COCO(annFile)

    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)

    imgMVIds = read_integers(minival_path)
    personImgIds = list(set(imgMVIds) & set(imgIds))
    nonPersonImgIds = list(set(imgMVIds) - set(imgIds))

    print(f"Minival IDs: {len(imgMVIds)}")
    print(f"Person images (before threshold): {len(personImgIds)}")
    print(f"Non-person images: {len(nonPersonImgIds)}")

    # Count valid persons (area > threshold)
    numValidPersons = 0
    for imgId in personImgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
        anns = coco.loadAnns(annIds)
        if checkPersonThreshold(anns):
            numValidPersons += 1

    total = numValidPersons + len(nonPersonImgIds)
    print(f"Valid persons (area>{PERSONTHR}): {numValidPersons}")
    print(f"Total eval images: {total}")

    images = np.ones((total, IMAGESIZE, IMAGESIZE, 3), np.uint8)
    labels = np.zeros((total, 1), np.uint8)
    k = 0

    # Load person images
    for i, imgId in enumerate(personImgIds):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
        anns = coco.loadAnns(annIds)
        if not checkPersonThreshold(anns):
            continue

        name = os.path.join(data_dir, dataType, img['file_name'])
        if not os.path.exists(name):
            print(f"  WARNING: Image not found: {name}")
            continue

        img1 = cv2.imread(name)
        if img1 is None:
            continue
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        if len(img1.shape) != 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

        images[k] = resize2SquareKeepingAspectRatio(img1, IMAGESIZE, cv2.INTER_AREA)
        labels[k] = 1
        k += 1

        if (i + 1) % 500 == 0:
            print(f"  Loaded {k} person images...")

    print(f"  Total valid person images loaded: {k}")
    person_count = k

    # Load non-person images
    for i, imgId in enumerate(nonPersonImgIds):
        img = coco.loadImgs(imgId)[0]
        name = os.path.join(data_dir, dataType, img['file_name'])
        if not os.path.exists(name):
            continue

        img1 = cv2.imread(name)
        if img1 is None:
            continue
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        if len(img1.shape) != 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

        images[k] = resize2SquareKeepingAspectRatio(img1, IMAGESIZE, cv2.INTER_AREA)
        labels[k] = 0
        k += 1

        if (i + 1) % 500 == 0:
            print(f"  Loaded {k - person_count} non-person images...")

    print(f"  Total non-person images loaded: {k - person_count}")
    print(f"  Total images loaded: {k}")

    # Trim to actual loaded count
    images = images[:k]
    labels = labels[:k]

    # Shuffle
    idx = np.arange(k)
    np.random.seed(42)
    np.random.shuffle(idx)
    images = images[idx]
    labels = labels[idx]

    # Normalize to [-1, 1]
    images = images.astype('float32') / 127.5 - 1.0

    return images, labels


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model, x_test, y_test, name="Model"):
    """Evaluate a Keras model and return accuracy."""
    preds = model.predict(x_test, batch_size=64, verbose=0)
    pred_labels = (preds > 0.5).astype(np.uint8)
    correct = np.sum(pred_labels == y_test)
    total = len(y_test)
    acc = correct / total * 100
    print(f"  {name}: {correct}/{total} correct = {acc:.2f}%")
    return acc


def evaluate_tflite(tflite_path, x_test, y_test, name="TFLite"):
    """Evaluate a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    correct = 0
    total = len(y_test)

    for i in range(total):
        img = x_test[i:i+1]
        if input_dtype == np.int8:
            # Quantize input
            input_scale = input_details[0]['quantization_parameters']['scales'][0]
            input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
            img = (img / input_scale + input_zp).astype(np.int8)

        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)

        if output_details[0]["dtype"] == np.int8:
            output_scale = output_details[0]['quantization_parameters']['scales'][0]
            output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
            output = (output.astype(np.float32) - output_zp) * output_scale

        pred = 1 if output[0] > 0.5 else 0
        if pred == y_test[i]:
            correct += 1

        if (i + 1) % 1000 == 0:
            print(f"    {name}: {i+1}/{total} processed, running acc={correct/(i+1)*100:.1f}%")

    acc = correct / total * 100
    print(f"  {name}: {correct}/{total} correct = {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate VWW models on COCO minival")
    parser.add_argument("--data-dir", default="/home/amit_mate2009/coco/raw-data",
                        help="Path to COCO raw-data directory")
    parser.add_argument("--minival-ids", default="mscocominival.txt",
                        help="Path to minival image IDs file")
    parser.add_argument("--h5-model", default="modelVisualWakeWord.h5",
                        help="Path to original Keras H5 model")
    parser.add_argument("--tflite-model", default="modelVisualWakeWord.tflite",
                        help="Path to original TFLite model")
    parser.add_argument("--fp4-bin", default="compressed_models/model_mxfp4.fp4bin",
                        help="Path to FP4 binary model")
    parser.add_argument("--fp4-tflite", default="compressed_models/model_mxfp4_int8.tflite",
                        help="Path to FP4→int8 TFLite model")
    args = parser.parse_args()

    # Load eval data
    x_test, y_test = load_minival(args.data_dir, args.minival_ids)
    print(f"\nEval set: {x_test.shape[0]} images, {int(np.sum(y_test))} persons, "
          f"{x_test.shape[0] - int(np.sum(y_test))} non-persons\n")

    results = []

    # 1. Original H5 model
    print("=" * 60)
    print("Evaluating original H5 model...")
    model_orig = tf.keras.models.load_model(args.h5_model, compile=False)
    acc = evaluate_model(model_orig, x_test, y_test, "Original H5")
    results.append(("Original H5", acc))

    # 2. Original TFLite model
    if os.path.exists(args.tflite_model):
        print("\nEvaluating original TFLite model...")
        acc = evaluate_tflite(args.tflite_model, x_test, y_test, "Original TFLite")
        results.append(("Original TFLite", acc))

    # 3. FP4-dequantized H5 model
    if os.path.exists(args.fp4_bin):
        print("\nEvaluating FP4-dequantized model...")
        model_fp4 = tf.keras.models.load_model(args.h5_model, compile=False)
        entries = load_fp4_binary(args.fp4_bin)
        apply_fp4_weights(model_fp4, entries)
        acc = evaluate_model(model_fp4, x_test, y_test, "FP4 (dequantized)")
        results.append(("FP4 dequantized", acc))

    # 4. FP4→int8 TFLite model
    if os.path.exists(args.fp4_tflite):
        print("\nEvaluating FP4→int8 TFLite model...")
        acc = evaluate_tflite(args.fp4_tflite, x_test, y_test, "FP4→int8 TFLite")
        results.append(("FP4→int8 TFLite", acc))

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for name, acc in results:
        size_info = ""
        if "Original H5" in name:
            size_info = f"  ({os.path.getsize(args.h5_model)/1024:.0f} KB)"
        elif "Original TFLite" in name and os.path.exists(args.tflite_model):
            size_info = f"  ({os.path.getsize(args.tflite_model)/1024:.0f} KB)"
        elif "FP4 dequantized" in name and os.path.exists(args.fp4_bin):
            size_info = f"  ({os.path.getsize(args.fp4_bin)/1024:.0f} KB)"
        elif "FP4→int8" in name and os.path.exists(args.fp4_tflite):
            size_info = f"  ({os.path.getsize(args.fp4_tflite)/1024:.0f} KB)"
        print(f"  {name:<25} {acc:>6.2f}%{size_info}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
