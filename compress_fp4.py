#!/usr/bin/env python3
"""
FP4 (E2M1) Microscaling Weight Quantization for the Visual Wake Word model.

Implements the OCP Microscaling FP4 format:
  - Element format: 4-bit floating point with E2M1 layout (1 sign, 2 exponent, 1 mantissa)
  - Block size: 32 elements share one 8-bit scale factor (E8M0)
  - Effective bits per weight: ~4.25 (4 bits + amortized scale)

Produces:
  1. A compact binary file (.fp4bin) with packed weights (2 per byte) + scales
  2. A loader that reconstructs the Keras model from the .fp4bin file
  3. Accuracy comparison between original and FP4-quantized models
"""
import os
import struct
import gzip
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

H5_PATH = "modelVisualWakeWord.h5"
ORIGINAL_TFLITE = "modelVisualWakeWord.tflite"
INPUT_SHAPE = (96, 96, 3)
BLOCK_SIZE = 32  # MX block size per the OCP spec
OUT_DIR = "compressed_models"

# ─── FP4 E2M1 codec ─────────────────────────────────────────────────────────
# E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit → 4 bits total
# Representable values (positive): 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
# With sign: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} → 16 total codes

# Build lookup table: 4-bit code → float value
FP4_TABLE = np.zeros(16, dtype=np.float32)
for code in range(16):
    sign = -1.0 if (code >> 3) & 1 else 1.0
    exp_bits = (code >> 1) & 0x3
    mant_bit = code & 0x1
    if exp_bits == 0:  # subnormal
        value = 0.5 * mant_bit  # 0.0 or 0.5
    else:
        value = 2.0 ** (exp_bits - 1) * (1.0 + 0.5 * mant_bit)
    FP4_TABLE[code] = sign * value

# Inverse: float → nearest FP4 code
_POS_VALUES = FP4_TABLE[:8]  # positive codes 0..7


def float_to_fp4(x):
    """Quantize a float to the nearest FP4 E2M1 code (0-15)."""
    sign = 0
    if x < 0:
        sign = 8
        x = -x
    # Find nearest positive FP4 value
    idx = int(np.argmin(np.abs(_POS_VALUES - x)))
    return sign | idx


float_to_fp4_vec = np.vectorize(float_to_fp4, otypes=[np.uint8])


def fp4_to_float(codes):
    """Decode FP4 codes back to float32."""
    return FP4_TABLE[codes]


# ─── Microscaling: block-wise scale + FP4 ────────────────────────────────────

def quantize_block_mx_fp4(block):
    """Quantize a flat block of floats using microscaling FP4.

    Returns: (scale_e8m0: float32, fp4_codes: uint8 array)
    The scale is the power-of-2 that maps max(|block|) into FP4 range [0, 6.0].
    """
    amax = np.max(np.abs(block))
    if amax == 0:
        return 1.0, np.zeros(len(block), dtype=np.uint8)

    # E8M0 scale: power-of-2 only (per MX spec)
    # We want scale * 6.0 >= amax  →  scale >= amax / 6.0
    # E8M0 encodes 2^(e-127), so pick e = ceil(log2(amax/6.0)) + 127
    raw_exp = np.ceil(np.log2(amax / 6.0))
    scale = 2.0 ** raw_exp  # power-of-2 scale

    # Scale block into FP4 representable range
    scaled = block / scale
    codes = float_to_fp4_vec(scaled)
    return scale, codes


def dequantize_block_mx_fp4(scale, codes):
    """Dequantize FP4 codes back to float32 using the block scale."""
    return scale * fp4_to_float(codes)


# ─── Pack/unpack FP4 codes (2 per byte) ──────────────────────────────────────

def pack_fp4(codes):
    """Pack uint8 FP4 codes (each 0-15) into bytes, 2 codes per byte."""
    n = len(codes)
    if n % 2 != 0:
        codes = np.append(codes, 0)  # pad
    packed = (codes[0::2] << 4) | codes[1::2]
    return packed.astype(np.uint8).tobytes()


def unpack_fp4(data, count):
    """Unpack bytes into uint8 FP4 codes."""
    arr = np.frombuffer(data, dtype=np.uint8)
    hi = (arr >> 4) & 0xF
    lo = arr & 0xF
    codes = np.empty(len(arr) * 2, dtype=np.uint8)
    codes[0::2] = hi
    codes[1::2] = lo
    return codes[:count]


# ─── Weight classification ──────────────────────────────────────────────────

# Only quantize conv/dense kernels. Keep BN params and biases at full precision.
_SKIP_KEYWORDS = {'bias', 'gamma', 'beta', 'moving_mean', 'moving_variance'}


def _should_quantize(weight_name):
    """Return True if this weight should be FP4-quantized (kernels only)."""
    name_lower = weight_name.lower()
    return all(kw not in name_lower for kw in _SKIP_KEYWORDS)


# ─── Quantize entire model ───────────────────────────────────────────────────

def quantize_weights_fp4(model):
    """Quantize kernel weights to MX-FP4 format, keep BN/bias at full precision.

    Returns list of entries per weight tensor. Entries with 'quantized'=True
    have FP4 scales+codes; entries with 'quantized'=False store raw float32.
    """
    entries = []
    total_original_bytes = 0
    total_fp4_bytes = 0
    total_kept_bytes = 0

    for idx, w in enumerate(model.weights):
        name = f"w{idx}_{w.name}"
        arr = w.numpy().flatten()
        shape = w.shape

        total_original_bytes += arr.nbytes  # float32

        if _should_quantize(w.name):
            # Quantize kernel weights to FP4
            pad_len = (BLOCK_SIZE - len(arr) % BLOCK_SIZE) % BLOCK_SIZE
            padded = np.concatenate([arr, np.zeros(pad_len, dtype=np.float32)])
            n_blocks = len(padded) // BLOCK_SIZE

            all_scales = []
            all_codes = []
            for b in range(n_blocks):
                block = padded[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
                scale, codes = quantize_block_mx_fp4(block)
                all_scales.append(scale)
                all_codes.append(codes)

            scales = np.array(all_scales, dtype=np.float32)
            codes = np.concatenate(all_codes)[:len(arr)]  # trim padding
            fp4_size = scales.nbytes + (len(arr) + 1) // 2
            total_fp4_bytes += fp4_size

            entries.append({
                'name': name,
                'shape': list(shape),
                'n_elements': len(arr),
                'quantized': True,
                'scales': scales,
                'codes': codes,
            })
            print(f"    FP4  {name:45s}  {arr.nbytes:>6} → {fp4_size:>6} bytes")
        else:
            # Keep BN params and biases at full float32 precision
            total_kept_bytes += arr.nbytes
            entries.append({
                'name': name,
                'shape': list(shape),
                'n_elements': len(arr),
                'quantized': False,
                'raw_data': arr.astype(np.float32),
            })
            print(f"    F32  {name:45s}  {arr.nbytes:>6} bytes (kept)")

    total_compressed = total_fp4_bytes + total_kept_bytes
    print(f"\n  Original weights:  {total_original_bytes:>8,} bytes (float32)")
    print(f"  FP4 kernels:       {total_fp4_bytes:>8,} bytes")
    print(f"  Full-prec BN/bias: {total_kept_bytes:>8,} bytes")
    print(f"  Total compressed:  {total_compressed:>8,} bytes")
    print(f"  Compression ratio: {total_original_bytes / total_compressed:.1f}x")

    return entries


def dequantize_entry(entry):
    """Reconstruct float32 weights from an entry (FP4-quantized or full precision)."""
    shape = entry['shape']

    if not entry.get('quantized', True):
        # Full precision entry — return raw data directly
        return entry['raw_data'].reshape(shape)

    n = entry['n_elements']
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


# ─── Binary file format ──────────────────────────────────────────────────────
# Header: magic(4) + version(1) + n_entries(4)
# Per entry: quantized_flag(1) + name_len(2) + name(utf8) + ndims(1) + shape(ndims*4) +
#   If quantized: n_elements(4) + n_blocks(4) + scales(n_blocks*4) + packed_codes(ceil(n/2))
#   If full prec: n_elements(4) + raw_data(n_elements*4)

MAGIC = b'MXF4'
FORMAT_VERSION = 2  # v2: mixed quantized + full-precision entries


def save_fp4_binary(entries, path):
    """Save weights to a compact binary file (FP4 kernels + float32 BN/bias)."""
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<B', FORMAT_VERSION))
        f.write(struct.pack('<I', len(entries)))

        for e in entries:
            quantized = e.get('quantized', True)
            f.write(struct.pack('<B', 1 if quantized else 0))

            name_bytes = e['name'].encode('utf-8')
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)

            shape = e['shape']
            f.write(struct.pack('<B', len(shape)))
            for s in shape:
                f.write(struct.pack('<I', s))

            f.write(struct.pack('<I', e['n_elements']))

            if quantized:
                scales = e['scales']
                f.write(struct.pack('<I', len(scales)))
                f.write(scales.tobytes())

                packed = pack_fp4(e['codes'])
                f.write(struct.pack('<I', len(packed)))
                f.write(packed)
            else:
                f.write(e['raw_data'].astype(np.float32).tobytes())

    return os.path.getsize(path)


def load_fp4_binary(path):
    """Load weights from a compact binary file."""
    entries = []
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        version = struct.unpack('<B', f.read(1))[0]
        assert version == FORMAT_VERSION, f"Unsupported format version: {version}"
        n_entries = struct.unpack('<I', f.read(4))[0]

        for _ in range(n_entries):
            quantized = struct.unpack('<B', f.read(1))[0] == 1

            name_len = struct.unpack('<H', f.read(2))[0]
            name = f.read(name_len).decode('utf-8')

            ndims = struct.unpack('<B', f.read(1))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]

            n_elements = struct.unpack('<I', f.read(4))[0]

            if quantized:
                n_scales = struct.unpack('<I', f.read(4))[0]
                scales = np.frombuffer(f.read(n_scales * 4), dtype=np.float32).copy()

                packed_len = struct.unpack('<I', f.read(4))[0]
                packed_data = f.read(packed_len)
                codes = unpack_fp4(packed_data, n_elements)

                entries.append({
                    'name': name,
                    'shape': shape,
                    'n_elements': n_elements,
                    'quantized': True,
                    'scales': scales,
                    'codes': codes,
                })
            else:
                raw = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()
                entries.append({
                    'name': name,
                    'shape': shape,
                    'n_elements': n_elements,
                    'quantized': False,
                    'raw_data': raw,
                })
    return entries


# ─── Evaluate accuracy impact ────────────────────────────────────────────────

def evaluate_quantization_error(model, entries):
    """Measure per-layer quantization error (only for FP4-quantized layers)."""
    print("\n  Per-layer quantization error (FP4 kernels only):")
    total_mse = 0
    total_n = 0
    for idx, w in enumerate(model.weights):
        entry = entries[idx]
        if not entry.get('quantized', True):
            continue  # skip full-precision entries (no error)
        original = w.numpy()
        reconstructed = dequantize_entry(entry)
        diff = original - reconstructed
        rmse = np.sqrt(np.mean(diff ** 2))
        wmax = np.max(np.abs(original))
        rel = rmse / wmax if wmax > 0 else 0
        total_mse += np.sum(diff ** 2)
        total_n += diff.size
        print(f"    {entry['name']:45s}  RMSE={rmse:.6f}  rel={rel:.4f}  shape={list(original.shape)}")

    total_rmse = np.sqrt(total_mse / total_n)
    print(f"  Overall kernel RMSE: {total_rmse:.6f}")


def apply_fp4_weights(model, entries):
    """Replace model weights with FP4-dequantized values."""
    new_weights = []
    for idx, w in enumerate(model.weights):
        new_weights.append(dequantize_entry(entries[idx]))
    model.set_weights(new_weights)


def convert_fp4_model_to_tflite(model, entries, train_dir=None):
    """Apply FP4 weights, then convert to int8 TFLite for smallest size.

    Args:
        train_dir: Path to training images (e.g. COCO train2014/) for calibration.
                   If None, uses synthetic data with realistic image statistics.
    """
    import cv2
    apply_fp4_weights(model, entries)

    N_CAL = 200  # number of calibration samples

    if train_dir and os.path.isdir(train_dir):
        # Use real training images for calibration
        image_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
        rng = np.random.RandomState(0)
        rng.shuffle(image_files)
        cal_files = image_files[:N_CAL]
        print(f"  Using {len(cal_files)} training images from {train_dir} for calibration")

        def rep_data():
            for fname in cal_files:
                img = cv2.imread(os.path.join(train_dir, fname))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize keeping aspect ratio (same as model training)
                h, w = img.shape[:2]
                dif = max(h, w)
                mask = np.zeros((dif, dif, 3), dtype=img.dtype)
                x_pos = int((dif - w) / 2.)
                y_pos = int((dif - h) / 2.)
                mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img
                img = cv2.resize(mask, (96, 96), cv2.INTER_AREA)
                # MobileNet preprocessing: [0,255] → [-1,1]
                img = img.astype(np.float32) / 127.5 - 1.0
                yield [img.reshape(1, 96, 96, 3)]
    else:
        # Fallback: synthetic data with realistic image statistics
        print("  No training dir provided. Using synthetic calibration data.")
        IMG_MEAN = np.array([-0.030, -0.088, -0.188], dtype=np.float32)
        IMG_STD = np.array([0.458, 0.448, 0.450], dtype=np.float32)

        def rep_data():
            rng = np.random.RandomState(0)
            for _ in range(N_CAL):
                img = rng.randn(1, 96, 96, 3).astype(np.float32) * IMG_STD + IMG_MEAN
                img = np.clip(img, -1.0, 1.0)
                yield [img]

    # Convert with full int8 quantization on top of FP4-dequantized weights
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MX-FP4 quantization for VWW model")
    parser.add_argument("--train-dir", type=str, default=None,
                        help="Path to training images (e.g. COCO train2014/) for TFLite calibration")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    orig_tflite_size = os.path.getsize(ORIGINAL_TFLITE)

    print(f"Loading {H5_PATH}...")
    model = tf.keras.models.load_model(H5_PATH, compile=False)
    n_params = model.count_params()
    print(f"Parameters: {n_params:,}")
    print(f"Original .tflite: {orig_tflite_size:,} bytes ({orig_tflite_size/1024:.1f} KB)\n")

    # ── Quantize to MX-FP4 ──
    print("=" * 65)
    print("MX-FP4 (E2M1) Microscaling Quantization (block_size=32)")
    print("=" * 65)
    entries = quantize_weights_fp4(model)

    # ── Save binary ──
    fp4_path = os.path.join(OUT_DIR, "model_mxfp4.fp4bin")
    fp4_size = save_fp4_binary(entries, fp4_path)
    print(f"\n  FP4 binary file:   {fp4_size:>8,} bytes ({fp4_size/1024:.1f} KB)")

    # ── gzip the fp4 binary ──
    with open(fp4_path, 'rb') as f:
        fp4_data = f.read()
    fp4_gz = gzip.compress(fp4_data, compresslevel=9)
    fp4_gz_path = fp4_path + ".gz"
    with open(fp4_gz_path, 'wb') as f:
        f.write(fp4_gz)
    print(f"  FP4 binary + gzip: {len(fp4_gz):>8,} bytes ({len(fp4_gz)/1024:.1f} KB)")

    # ── Quantization error ──
    evaluate_quantization_error(model, entries)

    # ── Roundtrip: verify load ──
    print("\n  Verifying roundtrip (save → load → dequantize)...")
    loaded = load_fp4_binary(fp4_path)
    for orig_e, load_e in zip(entries, loaded):
        orig_w = dequantize_entry(orig_e)
        load_w = dequantize_entry(load_e)
        assert np.allclose(orig_w, load_w), f"Mismatch in {orig_e['name']}"
    print("  Roundtrip OK!")

    # ── Convert FP4-quantized model to TFLite (int8 on top) ──
    print("\n  Converting FP4-quantized model → TFLite (int8)...")
    model2 = tf.keras.models.load_model(H5_PATH, compile=False)
    tflite_bytes = convert_fp4_model_to_tflite(model2, entries, train_dir=args.train_dir)
    tflite_path = os.path.join(OUT_DIR, "model_mxfp4_int8.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_bytes)
    tflite_size = len(tflite_bytes)
    print(f"  FP4→int8 .tflite: {tflite_size:>8,} bytes ({tflite_size/1024:.1f} KB)")

    # gzip that too
    tflite_gz = gzip.compress(tflite_bytes, compresslevel=9)
    tflite_gz_path = tflite_path + ".gz"
    with open(tflite_gz_path, 'wb') as f:
        f.write(tflite_gz)
    print(f"  FP4→int8 + gzip:  {len(tflite_gz):>8,} bytes ({len(tflite_gz)/1024:.1f} KB)")

    # ── Summary ──
    print("\n  NOTE: Run eval_fp4_coco.py on real COCO images for true accuracy.")
    print("        Synthetic/random agreement tests are unreliable.\n")
    print(f"\n{'=' * 65}")
    print("FINAL COMPARISON")
    print(f"{'=' * 65}")
    rows = [
        ("Original .tflite (hybrid quant)", orig_tflite_size),
        ("MX-FP4 binary (.fp4bin)", fp4_size),
        ("MX-FP4 binary + gzip9", len(fp4_gz)),
        ("FP4→int8 .tflite", tflite_size),
        ("FP4→int8 .tflite + gzip9", len(tflite_gz)),
    ]
    for label, size in sorted(rows, key=lambda x: x[1]):
        pct = (1 - size / orig_tflite_size) * 100
        print(f"  {label:<40} {size:>7,} B  ({size/1024:>5.1f} KB)  {pct:>+6.1f}%")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
