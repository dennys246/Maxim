#!/usr/bin/env python
"""Minimal test to isolate Whisper segfault issue."""

import os
import sys

# Hide GPU from CTranslate2
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("=" * 60)
print("WHISPER SEGFAULT DIAGNOSTIC TEST")
print("=" * 60)

# Step 1: Check imports
print("\n[1/5] Testing imports...")
try:
    import ctranslate2

    print(f"  ✓ ctranslate2: {ctranslate2.__version__}")
except ImportError as e:
    print(f"  ✗ ctranslate2: {e}")
    sys.exit(1)

try:
    import faster_whisper

    print(f"  ✓ faster-whisper: {faster_whisper.__version__}")
except ImportError as e:
    print(f"  ✗ faster-whisper: {e}")
    sys.exit(1)

# Step 2: Check cuDNN
print("\n[2/5] Checking CUDA libraries...")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

# Step 3: Try to create model
print("\n[3/5] Creating WhisperModel (float32, cpu)...")
print("  This is where the segfault likely occurs...")

try:
    from faster_whisper import WhisperModel

    model = WhisperModel("tiny", device="cpu", compute_type="float32", download_root="/tmp/whisper_test")
    print("  ✓ Model created successfully!")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 4: Try a dummy transcription
print("\n[4/5] Testing transcription...")
try:
    import numpy as np

    # Create 1 second of silence
    dummy_audio = np.zeros(16000, dtype=np.float32)
    segments, info = model.transcribe(dummy_audio, beam_size=1)
    segments_list = list(segments)
    print(f"  ✓ Transcription completed ({len(segments_list)} segments)")
except Exception as e:
    print(f"  ✗ Transcription failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 5: Success
print("\n[5/5] ✓✓✓ ALL TESTS PASSED ✓✓✓")
print("\nIf this script works but Maxim still crashes, the issue is")
print("in the multiprocessing/subprocess context, not the model itself.")
print("=" * 60)
