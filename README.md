# Transcription Audio Research

A comparative study tool for automatic speech recognition (ASR) and speaker diarization using two different approaches: **BatchAlign** (multi-speaker focused) and **CrisperWhisper** (timestamp-detailed). This is designed for transcription research workflows, particularly for linguistic analysis of multi-speaker audio.

## Quick Start

1. **Place audio files** in `data/input/` (only `.wav` files are processed)
2. **Install dependencies**: `uv sync`
3. **Choose your approach** (see below)
4. **Run the corresponding notebook**

---

## Two Transcription Approaches

### 1. BatchAlign — Multi-Speaker Linguistic Analysis

**Best for:** Linguistic research, CHAT format output, multi-speaker conversations, speaker diarization

- **Output format:** `.cha` files (CHAT format, standard in linguistic research)
- **Speaker detection:** Automatic multi-speaker diarization with configurable speaker count
- **Additional processing:** Includes disfluency detection and retracing analysis
- **Performance:** Multi-speaker ASR optimized
- **Hardware:** CPU, GPU (CUDA), or Apple Silicon (MPS) support

**Setup:**
1. In `pyproject.toml`, ensure batchalign is enabled and crisper-whisper is commented out:
```toml
    # "transformers @ git+https://github.com/nyrahealth/transformers.git@crisper_whisper",
    "batchalign>=0.8.2.post13",
```

2. Install: `uv sync`
3. Run: Open `batch-align.ipynb` and execute all cells

**Output:**
- Generated `.cha` files in `data/batchalign/output/`
- Performance metrics (duration, inference time) in CSV format

---

### 2. CrisperWhisper — Word-Level Timestamps & Detailed Analysis

**Best for:** Detailed temporal analysis, word-level timing, timing research, JSON-based workflows

- **Output format:** JSON with word-level timestamps and transcription
- **Timestamp precision:** Per-word timing with automatic pause adjustment
- **Model:** `nyrahealth/CrisperWhisper` (Whisper-based, optimized for clarity)
- **Pause handling:** Intelligent pause duration distribution between words
- **Performance:** Single-pass transcription with detailed timing data
- **Hardware:** CPU or GPU (CUDA) support

**Setup:**
1. In `pyproject.toml`, ensure crisper-whisper is enabled and batchalign is commented out:
```toml
    "transformers @ git+https://github.com/nyrahealth/transformers.git@crisper_whisper",
    # "batchalign>=0.8.2.post13",
```

2. Install: `uv sync`
3. Run: Open `crisper-whisper.ipynb` and execute all cells

**Output:**
- Generated JSON files with transcriptions and word-level timestamps in `data/crisper-whisper/output/`
- Metadata with performance metrics in `data/crisper-whisper/metadata/`

---

## Hardware Acceleration

Both notebooks automatically detect available hardware:

- **Apple Silicon (M1/M2/M3/M4/M5):** Uses MPS backend with float32 (BatchAlign only)
- **NVIDIA GPU:** Uses CUDA backend with float16 for better performance
- **CPU fallback:** Slowest but always available option

Detection is automatic—no configuration needed.

---

## File Structure

```
data/
├── input/              # Place your .wav files here
├── batchalign/
│   ├── output/        # Generated .cha files
│   └── metadata/      # Performance metrics and timing data
├── crisper-whisper/
│   ├── output/        # Generated JSON files with timestamps
│   └── metadata/      # Performance metrics
├── sample/            # Example outputs
└── sample-batchalign-output.cha  # Reference output
```

---

## Research Use Cases

- **Linguistic analysis:** Use BatchAlign for CHAT format compatibility with linguistic tools
- **Temporal analysis:** Use CrisperWhisper for word-level timing data
- **Model comparison:** Run both approaches to compare outputs and performance metrics
- **Speech clarity research:** CrisperWhisper specializes in articulation clarity
- **Multi-speaker conversations:** BatchAlign excels at speaker diarization

---

## Performance Metrics

Both notebooks track:
- Audio duration (seconds)
- Pipeline initialization time
- Transcription/inference time
- Hardware backend used
- Data type precision used

Metrics are logged with each transcription for comparative analysis.