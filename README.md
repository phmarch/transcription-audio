# Transcription Audio

Experimental audio transcription workflows for speech-language research.

The current main workflow uses **Batchalign2** to generate CLAN/CHAT-compatible `.cha` transcription files.


## Quick Start

Have uv installed and recognized in your terminal: [Download UV here](https://docs.astral.sh/uv/getting-started/installation/)

Install the Batchalign2 environment:

```bash
uv sync --group batchalign2
```

For NVIDIA GPU support, include the CUDA group:

```bash
uv sync --group batchalign2 --group cuda
```


## Minimal Example

```python
from src.batchalign import transcribe_audio, TorchBackend, WhisperModelSize

result = transcribe_audio(
    audio_file="data/input/sample/your_file.wav",
    outfile="data/batchalign/output/your_file.cha",
    torch_backend=TorchBackend.CPU,
    model_size=WhisperModelSize.TURBO,
    include_morphosyntax=False,
)

print(result)
```

```python
{
    "audio_file": "data/input/sample/your_file.wav",
    "torch_backend": "cpu",
    "torch_dtype": "torch.float32",
    "audio_duration_s": 10.24,
    "pipeline_creation_time_s": 4.10,
    "transcription_time_s": 7.85,
}
```


### Parameters

| Parameter | Description |
|---|---|
| `audio_file` | Path to the input audio file, usually `.wav`. |
| `outfile` | Path where the output `.cha` file should be written. |
| `torch_backend` | Compute backend: `CPU` (processor), `CUDA` (Nvidia GPU), or `.MPS` (Apple). |
| `model_size` | Whisper model size: `TINY`, `BASE`, `SMALL`, `MEDIUM`, `LARGE`, or `TURBO`. |
| `include_morphosyntax` | Whether to run morphosyntax processing after transcription. Disabled by default. |

The function returns a dictionary containing the following keys:
- `audio_file`: The input audio file path.
- `torch_backend`: The compute backend used.
- `torch_dtype`: The data type used for PyTorch tensors.
- `audio_duration_s`: The duration of the input audio in seconds.
- `pipeline_creation_time_s`: Time taken to set up the transcription pipeline.
- `transcription_time_s`: Time taken to perform the transcription.



### Future Development

- Supporting more automatic speech recognition (ASR) models beyond Whisper
- Timing the ASR inference separately from the morphosyntax processing
- Implementing CLI tools for batch processing and integration into larger workflows
- Improving cha file formatting by adding a python / LLM layer to improve codification