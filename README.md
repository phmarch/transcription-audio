# transcription-audio

## Getting Started

Copy the audio files as `*.wav` in the `data/input/` directory

Install the dependencies with `uv sync`


#### BatchAlign

To use batchalign, comment out the crisper-whisper transformers dependency, then run `uv sync`

```toml
    # "transformers @ git+https://github.com/nyrahealth/transformers.git@crisper_whisper",
    "batchalign>=0.8.2.post13",
```

Then open the `batch-align.ipynb` notebook and run the cells to transcribe the audio files in `data/input/` and generate `.cha` files in `data/output/`

#### Crisper-Whisper

To use crisper-whisper, comment out the batchalign dependency, then run `uv sync`

```toml
    "transformers @ git+https://github.com/nyrahealth/transformers.git@crisper_whisper",
    # "batchalign>=0.8.2.post13",
``` 

Then open the `crisper-whisper.ipynb` notebook and run the cells to transcribe the audio files in `data/input/` and generate a JSON file in `data/output/` with the transcription and word-level timestamps.