import time,json
import os, sys, pathlib, contextlib
import pandas as pd
from typing import TypedDict
from enum import Enum

import wave
import torch
import whisper
from batchalign.pipelines.pipeline import BatchalignPipeline
from batchalign import Document
from batchalign.formats.chat import CHATFile



class TorchBackend(Enum):
    MPS = "mps"  # Apple Silicon (M1/M2/M3/M4/M5 Macbooks)
    CUDA = "cuda"  # Nvidia (dedicated windows GPU)
    CPU = "cpu"  # CPU (fallback, slower)


class ModelDescription(TypedDict):
    name: str
    device: str
    backend: str
    dtype: str
    total_params: int
    ram_usage: str
    ram_usage_bytes: int


class WhisperModelSize(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


def get_device_and_dtype() -> tuple[TorchBackend, torch.dtype]:
    if torch.backends.mps.is_available():
        # Macbook with Apple Silicon
        device = TorchBackend.MPS
        torch_dtype = torch.float32
    elif torch.cuda.is_available():
        # Windows with Nvidia GPU
        device = TorchBackend.CUDA
        torch_dtype = torch.float16
    else:
        # CPU fallback
        device = TorchBackend.CPU
        torch_dtype = torch.float32
    return device, torch_dtype


def convert_model(nlp: BatchalignPipeline, torch_backend: TorchBackend) -> BatchalignPipeline:
    """Convertit le modèle Whisper du pipeline Batchalign pour qu'il utilise le backend spécifié."""\
    

    model = nlp.__dict__["_BatchalignPipeline__generator"].__dict__[
        "_OAIWhisperEngine__whisper"
    ]
    model_desc = get_model_description(
        model
    )

    if model_desc['backend'] == torch_backend.value:
        print(f"Model is already on the correct backend ({torch_backend.value}), no conversion needed.")
        return nlp
    else:
        print(f"Converting model from {model_desc['backend']} to {torch_backend.value}...")
    
    model = model.to(torch_backend.value)
    nlp.__dict__["_BatchalignPipeline__generator"].__dict__[
        "_OAIWhisperEngine__whisper"
    ] = model

    if torch_backend == TorchBackend.MPS:
        import whisper.timing as whisper_timing

        # Set Torch MPS fallback to allow using MPS even if some operations are not supported
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Patch whisper DTW to use CPU because of unsupported operations on MPS
        _original_dtw = whisper_timing.dtw

        def dtw_mps_safe(x):
            if isinstance(x, torch.Tensor) and x.device.type == "mps":
                # print("Using CPU fallback for DTW on MPS device")
                return whisper_timing.dtw_cpu(x.detach().cpu().double().numpy())

            return _original_dtw(x)

        whisper_timing.dtw = dtw_mps_safe

    return nlp




def build_pipeline(lang="fra", num_speakers=2, model_size=WhisperModelSize.TURBO, torch_backend: TorchBackend = TorchBackend.CPU, include_morphosyntax=False) -> tuple[BatchalignPipeline, ModelDescription, torch.dtype]:
    device, torch_dtype = get_device_and_dtype()

    task = 'asr'
    if include_morphosyntax:
        task += ',morphosyntax'

    nlp = BatchalignPipeline.new(task, lang=lang, num_speakers=num_speakers)


    # Inject the model size
    model = whisper.load_model(model_size.value)
    nlp.__dict__["_BatchalignPipeline__generator"].__dict__[
        "_OAIWhisperEngine__whisper"
    ] = model

    # Convert the model to the correct device and dtype
    convert_model(nlp, torch_backend)

    # Get the model description
    model_desc = get_model_description(model)

    return nlp, model_desc, torch_dtype


def get_model_description(model: torch.nn.Module) -> ModelDescription:
    name = model.__class__.__name__
    first_param = next(model.parameters())
    device = first_param.device
    backend = first_param.device.type
    dtype = first_param.dtype
    total_params = sum(p.numel() for p in model.parameters())

    # Estimated RAM usage : total params x dtype
    ram_usage_bytes = total_params * torch.tensor([], dtype=dtype).element_size()

    def display_ram_usage(bytes: int) -> str:
        if bytes < 1024:
            return f"{bytes} B"
        elif bytes < 1024**2:
            return f"{bytes / 1024:.2f} KB"
        elif bytes < 1024**3:
            return f"{bytes / 1024**2:.2f} MB"
        else:
            return f"{bytes / 1024**3:.2f} GB"

    return {
        "name": name,
        "device": str(device),
        "backend": backend,
        "dtype": str(dtype),
        "total_params": total_params,
        "ram_usage": display_ram_usage(ram_usage_bytes),
        "ram_usage_bytes": ram_usage_bytes,
    }


def describe_model(nlp: BatchalignPipeline):
    model = nlp.__dict__["_BatchalignPipeline__generator"].__dict__[
        "_OAIWhisperEngine__whisper"
    ]
    model_desc = get_model_description(model)

    # Print it nicely
    print(f"### Model ###  " + "-" * 50)
    print(f'\t> Name  :\t{model_desc["name"]}')
    print(f"\t> Device:\t{model_desc['device']} ")
    print(f"\t> dtype :\t{model_desc['dtype']} ")
    print(f"\t> RAM   :\t{model_desc['ram_usage']}")


def get_wav_duration_seconds(wav_path: str) -> float:
    """Retourne la durée d'un fichier WAV en secondes."""
    with contextlib.closing(wave.open(wav_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        if rate > 0:
            return frames / float(rate)
    return -1


class TranscriptionResult(TypedDict):
    audio_file: str
    torch_backend: str
    torch_dtype: str
    audio_duration_s: float
    pipeline_creation_time_s: float
    transcription_time_s: float


def transcribe_audio(
    audio_file: str, outfile: str, torch_backend: TorchBackend = TorchBackend.CPU, model_size: WhisperModelSize = WhisperModelSize.TURBO, include_morphosyntax=False
) -> TranscriptionResult:

    # Build the pipeline
    t0 = time.time()
    nlp, device, torch_dtype = build_pipeline(torch_backend=torch_backend, model_size=model_size, include_morphosyntax=include_morphosyntax)
    t_pipeline = time.time() - t0
    desc = get_model_description(nlp.__dict__["_BatchalignPipeline__generator"].__dict__["_OAIWhisperEngine__whisper"])


    # Extract audio duration
    audio_duration = get_wav_duration_seconds(audio_file)

    # Create a Batchalign Document from the audio file
    doc = Document.new(media_path=audio_file, lang="fra")

    t0 = time.time()
    doc = nlp(doc)
    t_transcription = time.time() - t0

    chat = CHATFile(doc=doc)
    chat.write(outfile)

    print(f">>> Transcription of {audio_file} <<<")
    print(f"\t> Whisper backend:  \t{desc['backend']} ({desc['dtype']})")
    print(f"\t> Audio duration:   \t{audio_duration:.2f} s")
    print(f"\t> Pipeline creation:\t{t_pipeline:.2f} s")
    print(f"\t> Transcription time:\t{t_transcription:.2f} s")
    return TranscriptionResult(
        audio_file=audio_file,
        torch_backend=desc['backend'],
        torch_dtype=desc['dtype'],
        audio_duration_s=audio_duration,
        pipeline_creation_time_s=t_pipeline,
        transcription_time_s=t_transcription,
    )

