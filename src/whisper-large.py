import csv
import sys
import time
import wave
import contextlib
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# =========================
# PARAMÈTRES
# =========================
INPUT_FOLDER = Path.home() / "Documents/AnneSo/Audio"
OUTPUT_CSV = Path.home() / "Documents/AnneSo/transcriptions_whisper_large.csv"
MODEL_ID = "openai/whisper-large-v3"
LANGUAGE = "french"
CHUNK_LENGTH_S = 20


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # plus stable sur Mac
    elif torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
    return device, torch_dtype


def get_audio_duration_seconds(audio_path: Path):
    """
    Lit la durée seulement pour les WAV.
    Pour les autres formats, retourne None.
    """
    try:
        with contextlib.closing(wave.open(str(audio_path), "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return None


def build_pipeline(model_id=MODEL_ID, chunk_length_s=CHUNK_LENGTH_S):
    device, torch_dtype = get_device_and_dtype()

    print("MPS disponible :", torch.backends.mps.is_available())
    print("CUDA disponible :", torch.cuda.is_available())
    print("Device utilisé :", device)
    print("Type :", torch_dtype)
    print("Modèle :", model_id)

    t0 = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    print(f"Chargement modèle : {time.time() - t0:.2f} s")

    t1 = time.time()
    model.to(device)
    print(f"Transfert vers device : {time.time() - t1:.2f} s")

    t2 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Chargement processor : {time.time() - t2:.2f} s")

    t3 = time.time()
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length_s,
        torch_dtype=torch_dtype,
        device=device,
    )
    print(f"Création pipeline : {time.time() - t3:.2f} s")

    return asr_pipeline, device, torch_dtype


def transcribe_audio(asr_pipeline, file_path, language=LANGUAGE):
    t0 = time.time()
    result = asr_pipeline(
        str(file_path),
        generate_kwargs={
            "language": language,
            "task": "transcribe",
        },
    )
    transcription_time_s = time.time() - t0
    return result, transcription_time_s


def write_results_to_csv(results, csv_path):
    fieldnames = [
        "file_name",
        "file_path",
        "audio_duration_s",
        "transcription_time_s",
        "total_time_s",
        "realtime_factor",
        "model",
        "language",
        "transcription",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def process_folder(input_folder, output_csv, model_id, language, chunk_length_s):
    if not input_folder.exists():
        print(f"Erreur : le dossier audio n'existe pas : {input_folder}")
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".mp4", ".mpeg", ".ogg"}
    audio_files = sorted(
        [
            p
            for p in input_folder.iterdir()
            if p.is_file() and p.suffix.lower() in audio_extensions
        ]
    )

    if not audio_files:
        print(f"Aucun fichier audio trouvé dans : {input_folder}")
        sys.exit(1)

    print(f"{len(audio_files)} fichier(s) audio trouvé(s).")

    asr_pipeline, device, torch_dtype = build_pipeline(
        model_id=model_id,
        chunk_length_s=chunk_length_s,
    )

    results = []
    overall_start = time.time()

    for audio_file in audio_files:
        print(f"\nTraitement : {audio_file.name}")
        file_start = time.time()

        audio_duration_s = get_audio_duration_seconds(audio_file)

        try:
            result, transcription_time_s = transcribe_audio(
                asr_pipeline,
                audio_file,
                language=language,
            )

            text = result.get("text", "").strip()  # type: ignore
            total_time_s = time.time() - file_start

            realtime_factor = ""
            if audio_duration_s and audio_duration_s > 0:
                realtime_factor = round(transcription_time_s / audio_duration_s, 2)

            print(f"  Transcription : {transcription_time_s:.2f} s")
            print(f"  Temps total   : {total_time_s:.2f} s")
            if audio_duration_s is not None:
                print(f"  Durée audio   : {audio_duration_s:.2f} s")
            if realtime_factor != "":
                print(f"  RTF           : {realtime_factor}x")

            results.append(
                {
                    "file_name": audio_file.name,
                    "file_path": str(audio_file),
                    "audio_duration_s": (
                        round(audio_duration_s, 2)
                        if audio_duration_s is not None
                        else ""
                    ),
                    "transcription_time_s": round(transcription_time_s, 2),
                    "total_time_s": round(total_time_s, 2),
                    "realtime_factor": realtime_factor,
                    "model": model_id,
                    "language": language,
                    "transcription": text,
                }
            )

        except Exception as e:
            print(f"  Échec pour {audio_file.name} : {e}")
            results.append(
                {
                    "file_name": audio_file.name,
                    "file_path": str(audio_file),
                    "audio_duration_s": (
                        round(audio_duration_s, 2)
                        if audio_duration_s is not None
                        else ""
                    ),
                    "transcription_time_s": "",
                    "total_time_s": "",
                    "realtime_factor": "",
                    "model": model_id,
                    "language": language,
                    "transcription": f"ERREUR: {e}",
                }
            )

    overall_duration = time.time() - overall_start

    write_results_to_csv(results, output_csv)

    print("\n=== RÉSUMÉ FINAL ===")
    print(f"Temps total global : {overall_duration:.2f} s")
    print(f"CSV écrit : {output_csv}")
    print(f"Dossier audio : {input_folder}")
    print(f"Device détecté : {device}")
    print(f"Type détecté : {torch_dtype}")


def main():
    process_folder(
        input_folder=INPUT_FOLDER,
        output_csv=OUTPUT_CSV,
        model_id=MODEL_ID,
        language=LANGUAGE,
        chunk_length_s=CHUNK_LENGTH_S,
    )


if __name__ == "__main__":
    main()
