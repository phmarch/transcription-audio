import sys
import time
import csv
import wave
import contextlib
from pathlib import Path

import torch
import batchalign as ba


# =========================
# PARAMÈTRES
# =========================
AUDIO_FOLDER = Path.home() / "Documents/AnneSo/Audio"
OUTPUT_FOLDER = Path.home() / "Documents/AnneSo/ba_whisperlarge_output"
CSV_PATH = OUTPUT_FOLDER / "batchalign_temps_transcription.csv"
LANG = "fra"
NUM_SPEAKERS = 2  # mets 2 si tu veux forcer 2 locuteurs


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
    return device, torch_dtype


def get_wav_duration_seconds(wav_path: Path):
    try:
        with contextlib.closing(wave.open(str(wav_path), "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return None


def build_pipeline(lang="fra", num_speakers=1):
    device, torch_dtype = get_device_and_dtype()

    print("MPS disponible :", torch.backends.mps.is_available())
    print("CUDA disponible :", torch.cuda.is_available())
    print("Device détecté :", device)
    print("Type détecté :", torch_dtype)
    print("Création du pipeline Batchalign...")

    # API compatible avec la doc actuelle
    nlp = ba.BatchalignPipeline.new("asr", lang=lang, num_speakers=num_speakers)

    return nlp, device, torch_dtype


def process_one_file(audio_file, output_folder, nlp, lang="fra"):
    print(f"\nTraitement de : {audio_file.name}")

    audio_duration_s = get_wav_duration_seconds(audio_file)
    file_start = time.time()

    doc_create_start = time.time()
    doc = ba.Document.new(media_path=str(audio_file), lang=lang)
    doc_create_end = time.time()
    doc_creation_time_s = doc_create_end - doc_create_start
    print(f"  Document créé en {doc_creation_time_s:.2f} s")

    trans_start = time.time()
    print("  Début transcription...")
    doc = nlp(doc)
    trans_end = time.time()
    transcription_time_s = trans_end - trans_start
    print(f"  Transcription terminée en {transcription_time_s:.2f} s")

    export_start = time.time()
    chat_file_path = output_folder / f"{audio_file.stem}.cha"
    chat = ba.CHATFile(doc=doc)
    chat.write(str(chat_file_path))
    export_end = time.time()
    export_time_s = export_end - export_start

    total_time_s = time.time() - file_start

    realtime_factor = ""
    if audio_duration_s and audio_duration_s > 0:
        realtime_factor = round(transcription_time_s / audio_duration_s, 2)

    print(f"  Export .cha : {export_time_s:.2f} s")
    print(f"  Temps total fichier : {total_time_s:.2f} s")
    if audio_duration_s is not None:
        print(f"  Durée audio : {audio_duration_s:.2f} s")
    if realtime_factor != "":
        print(f"  Facteur temps réel (RTF) : {realtime_factor}x")

    return {
        "file_name": audio_file.name,
        "audio_duration_s": (
            round(audio_duration_s, 2) if audio_duration_s is not None else ""
        ),
        "doc_creation_time_s": round(doc_creation_time_s, 2),
        "transcription_time_s": round(transcription_time_s, 2),
        "export_time_s": round(export_time_s, 2),
        "total_time_s": round(total_time_s, 2),
        "realtime_factor": realtime_factor,
        "output_path": str(chat_file_path),
        "transcription": str(doc),
    }


def write_results_to_csv(results, csv_path):
    fieldnames = [
        "file_name",
        "audio_duration_s",
        "doc_creation_time_s",
        "transcription_time_s",
        "export_time_s",
        "total_time_s",
        "realtime_factor",
        "output_path",
        "transcription",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    if not AUDIO_FOLDER.exists():
        print(f"Erreur : dossier audio introuvable : {AUDIO_FOLDER}")
        sys.exit(1)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(AUDIO_FOLDER.glob("*.wav"))

    if not audio_files:
        print(f"Aucun fichier .wav trouvé dans : {AUDIO_FOLDER}")
        sys.exit(1)

    print(f"{len(audio_files)} fichier(s) trouvé(s).")

    overall_start = time.time()

    try:
        nlp, device, torch_dtype = build_pipeline(lang=LANG, num_speakers=NUM_SPEAKERS)
    except Exception as e:
        print(f"Erreur lors de la création du pipeline : {e}")
        sys.exit(1)

    results = []
    success_count = 0
    fail_count = 0

    for audio_file in audio_files:
        try:
            result = process_one_file(
                audio_file=audio_file,
                output_folder=OUTPUT_FOLDER,
                nlp=nlp,
                lang=LANG,
            )
            results.append(result)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"  Échec pour {audio_file.name} : {e}")

    overall_duration = time.time() - overall_start

    write_results_to_csv(results, CSV_PATH)

    print("\n=== RÉSUMÉ FINAL ===")
    print(f"Succès : {success_count}")
    print(f"Échecs : {fail_count}")
    print(f"Temps total global : {overall_duration:.2f} s")
    print(f"CSV écrit : {CSV_PATH}")
    print(f"Dossier de sortie : {OUTPUT_FOLDER}")
    print(f"Device détecté : {device}")
    print(f"Type détecté : {torch_dtype}")

    if results:
        print("\nTemps par fichier :")
        for r in results:
            print(
                f"- {r['file_name']} | "
                f"audio = {r['audio_duration_s']} s | "
                f"transcription = {r['transcription_time_s']} s | "
                f"total = {r['total_time_s']} s | "
                f"RTF = {r['realtime_factor']}"
            )


if __name__ == "__main__":
    main()
