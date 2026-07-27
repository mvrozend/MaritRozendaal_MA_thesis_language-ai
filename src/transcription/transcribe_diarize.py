import time 

from pydub import AudioSegment
import math

import whisper
import torch
from pyannote.audio import Pipeline
import utils as utils

import spacy

import json
import os

def split_audio(audio_path, chunk_minutes=2):
    audio = AudioSegment.from_file(audio_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i+chunk_ms]
        chunk_path = f"{audio_path[:-4]}_chunk{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_diarize_audio(transcript_model, diarize_pipeline, audio_path, filename, nlp, patient, HP, T):
    """
        Transcribes and diarizes an audiofile.

        :param transcript_model: The transcription model (Whisper)
        :param diarize_pipeline: The diarization pipeline (Pyannote)
        :param audio_path: The path to the audiofile
        :param filename: String of the file name
        :param nlp: Spacy's nl_core_news_lg 
        :param patient: String of the patient id
        :param HP: String of the healthcare professional's id
        :param T: String of the term of the interview
    """
    print(f"\nStarting transcribing the text of {filename}. Please wait.")
    chunks = split_audio(audio_path)
    transcription = {"text": [], "segments": [], "language": []}
    for i, chunk in enumerate(chunks):
        turns = transcript_model.transcribe(chunk, language="nl", temperature=0.0)

        for utterance in turns["segments"]:
            utterance["text"] = anonymize_text(utterance["text"], nlp)
        
        transcription["text"].extend(turns["text"])
        transcription["segments"].extend(turns["segments"])
        transcription["language"].extend(turns["language"])

    print("Starting diarizing the text. Please wait.")
    diarization = diarize_pipeline(audio_path)  

    # Combine transcription and diarization
    final_result = utils.diarize_text(transcription, diarization)

    # Save results in a dictionary
    transcript_dict = {"filename": filename,
                       "patient": patient,
                       "HP": HP,
                       "T": T,
                       "conv": {}}
    for i, (seg, spk, sent) in enumerate(final_result):
        transcript_dict["conv"][f"turn_{i}"] = {
                                                "speaker": spk,
                                                "text": sent,
                                                "relative_time": None,
                                                "ICF_category": None,
                                                "ICF_level": None
                                                }
    
    # Save dictionary in a JSON file
    save_json(transcript_dict, f"../transcriptions/{filename}.json")

def anonymize_text(txt, nlp):
    """
        Adapted from: https://github.com/cltl/aproof-icf17-classifier/tree/main

        Replace entities of type PERSON and GPE with 'PERSON', 'GPE'.
        Return anonymized text.
    """
    doc = nlp(txt)
    anonym = str(doc)
    to_repl = {str(ent):ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'GPE']}
    for string, replacement in to_repl.items():
        anonym = anonym.replace(string, replacement)
    return anonym

def save_json(transcript_dict, output_path):
    """
        Saves a JSON object to a file.

        :param transcript_json: The JSON object to be saved.
        :param output_path: The path to the file where the JSON object should be saved.
        :return: None
    """
    with open(output_path, "w") as json_file:
        json.dump(transcript_dict, json_file, indent=2, ensure_ascii=False)

def main():
    begin_time = time.time()

    # Check whether GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Whisper model for transcription
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("large", device=device)
    
    # Load the Pyannote pipeline for speaker diarization
    try:
        with open("../hf_key.txt", "r") as file:
            hf_key = file.read().strip()
    except:
        with open("hf_key.txt", "r") as file:
            hf_key = file.read().strip()
        print("hf key exists in this folder")

    print("Loading Pyannote pipeline...")
    pyannote_pipeline = Pipeline.from_pretrained(
                                        "pyannote/speaker-diarization-3.1",
                                        use_auth_token=hf_key)
    pyannote_pipeline.to(device)

    # NLP 
    nlp = spacy.load("nl_core_news_lg")

    # Path to the audio files to be transcribed and diarized
    files_path = "../recordings/mapping_audio_files.json"
    with open(files_path, 'r') as file:
        data = json.load(file)

    # Transcribe and diarize each file
    for i, (patient_id, patient) in data.items():
        HP_name = patient["HP"]
        for t_label, files in patient["files"].items(): 
            for file in files:
                filename = os.path.splitext(os.path.basename(file))[0]
                filelocation = f"../recordings/{filename}.wav"
                
                # Skip if the file has already been processed
                output_file = f"../transcriptions/{filename}.json"
                if os.path.exists(output_file):
                    continue
                
                transcribe_diarize_audio(
                    whisper_model,
                    pyannote_pipeline,
                    filelocation,
                    filename,
                    nlp,
                    patient_id,
                    HP_name,
                    t_label
                )


    
    end_time = time.time()

    print(f"This took {end_time - begin_time} seconds.")
   
if __name__ == "__main__":
    main()