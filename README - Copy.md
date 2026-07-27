## Thesis Project — Marit Rozendaal

This repository contains the code for the thesis:

> *Modelling Patient Functioning Over Time: Automated ICF 
> Categorisation and Its Representation in an Event-Centred 
> Knowledge Graph*

The pipeline transcribes and diarises Dutch medical conversations 
between healthcare professionals and geriatric patients following 
interRAI questionnaires. Conversations are then cleaned and 
synthetically annotated with ICF labels using an LLM. Models are 
trained to predict ICF categories, levels, and relative time. Results 
are evaluated, a knowledge graph is extracted from the annotated 
conversations, and an error analysis is included.


## Project structure

- `thesis_report` — thesis report.
- `requirements.txt` — Python dependencies used in the project.
- `LICENSE` — project license.
- `README.md` — project documentation.
- `src/` — implementation code and notebooks.
  - `src/transcription/` — audio transcription and diarization pipeline.
  - `src/annotation/` — LLM-based annotation and speaker correction.
  - `src/model_training/` — python notebooks for training models.
  - `src/evaluation/` — evaluation scripts for model and LLM outputs.
  - `src/KG/` — knowledge graph extraction.
  - `src/EA/` — error analysis notebooks and scripts.

## Main Components

### 1. Transcription — `src/transcription/transcribe_diarize.py`
- Transcribes Dutch interview audio using OpenAI Whisper
- Performs speaker diarisation with Pyannote
- Anonymises named entities with spaCy
- Saves output as intermediate JSON files

### 2. Speaker Correction — `src/annotation/`
- Requires valid Azure credentials
- `llm_speaker_annotation.py` — Refines speaker labels using Azure OpenAI
- `llm_annotation.py` — Annotates conversations with ICF categories, levels, and relative time using LLM prompts

### 3. Model training — `src/model_training/`
- Trains models (mBERT and MedRoBERTa) for predicting ICF categories, levels, and relative time
- Trains on different datasets

### 4. Evaluation — `src/evaluation/`
- `evaluation_models.py` — evaluates model output quality
- `evaluation_llm.py` — evaluates LLM output quality

### 5. Knowledge Graph — `src/KG/KG.py`
- Extracts mention triples and knowledge graph data from annotated conversations

### 6. Error Analysis — `src/EA/`
- Notebooks and scripts for error analysis across categories, levels, and time annotations

## Data 
The medical data is not included in this repository and must be requested separately before running the pipeline.

## Notes

- GPU support is recommended for this thesis.
- The speaker correction pipeline uses Azure OpenAI and requires valid Azure credentials.
- The transcription pipeline and model training both use `torch`, but require different versions. The correct versions are specified in `requirements.txt`.

## Licence

This project is released under an open MIT-style licence. See `LICENSE` for details.
