from openai import AzureOpenAI

import json
import os
import httpx
import pandas as pd

import time
import copy
import traceback

def correct_chunk(client, temperature, system_prompt, chunk, max_tries=3):
    print("Starting first attempt.")
    for attempt in range(max_tries):
        try:
            # Use the gpt-5-mini model to correct the transcript
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": json.dumps(chunk, ensure_ascii=False)
                    }
                ]
            )
            return json.loads(response.choices[0].message.content.encode('utf-8').decode('utf-8'))
        except json.JSONDecodeError as e:
            raw = response.choices[0].message.content
            print(f"JSONDecodeError at char {e.pos}: ...{raw[max(0,e.pos-50):e.pos+50]}...")
            
            # Do try to run as much as possible through the LLM
            corrected_chunk = {}
            for key, value in chunk.items():
                try:
                    test = json.dumps(value, ensure_ascii=False)
                    corrected_chunk[key] = value
                except Exception:
                    print(f"Skipping turn {key}.")
            return corrected_chunk
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_tries - 1:
                time.sleep(5)
            else:
                print(f"All retries failed for this chunk, skipping.")
                return chunk

def correct_transcript(client, temperature, system_prompt, transcription_json, chunk_size=10):
    """
        Corrects a transcript of an interview by assigning the correct speaker labels to each turn.

        :param client: An instance of the AzureOpenAI client.
        :param temperature: The temperature to use in the model.
        :param system_prompt: The model prompt to use for the correction.
        :param transcription_json: The diarised JSON object containing the transcription to be corrected.
        :param chunk_size: Integer with chunking size. Default=10.
        :return: A corrected JSON object containing the cleaned up transcription of the interview.
    """
    conv = transcription_json["conv"]
    keys = list(conv.keys())
    corrected_conv = {}

    # Copy transcript
    copy_transcription_json = copy.deepcopy(transcription_json)

    for i in range(0, len(keys), chunk_size):
        # Take only a chunk of the conversation
        chunk = {k: conv[k] for k in keys[i:i+chunk_size]}
        corrected_chunk = correct_chunk(client, temperature, system_prompt, chunk)
        print("Corrected the chunk, continuing.")
        # Gather the whole conversation
        corrected_conv.update(corrected_chunk)
        print(f"Completed chunk {i/chunk_size} of {len(keys)/chunk_size}")

    copy_transcription_json["conv"] = corrected_conv

    # Return the corrected transcript as a JSON object
    return copy_transcription_json

def make_prompt(patient, HP):
    """
        Creates a model prompt for correcting the speaker labels in a diarised interview transcript.
        :param patient: The name of the patient in the interview.
        :param HP: The name of the healthcare professional in the interview.
        :return: A string containing the model prompt for correcting the speaker labels in a diarized interview transcript.
    """
    return f"""You are a helpful assistant correcting the speaker labels in a diarized interview transcript in Dutch.
        The transcript is a chunk of a JSON object where each turn has a speaker label, such as SPEAKER_00 or SPEAKER_01.
        First, determine which speaker label corresponds to which person by looking at the conversation as a whole.
        {HP} is the person who asks the questions most of the time, and {patient} is the person who answers most of the time.
        A proxy may be present and should be labelled as 'Proxy_{patient}'.
        Any other speakers should be labelled as 'Third Party'.
        Once you have determined which speaker label corresponds to which person, make a new field to each turn entitled "speaker_llm" with the speaker label that you think it is.
        If you can't identify the speaker_llm, but only if you cannot identify them, assign it the value None.
        Do not change anything else. Keep the structure and text exactly as they are.
        Additionally, add three extra fields to each turn: 
            - 'confidence_score_speaker': indicates how certain you are about the assigned speaker label. This ranges from 0 to 1. A low score indicates that there might be something wrong with the speaker label, or that you are not confident that it is the correct label. A high score indicates that the speaker label is likely correct. 
            - 'confidence_score_text': indicates how certain you are that the text is a correct grammatical, coherent sentence which follows logically based on the previous 10 utterances. This ranges from 0 to 1. A low score indicates that the sentence is not coherent, doesn't follow logically from the previous turns, or contains non-existent words. A high score indicates a coherent sentence in Dutch.
            - 'notes': you can write any notes about the transcription, such as the presence of noise or the presence of a proxy or third party. You can also comment on why you assigned certain confidence scores here.
        Always fill in the 'confidence_score_speaker' and 'confidence_score_text' field, but only fill in the 'notes' field if there is something worth noting.
        If a word or phrase is continuously repeated in one turn or across multiple turns, it could mean that the Whisper model got stuck. This would result in a low confidence score for the text.
        Only return the corrected JSON object and nothing else.
        """

def save_json(transcript_dict, output_path):
    """
        Saves a JSON object to a file.

        :param transcript_dict: The dictionary to be saved.
        :param output_path: The path to the file where the JSON object should be saved.
        :return: None
    """
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(transcript_dict, json_file, indent=2, ensure_ascii=False)

def main():
    begin_time = time.time()
    
    # Set up the AzureOpenAI client
    domain="openai4reha.openai.azure.com"
    with open(r"..\ip.txt", "r") as file:
        ip = file.read().strip()
    with open(r"..\api_key.txt", "r") as file:
        api_key = file.read().strip()

    os.environ["NO_PROXY"] = domain+","+ip

    # Create an HTTP client 
    http=httpx.Client(verify=False, headers={"Host": domain})
    domain_url ="https://"+domain
    ip_url = "https://"+ip
    client = AzureOpenAI(api_key=api_key,
                         api_version="2024-02-01",
                         azure_endpoint=ip_url, 
                         http_client=http)
    
    # Load the mapping of audio files to patients and healthcare professionals
    files_path = "mapping_audio_files.json"
    with open(files_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    # Clean up the transcriptions of each interview
    for i, (patient_id, patient) in enumerate(data.items()):
        
        HP_name = patient["HP"]

        model_prompt = make_prompt(patient_id, HP_name)

        for t_label, files in patient["files"].items(): 
            for file in files:
                filename = os.path.splitext(os.path.basename(file))[0]
                filelocation = f"../transcriptions/{filename}.json"
                # Skip if the file is not yet in transcriptions
                if not os.path.exists(filelocation):
                    continue

                # Skip if the file has already been processed
                output_file = f"../clean_transcriptions/{filename}.json"
                if os.path.exists(output_file):
                    continue
                
                with open(filelocation, 'r', encoding="utf-8") as f:
                    transcription_json = json.load(f)

                # Correct the transcript 
                print(f"We are at the {i}th patient.")
                print(f"Starting on file: {filename}")
                try:
                    corrected_text = correct_transcript(
                        client, 0, model_prompt, transcription_json, chunk_size=50
                    )
                    # Save the corrected transcript to a new JSON file
                    save_json(corrected_text, output_file)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    traceback.print_exc()
                    
                mid_time = time.time()
                print(f"This file took {mid_time - begin_time} seconds to process.")
            
    end_time = time.time()
    print(f"This took {end_time - begin_time} seconds to process in total.")

if __name__ == "__main__":
    main()