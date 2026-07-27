import os
import csv
import json
import pathlib
import httpx
import re
import time
from openai import AzureOpenAI
import argparse, random, itertools

parser = argparse.ArgumentParser()
parser.add_argument('--temp', type=float, default=0.1,
                    help='Sampling temperature for the chat-completion call')
parser.add_argument('--no_defs', action='store_true',
                    help='DO NOT pretend one-line category definitions')
parser.add_argument('--fewshot', default=None,
                    help='Path to a JSON file with few-shot examples (omit for zero-shot)')
parser.add_argument('--no_fewshot', action='store_true',
                    help='Force zero-shot even if --fewshot is given')
parser.add_argument('--shuffle', action='store_true',
                    help='Shuffle the order of sentences inside each 50-row batch')
args = parser.parse_args()

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
    
FEWSHOT = []
if args.fewshot and not args.no_fewshot:
    fp = pathlib.Path(args.fewshot)
    FEWSHOT = json.loads(fp.read_text(encoding='utf-8'))

CATEGORIES = [
    "B1300 Energy level",
    "B140 Attention functions",
    "B152 Emotional functions",
    "B440 Respiration functions",
    "B455 Exercise tolerance functions",
    "B530 Weight maintenance functions",
    "D450 Walking",
    "D550 Eating",
    "D840-D859 Work and employment",
    "B280 Sensations of pain",
    "B134 Sleep functions",
    "D760 Family relationships",
    "B164 Higher-level cognitive functions",
    "D465 Moving around using equipment",
    "D410 Changing basic body position",
    "B230 Hearing functions",
    "D240 Handling stress and other psychological demands",
    "None"]

DEFINITION = {"B1300 Energy level": "Mental functions that produce vigour and stamina",
              "B140 Attention functions": "Specific mental functions of focusing on an external stimulus or internal experience for the required period of time",
              "B152 Emotional functions": "Specific mental functions related to the feeling and affective components of the processes of the mind",
              "B440 Respiration functions": "Functions of inhaling air into the lungs, the exchange of gases between air and blood, and exhaling air",
              "B455 Exercise tolerance functions": "Functions related to respiratory and cardiovascular capacity as required for enduring physical exertion",
              "B530 Weight maintenance functions": "Functions of maintaining appropriate body weight, including weight gain during the development period",
              "D450 Walking": "Moving along a surface on foot, step by step, so that one foot is always on the ground, such as when strolling, sauntering, walking forwards, backwards, or sideways. Include: walking short or long distances; walking on different surfaces; walking around obstacles",
              "D550 Eating": "Carrying out the coordinated tasks and actions of eating food that has been served, bringing it to the mouth and consuming it in culturally acceptable ways, cutting or breaking food into pieces, opening bottles and cans, using eating implements, having meals, feasting or dining. Exclude: ingestion functions (chewing, swallowing, etc.), appetite",
              "D840-D859 Work and employment": "apprenticeship (work preparation); acquiring, keeping and terminating a job; remunerative employment; non-remunerative employment",
              "B280 Sensations of pain": "Sensation of unpleasant feeling indicating potential or actual damage to some body structure",
              "B134 Sleep functions": "General mental functions of periodic, reversible and selective physical and mental disengagement from one's immediate environment accompanied by characteristic physiological changes",
              "D760 Family relationships": "Creating and maintaining kinship relationships, such as with members of the nuclear family, extended family, foster and adopted family and step-relationships, more distant relationships such as second cousins, or legal guardians",
              "B164 Higher-level cognitive functions": "Specific mental functions especially dependent on the frontal lobes of the brain, including complex goal-directed behaviours such as decision-making, abstract thinking, planning and carrying out plans, mental flexibility, and deciding which behaviours are appropriate under what circumstances; often called executive functions",
              "D465 Moving around using equipment": "Moving the whole body from place to place, on any surface or space, by using specific devices designed to facilitate moving or create other ways of moving around, such as with skates, skis, scuba equipment, swim fins, or moving down the street in a wheelchair or a walker",
              "D410 Changing basic body position": "Getting into and out of a body position and moving from one location to another, such as rolling from one side to the other, sitting, standing, getting up out of a chair to lie down on a bed, and getting into and out of positions of kneeling or squatting",
              "B230 Hearing functions": "Sensory functions relating to sensing the presence of sounds and discriminating the location, pitch, loudness and quality of sounds",
              "D240 Handling stress and other psychological demands": "Carrying out simple or complex and coordinated actions to manage and control the psychological demands required to carry out tasks demanding significant responsibilities and involving stress, distraction, or crises, such as taking exams, driving a vehicle during heavy traffic, putting on clothes when hurried by parents, finishing a task within a time-limit or taking care of a large group of children",
              "None": "Does not belong to any of the ICF categories in the list"}

TOPICS = [
    "Cognition", 
    "Communication", 
    "Mood and behaviour", 
    "Psychosocial wellbeing", 
    "General daily functioning", 
    "Continence", 
    "Diseases", 
    "Health condition", 
    "Mouth and nutrition", 
    "Skin", 
    "Medication", 
    "Treatments and procedures", 
    "Responsibility", 
    "Social interaction and support", 
    "Environment", 
    "Dismissal options",
    "Other"
]

TOPIC_DEFINITION = {"Cognition": "Conversation about memory, thinking, or understanding", 
            "Communication": "Conversation about speaking, understanding language, hearing, or interacting with others", 
            "Mood and behaviour": "Conversation about emotions, mood, motivation, behaviour, anxiety, or depression", 
            "Psychosocial wellbeing": "Conversation about mental wellbeing, quality of life, loneliness, coping, or emotional support", 
            "General daily functioning": "Conversation about daily activities, independence, self-care, or functioning at home", 
            "Continence": "Conversation about bladder control, bowel control, incontinence, or toilet use", 
            "Diseases": "Conversation about diagnoses, illnesses, symptoms, or medical conditions", 
            "Health condition": "Conversation about general health status, physical condition, or overal wellbeing", 
            "Mouth and nutrition": "Conversation about eating, drinking, appetite, swallowing, dental health, or nutrition", 
            "Skin": "Conversation about wounds, pressure sores, skin problems, itching, or skin care", 
            "Medication": "Conversation about medicine use, prescriptions, dosage, side effects, or medication management", 
            "Treatments and procedures": "Conversation about therapies, surgeries, rehabilitation, examinations, or medical procedures", 
            "Responsibility": "Conversation about caregiving, managing tasks, decision making, or responsibilities in daily life", 
            "Social interaction and support": "Conversation about family, friends, caregivers, social contact, or support systems", 
            "Environment": "Conversation about housing, accessibility, aids, living situation, or environmental factors", 
            "Dismissal options": "Conversation about discharge, future care arrangements, transfers, or leaving care facilities",
            "Other": "Any other topics of conversation"}

def prompt_type(prompt):
    if prompt == "categories":
        return (
                "You are an annotation assistant.\n"
                "You will receive turns from a conversation in Dutch between a healthcare professional and an elderly person. "
                "The turns are given in a JSON format.\n"
                "Your task is to assign an ICF category to each turn based on the categories in the detailed definitions. "
                "You can fill in more than one category to each turn if applicable. "
                "If you fill in None, you can NOT assign another category.\n"
                "Only fill in the category if the turn fits into the description of that category as listed in the detailed definitions."
                "ONLY assign a category if it is clearly and explicitly discussed in that turn. "
                "If it is unclear from the turn what category is discussed, do NOT fill in a category. "
                "If you are unsure, assign None. It is better to assign None than t guess.\n"
                "Do not assign a category to turns where only an introduction is given, unless the turn clearly specifies a category.\n"
                "ONLY use categories from the list above. Do NOT invent or paraphrase categories.\n"
                "Return the same number of turns as the input, with an ICF_category field added to each turn."
                "The ICF_category field should contain a list of the applicable ICF categories, and NO more.\n"
                "There should NOT be more or less turns in the output than in the input. Do NOT add or remove turns.\n"
                "Return only a valid JSON object with double quotes around all keys and string values. Do NOT use single quotes. Use colons as delimiter.\n"
                "Do not include explanations or markdown."
            )
    if prompt == "levels":
        return ("You are an annotation assistant.\n"
                "You will receive turns from a conversation in Dutch between a healthcare professional and an elderly person."
                "The turns are given in a JSON format. Each turn may have no (None), one, or more ICF categories assigned.\n"
                "Your task is to find the impairment level for each ICF category that appears in the conversation.\n"
                "The impairment level is a number from 0 to 4, where 0 means total impairment and functioning is limited, and 4 means no impairment at all.\n"
                "If no level is discussed, assign None. It is better to assign None than to guess.\n"
                "Only use a number (0, 1, 2, 3, or 4) or None for ICF_level. Do not add any text or explanation.\n"
                "Only assign a level if the turn is clearly discussing the level of function for that specific category, not just any level or number mentioned in the conversation. "
                "If the text in a turn or across turns is constantly repeating itself, only assign the level to the first two repetitions and assign None to the remaining turns that contain the same repetition."
                "The ICF_level field must contain a list with the same number of elements as the ICF_category field. "
                "Each level value should correspond to the category at the same position in the ICF_category list.\n"
                "Return the same number of turns as the input, with ICF_level updated for each turn.\n"
                "There should NOT be more or less turns in the output than in the input. Do NOT add or remove turns.\n"
                "Return only a valid JSON object with double quotes around all keys and string values. Do NOT use single quotes. Use colons as delimiter.\n"
                "Do not include explanations or markdown."
                "Do NOT assign 0 by default. 0 is a valid level only if total impairment is explicitly discussed.\n"
            )
    if prompt == "time":
        return (
                "You are an annotation assistant.\n"
                "You will receive turns from a conversation in Dutch between a healthcare professional and an elderly person. "
                "The turns are given in a JSON format. Each turn may have ICF categories and ICF levels assigned.\n"
                "Your task is to determine the relative time for each turn where an ICF level has been assigned.\n"
                "Go through the turns one by one. "
                "For each level in ICF_level assign a relative time.\n"
                "Only assign a relative time if the turn has an ICF level that is not None.\n"
                "If NO ICF level has been filled in at all, do NOT fill in the relative time but mark it with [None].\n"
                "The relative_time field must contain a list with the same number of elements as the ICF_level field. "
                "Each relative time value should correspond to the level at the same position in the ICF_level list.\n"
                "If the level has been filled out for one category, but not for another, fill in the relative time for the level that has been specified. Fill in None for elements that have not been specified."
                "Choose only from the following options:\n"
                "- None: when no ICF level was determined for that turn.\n"
                "- past: something that happened in the past and is no longer happening.\n"
                "- present: something that is currently happening, including routines and ongoing situations.\n"
                "- future: something that has not happened yet but will happen.\n"
                "Every turn must have a relative_time field with a list. This field should be equally long as the ICF_levels list.\n"
                "Do not add or remove turns. There should NOT be more or less turns in the output than in the input.\n"
                "Return only a valid JSON object with double quotes around all keys and string values. Do NOT use single quotes. Use colons as delimiter.\n"
                "Do not include explanations or markdown."
            )

# pairs each sentence back to its index and allows multi-label
def build_prompt(sentences, p, detailed_defs=True, fewshot=None):
    prompt = prompt_type(p)

    if fewshot is None:
        fewshot = []
    sys = {
        "role": "system",
        "content": prompt
    }

    if detailed_defs:
        defs = "\n".join(f"- **{cat}**: {DEFINITION[cat]}" for cat in CATEGORIES)
        #topic_defs = "\n".join(f"- **{topic}**: {TOPIC_DEFINITION[topic]}" for topic in TOPICS)

        #defs += "\n\nTopics:\n" + topic_defs
    else:
        defs = "Categories: " + ", ".join(CATEGORIES)

    example_txt = ""
    if fewshot:
        if p == "categories": 
            example_txt = "### Examples (already annotated):\n" + "\n".join(
            f"- **Sentence**: {ex['sentence']}\n"
            f"  **ICF_category**: {ex['ICF_category']}"
            for ex in fewshot
            ) + "\n\n"
        if p == "levels": 
            example_txt = "### Examples (already annotated):\n" + "\n".join(
            f"- **Sentence**: {ex['sentence']}\n"
            f"  **ICF_category**: {ex['ICF_category']}\n"
            f"  **ICF_level**: {ex['ICF_level']}"
            for ex in fewshot
            ) + "\n\n"
        elif p == "time": 
            example_txt = "### Examples (already annotated):\n" + "\n".join(
            f"- **Sentence**: {ex['sentence']}\n"
            f"  **relative_time**: {ex['relative_time']}\n"
            f"  **ICF_category**: {ex['ICF_category']}\n"
            f"  **ICF_level**: {ex['ICF_level']}"
            for ex in fewshot
            ) + "\n\n"

    user = {
        "role": "user",
        "content": (
            f"{defs}\n\n"
            f"{example_txt}"
            "### Conversation JSON:\n"
            + json.dumps(sentences, ensure_ascii=False, indent=2)
            + "\n\n"
            "### Output format:\n"
            'Return ONLY a valid JSON object.\n'
            'Preserve all turn IDs exactly.\n'
            'Do not add or remove turns.\n'
            'Annotate every turn exactly once.\n'
            'Return the updated conversation in this format (this is an example):\n\n'
            '  {\n'
            '    "filename": "conv_id",\n'
            '    "conv": {\n'
            '            "turn_1": {\n'
            '                   "speaker": "999000",\n'
            '                   "text": "Ik loop elke dag zelf naar de supermarkt.",\n'
            '                   "relative_time": ["present"],\n'
            '                   "ICF_category": ["D450"],\n'
            '                   "ICF_level": [4],\n'
            '                   },\n'
            '             }\n'
            '  }\n'
        )
    }

    return [sys, user]


def main(
    input_directory: str = "../EA_prompt_alteration",
    model: str = "gpt-4o",
    detailed_defs: bool = False,
    batch_size: int = 20
    ):

    for filename in sorted(os.listdir(input_directory)):
        #rank, filename = filename.split("_", 1)
        #print(f"Starting on file {filename} with rank {rank}.")
        extension = os.path.splitext(filename)[1]
        if extension != ".json":
            continue

        #input_json = f"{input_directory}/{rank}_{filename}"
        input_json = f"{input_directory}/{filename}"
        #output_json = f"../annotated_transcriptions/{filename}"
        output_json = f"{input_directory}/annotated_repetition/{filename}"
        # Skip if the file has already been processed
        if os.path.exists(output_json):
            continue

        prompt_types = ["categories", "levels", "time"]

        buffer = []
        global_idx = 0

        with open(input_json, "r", encoding="utf-8") as fin:
            data = json.load(fin)

        conv_id = data["filename"]

        results = {}
        results[conv_id] = {
            "conv": {}
        }
        for prompt in prompt_types:
            with open(f"../few_shot_examples/{prompt}.txt", "r", encoding = "utf-8") as f:
                examples = json.loads(f.read())

            # Reset buffer for each prompt
            buffer = []
            for turn_id, turn_data in data["conv"].items():
                
                speaker = turn_data["speaker"]
                sentence = turn_data["text"] 
            
                buffer.append((conv_id, turn_id, speaker, sentence))

                # once batch_size is hit, process
                if len(buffer) >= batch_size:
                    _flush(buffer, results, model, conv_id, detailed_defs=True, prompt_type=prompt, fewshot_examples=examples)
                   
                    buffer.clear()

            if buffer:
                _flush(buffer, results, model, conv_id, detailed_defs=True, prompt_type=prompt, fewshot_examples=examples)

        with open(output_json, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)
        print(f"Wrote {len(results)} annotated sentences to {output_json}")

FLUSH_COUNT = 0

def _flush(buffer, results, model, conv_id, detailed_defs, prompt_type, fewshot_examples):
    '''
    Send buffer of (note_id, idx, sentence) to GPT,
    parse the JSON and appends to results.
    '''
    global FLUSH_COUNT
    FLUSH_COUNT += 1
    print(f"[{time.strftime('%H:%M:%S')}] processed {FLUSH_COUNT*len(buffer):,} sentences ...")

    # extract just the sentences for the prompt
    sentences = {turn_id:
                    {"conv_id": conv_id,
                    "speaker": speaker,
                    "text": sentence
                    }
                for conv_id, turn_id, speaker, sentence in buffer}
    msgs = build_prompt(
            sentences,
            prompt_type,
            detailed_defs = (not args.no_defs),
            fewshot = fewshot_examples
    )

    resp = client.chat.completions.create(
        model = model,
        messages = msgs,
        temperature = args.temp,
        response_format = {"type": "json_object"}
    )
    raw = resp.choices[0].message.content.strip()

    try:
        annos = json.loads(raw)
    except json.JSONDecodeError:
        print("Retrying...")

        resp = client.chat.completions.create(
            model = model,
            messages = msgs,
            temperature = args.temp,
            response_format = {"type": "json_object"}
        )
        raw = resp.choices[0].message.content.strip()

        try:
            annos = json.loads(raw)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON (after extraction):", e)
            print("JSON substring was:\n", raw[:800])
        
            print("Continuing...")
            return

    if not annos or not annos.get("conv"):
        print(f"Annos only contained {annos.keys()}")
        print("Empty response")
        return

    
    updated_conv = annos["conv"]

    for turn_id, turn_data in updated_conv.items():
        if turn_id not in results[conv_id]["conv"]:
            results[conv_id]["conv"][turn_id] = {}
        results[conv_id]["conv"][turn_id].update(turn_data)

   
if __name__ == "__main__":
    main(detailed_defs=True)
