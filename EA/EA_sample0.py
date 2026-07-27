import os
import json
import pandas as pd


def get_top10(folder_name): 
    # Get files from folder
    results = []

    for filename in os.listdir(folder_name):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(folder_name, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        file_key = next(iter(data))

        turns = data[file_key]["conv"].values()

        total = len(turns)
        zeros = sum(
            1 for turn in turns
            if 0 in turn["ICF_level"]
        )

        results.append({
            "file": filename,
            "ratio": zeros / total
        })

    top10 = (
        pd.DataFrame(results)
        .sort_values("ratio", ascending=False)
        .head(10)
    )
    print("THIS IS THE TOP 10:")
    print(top10)

    return top10

def get_samples(top10, folder_name): 
    # Get samples from file
    window_size = 12
    context = 20
    rows = []

    for file in top10["file"]:

        with open(os.path.join(folder_name, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        file_key = next(iter(data))

        turn_names = list(data[file_key]["conv"].keys())
        turns = list(data[file_key]["conv"].values())

        best_start = 0
        best_count = -1

        for start in range(len(turns) - window_size + 1):

            window = turns[start:start + window_size]

            count = sum(0 in turn["ICF_level"] for turn in window)

            if count > best_count:
                best_count = count
                best_start = start

        best_end = best_start + window_size - 1
        sample_start = max(0, best_start - context)
        sample_end = min(len(turns) - 1, best_end + context)

        sample_conv = {}
        for i in range(sample_start, sample_end + 1):
            turn = turns[i].copy()
            turn["ICF_category"] = None
            turn["ICF_level"] = None
            turn["relative_time"] = None
            sample_conv[turn_names[i]] = turn
    
        sample_json = {
            "filename": file_key,
            "conv": sample_conv
        }
        with open(f"../EA_prompt_alteration/{file}", "w", encoding="utf-8") as f:
            json.dump(sample_json, f, indent=2, ensure_ascii=False)

        for i in range(best_start, best_start + window_size):
            turn = turns[i]
            rows.append({
                "file": file,
                "turn": turn_names[i],
                "speaker": turn["speaker"],
                "text": turn["text"],
                "ICF_categories": turn["ICF_category"],
                "ICF_levels": turn["ICF_level"],
                "relative_time": turn["relative_time"]

            })

    samples_df = pd.DataFrame(rows)
    


def main():
    input_folder = "../annotated_transcriptions"

    top10 = get_top10(input_folder)

    get_samples(top10, input_folder)

if __name__ == "__main__":
    main()