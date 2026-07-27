from ast import literal_eval
import ast
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score

def filter_conversation_strict(conv_df, file):
    """
        Filtering of the conversations with strict segmentation. The filter starts at the mention of 
        the category and stop when the level of that category is specified.

        :param conv_df: Dataframe with the conversations
        :param file: string with the file ID
    """
    conv_df = conv_df.copy()
    og_length = len(conv_df)
   
    conv_df["categories"] = conv_df["categories"].apply(literal_eval)  
    conv_df["levels"] = conv_df["levels"].apply(literal_eval)    
    conv_df["relative_time"] = conv_df["relative_time"].apply(literal_eval)
   
    conv_df = conv_df[~((conv_df["text"] == conv_df["text"].shift(1))
                      & (conv_df["text"] == conv_df["text"].shift(2)))].copy()
    print(f"{og_length - len(conv_df)} duplicate rows removed in strict filtering.")
    segment_dfs = []
    segment_id = 0
   
    open_segments = {}
   
    for idx, (_, row) in enumerate(conv_df.iterrows()):
        categories = row["categories"]
        levels = row["levels"]
           
        for cat_idx, cat in enumerate(categories):
            if cat in [None, "None"]:
                continue
           
            if cat not in open_segments:
                open_segments[cat] = {
                    "start_idx": idx,
                    "category_index": cat_idx
                    }
                   
        for cat_idx, (cat, level) in enumerate(zip(categories, levels)):
            if cat in [None, "None"]:
                continue
            if level in [None, "None", -1]:
                continue
            if cat not in open_segments:
                continue
             
            start_idx = open_segments[cat]["start_idx"]
            end_idx = idx
           
            segment_df = conv_df.iloc[start_idx:end_idx + 1].copy()
           
            segment_df["segment_id"] = segment_id
            segment_df["segment_category"] = cat
            segment_df["segment_level"] = levels[categories.index(cat)]
            segment_df["file"] = file
           
            segment_df["categories"] = segment_df["categories"].apply(
                lambda cats: [cat] if isinstance(cats, list) and cat in cats else ["None"])
               
            segment_df["levels"] = segment_df.apply(
                lambda r: [r["levels"][r["categories"].index(cat)]]
                if (isinstance(r["categories"], list)
                    and isinstance(r["levels"], list)
                    and cat in r["categories"])
                else ["None"], axis=1,)
               
            segment_dfs.append(segment_df)
            segment_id += 1
            del open_segments[cat]
           
    if not segment_dfs:
        return pd.DataFrame()
   
    strict_df = pd.concat(segment_dfs, ignore_index=True)
    print(f"{file}: {len(strict_df)} turns in strict filtering (from {og_length} turns).")
    print(f"There are {strict_df['segment_id'].nunique()} unique segments in strict filtering.")
   
    return strict_df
           
def filter_conversation_lenient(conv_df, file, context_size=10):
    """
        Filtering of the conversations with lenient segmentation. The filter starts 10 turns before 
        the category is mentioned and 10 turns after the level of that category is specified.

        :param conv_df: Dataframe with the conversations
        :param file: string with the file ID
        :param context_size: integer which specifies how many extra turns are used as context
    """
    conv_df = conv_df.copy()
    og_length = len(conv_df)
   
    conv_df["categories"] = conv_df["categories"].apply(literal_eval)  
    conv_df["levels"] = conv_df["levels"].apply(literal_eval)    
    conv_df["relative_time"] = conv_df["relative_time"].apply(literal_eval)
   
    conv_df = conv_df[~((conv_df["text"] == conv_df["text"].shift(1))
                      & (conv_df["text"] == conv_df["text"].shift(2)))].copy()
    print(f"{og_length - len(conv_df)} duplicate rows removed in lenient filtering.")
           
    segment_dfs = []
    segment_id = 0
   
    open_segments = {}
   
    for idx, (_, row) in enumerate(conv_df.iterrows()):
        categories = row["categories"]
        levels = row["levels"]
           
        for cat_idx, cat in enumerate(categories):
            if cat in [None, "None"]:
                continue
           
            if cat not in open_segments:
                open_segments[cat] = {
                    "start_idx": idx,
                    "category_index": cat_idx
                    }
                   
        for cat_idx, (cat, level) in enumerate(zip(categories, levels)):
            if cat in [None, "None"]:
                continue
            if level in [None, "None", -1]:
                continue
            if cat not in open_segments:
                continue
             
            start_idx = max(0, open_segments[cat]["start_idx"] - context_size)
            end_idx = min(len(conv_df) - 1, idx + context_size)
               
            segment_df = conv_df.iloc[start_idx:end_idx + 1].copy()
           
            segment_df["segment_id"] = segment_id
            segment_df["segment_category"] = cat
            segment_df["segment_level"] = levels[categories.index(cat)]
            segment_df["file"] = file
           
            segment_df["categories"] = segment_df["categories"].apply(
                lambda cats: [cat] if isinstance(cats, list) and cat in cats else ["None"])
               
            segment_df["levels"] = segment_df.apply(
                lambda r: [r["levels"][r["categories"].index(cat)]]
                if (isinstance(r["categories"], list)
                    and isinstance(r["levels"], list)
                    and cat in r["categories"])
                else ["None"], axis=1,)
               
            segment_dfs.append(segment_df)
            segment_id += 1
            del open_segments[cat]
           
    if not segment_dfs:
        return pd.DataFrame()
        
    lenient_df = pd.concat(segment_dfs, ignore_index=True)
    print(f"{file}: {len(lenient_df)} turns in lenient filtering.")
    print(f"There are {lenient_df['segment_id'].nunique()} unique segments in lenient filtering.")
   
    return lenient_df    

def calculate_time_difference(baseline_df, additional_df, annotator, sl):
    """
        Calculate the difference in levels between two annotated segments.

        :param baseline_df: Dataframe with the baseline annotations
        :param additional_df: Dataframe with the additional annotations
        :param annotator: annotator ID
        :param sl: string with "strict" or "lenient"
    """
    differences = []

    if sl == "strict": 
        additional_df["categories"] = additional_df["categories"].apply(literal_eval)
        additional_df["levels"] = additional_df["levels"].apply(literal_eval)

    for segment_id in baseline_df["segment_id"].unique():
        # Get all turns in segment
        segment = baseline_df[baseline_df["segment_id"] == segment_id]

        # Find segment in additional_df
        comparison = additional_df.merge(
            segment[["conversation_id", "turn"]],
            on=["conversation_id", "turn"],
            how="inner"
        )

        category = segment["segment_category"].iloc[0]
        baseline_level = segment["segment_level"].iloc[0]

        for _, row in comparison.iterrows():
            if category in row["categories"]:
                i = row["categories"].index(category)
                additional_level = row["levels"][i]
                if additional_level in ["None", None]:
                    print("No comparison could be done for segment", segment_id)
                    difference = None
                else:
                    additional_level = int(additional_level)
                    difference = abs(baseline_level - additional_level)

                differences.append({
                    "conversation_id": baseline_df["conversation_id"][0],
                    "annotator": annotator,
                    "segment_id": segment_id,
                    "category": category,
                    "filter": sl,
                    "baseline_level": baseline_level,
                    "additional_level": additional_level,
                    "difference": difference
                })
            
                break
    
    return pd.DataFrame(differences)

def main(
        input_path_baseline: str,
        input_path_additional: str
):
    baseline_df = pd.read_csv(input_path_baseline)

    strict_baseline_df = filter_conversation_strict(baseline_df, "baseline")
    lenient_baseline_df = filter_conversation_lenient(baseline_df, "baseline")

    additional_df = pd.read_csv(input_path_additional)

    annotator = 11

    print("Calculating similarity for strict filtering...")
    strict_df = calculate_time_difference(strict_baseline_df, additional_df, annotator, "strict")
    results_strict_df = pd.DataFrame({
        "score": [strict_df["difference"].dropna().mean()],
        "filter": "strict",
        "annotator": annotator
    })
    print(results_strict_df)


    print("Calculating similarity for lenient filtering...")
    lenient_df = calculate_time_difference(lenient_baseline_df, additional_df, annotator, "lenient")
    lenient_df = lenient_df.merge(
        strict_df[["segment_id", "category"]],
        on=["segment_id", "category"],
        how="inner"
    )
    results_lenient_df = pd.DataFrame({
        "score": [lenient_df["difference"].dropna().mean()],
        "filter": "lenient",
        "annotator": annotator
    })
    print(results_lenient_df)


    differences = pd.concat([strict_df, lenient_df], ignore_index=True)
    results = pd.concat([results_strict_df, results_lenient_df], ignore_index=True)

    
if __name__ == "__main__":
    main()