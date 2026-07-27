import pandas as pd
from ast import literal_eval
import json


def extract_mentions(conversations, cats, levs, time):
    """
        Extract mentions of a level and time.

        :param conversations: Dataframe
        :param cats: String with column name of the categories
        :param levs: String with column name of the levels
        :param time: String with column name of the relative time
    """
    mentions = []

    for conv_name, (df, present_time) in conversations.items():
        df[cats] = df[cats].apply(literal_eval)
        df[levs] = df[levs].apply(literal_eval)
        df[time] = df[time].apply(literal_eval)

        for _, row in df.iterrows():
            speaker = row["speaker_llm"]

            for cat, level, t in zip(row[cats], row[levs], row[time]):
                if cat in [None, "None"]:
                    continue

                if t == "past":
                    timepoint = present_time - 1
                elif t == "present":
                    timepoint = present_time
                elif t == "future":
                    timepoint = present_time + 1
                else:
                    continue

                mentions.append({
                    "conversation": conv_name,
                    "timepoint": timepoint,
                    "speaker": speaker,
                    "category": cat,
                    "level": level
                    }) 
    return pd.DataFrame(mentions)

def statistics(mentions_df):
    """
        Calculate the number of mentions per category per time point and level

        :param mentions_df: Dataframe
    """
    # Count the number of mentions per category
    category_counts = mentions_df.groupby("category").size().reset_index(name="count")
    #print(f"Category counts:\n{category_counts}\n")

    summary = (mentions_df.groupby(["timepoint", "speaker", "category"])["level"]
               .agg(["mean", "std", "min", "max", "count"])
               .reset_index()
    )    
    #print(summary)

    return category_counts, summary

def get_triples(mentions_df):
    """
        Extract triples

        :param mentions_df: Dataframe
    """
    triples = []

    for _, row in mentions_df.iterrows():
        triples.append((row["timepoint"], row["category"], row["level"]))

        return pd.DataFrame(triples, columns=["timepoint", "category", "level"])


def main(
    model="medRoBERTa",
    context_cats="catpred_levpred"
    ):

    files = []
    """df = pd.read_csv(f"{model}_results/convdata/KG_results_{context_cats}_filteringlenient.csv")
    for file in files:
        df[df["conversation_id"] == file].to_csv(f"KG/{file}_{model}_{context_cats}.csv", index=False)"""
        
    # Load the conversations
    T0 = pd.read_csv(f"file1")
    T1 = pd.read_csv(f"file2")
    T2 = pd.read_csv(f"file3")

    conversations = {
        "T0": (T0, 1),
        "T1": (T1, 2),
        "T2": (T2, 3)
        }

    # Extract mentions
    mentions_df = extract_mentions(conversations, "categories", "levels", "relative_time")
    mentions_df.to_csv("KG/gold_mentions.csv", index=False)
    #mentions_df = pd.read_csv("KG/mentions.csv")

    # Summary of mentions at all time points
    category_counts, summary = statistics(mentions_df)
    summary.to_csv("KG/gold_summary.csv", index=False)
    #summary = pd.read_csv("KG/summary.csv")
    
    # Summary of mentions at time point t2
    t2_summary = summary[summary["timepoint"] == 2]
    t2_summary.to_csv("KG/gold_t2_summary.csv", index=False)
    #t2_summary = pd.read_csv("KG/t2_summary.csv")

    triples_df = get_triples(mentions_df)
    print(triples_df)
    
    
if __name__ == "__main__":
    main()