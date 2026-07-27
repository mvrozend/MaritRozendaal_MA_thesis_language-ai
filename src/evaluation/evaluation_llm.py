import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ast import literal_eval
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset

import utils_evaluation as eval

category_names = [
    "B1300 Energy level",
    "B140 Attention functions",
    "B152 Emotional functions",
    "B440 Respiration functions",
    "B455 Exercise tolerance functions",
    "B530 Weight maintenance functions",
    "D450 Walking",
    "D550 Eating",
    "D840-859 Work and employment",
    "B280 Sensations of pain",
    "B134 Sleep functions",
    "D760 Family relationships",
    "B164 Higher-level cognitive functions",
    "D465 Moving around using equipment",
    "D410 Changing basic body position",
    "B230 Hearing functions",
    "D240 Handling stress and other psychological demands",
    "None"
]

time_names = [
    "past",
    "present",
    "future",
    "None"
]

label_map = {label.split()[0]: label for label in category_names}

def extend_df(df):
    """
        Extend the dataframe such that every row only has one category

        :param df: Dataframe
    """
    expanded_rows = []
    
    for i, row in df.iterrows():
    
        categories = row["categories"]
        levels = row["levels"]
        
        predicted_categories = row["predicted_categories"]
        predicted_levels = row["predicted_levels"]
        
        gold_map = dict(zip(categories, levels))
        pred_map = dict(zip(predicted_categories, predicted_levels))
        
        # Collect all categories in either categories or predicted_categories
        all_categories = []
        
        for cat in categories:
            if cat not in all_categories:
                all_categories.append(cat)
        
        for cat in predicted_categories:
            if cat not in all_categories:
                all_categories.append(cat)
        
        for j, cat in enumerate(all_categories):
        
            new_row = row.copy()
            
            new_row["full_turn_id"] = i
            new_row["turn_id"] = f"{i}_{j}"
            
            # gold side
            if cat in gold_map:
                new_row["category"] = cat
                new_row["level"] = gold_map[cat]
            else:
                new_row["category"] = "None"
                new_row["level"] = "None"
            
            # predicted side
            if cat in pred_map:
                new_row["predicted_category"] = cat
                new_row["predicted_level"] = pred_map[cat]
            else:
                new_row["predicted_category"] = "None"
                new_row["predicted_level"] = "None"
            
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows)

    
def draw_cm_strict(df, names, model_name, data, gold_col, pred_col):
    """
        Draw a confusion matrix for a strictly filtered dataframe.

        :param df: Dataframe
        :param names: List of strings with all names of the labels
        :param model_name: String with the name of the model
        :param data: String with the name of the data
        :param gold_col: String of the name of the gold labels
        :param pred_col: String of the name of the predicted labels
    """
    cm = pd.DataFrame(0, index=names, columns=names)
    
    for _, row in df.iterrows():
        gold_labels = [x.strip("'") for x in row[gold_col] if x not in [None, "None"]]
        pred_labels = [x.strip("'") for x in list(row[pred_col]) if x not in [None, "None"]]
        
        for gold in gold_labels:
            if gold in pred_labels:
                cm.loc[gold, gold] += 1
            elif pred_labels:
                for pred in pred_labels:
                    if pred in names:
                        cm.loc[gold, pred] += 1
            else:
                cm.loc[gold, "None"] += 1
              
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap= "Greens")
    
    plt.title(f"Confusion Matrix ({model_name}|{data})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.xticks(rotation=45, ha="right")
    
    plt.show()  

def draw_cm_lenient(df, names, model_name, data, gold_col, pred_col):
    """
        Draw a confusion matrix for a leniently filtered dataframe.

        :param df: Dataframe
        :param names: List of strings with all names of the labels
        :param model_name: String with the name of the model
        :param data: String with the name of the data
        :param gold_col: String of the name of the gold labels
        :param pred_col: String of the name of the predicted labels
    """
    cm = pd.DataFrame(0, index=names, columns=names)
    
    for _, row in df.iterrows():
        gold_labels = [x for x in row[gold_col] if x not in [None, "None"]]
        if isinstance(row[pred_col], tuple):
            pred_labels = [x for x in list(row[pred_col]) if x not in [None, "None"]]
        if isinstance(row[pred_col], list):
            pred_labels = [x for x in row[pred_col] if x not in [None, "None"]]
        else:
            pred_labels = []
        
        for gold in gold_labels:
            if gold in pred_labels:
                cm.loc[gold, gold] += 1
            elif pred_labels:
                for pred in pred_labels:
                    if pred in names:
                        cm.loc[gold, pred] += 1
            else:
                cm.loc[gold, "None"] += 1
        for pred in pred_labels:
            if pred not in gold_labels:
                if gold_labels:
                    for gold in gold_labels:
                        if gold in names:
                            cm.loc[gold, pred] += 1
                                    
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap= "Greens")
    
    plt.title(f"Confusion Matrix ({model_name}|{data})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.xticks(rotation=45, ha="right")
    
    plt.show() 
    
def is_none_level(x):
    if pd.isna(x):
        return True

    if x is None:
        return True

    if x == -1:
        return True

    if x == "None":
        return True

    if isinstance(x, list):
        return len(x) > 0 and x[0] in ("None", None, -1)

    return False

def complete_label(label):
    if label in [None, "", "None"]:
        return "None"
        
    parts = str(label).split()
    if len(parts) == 0:
        return "None"
        
    code = parts[0] 
    return label_map.get(code, "None")


def category_predictions(test_df, mlb, model_name, data, filtering):
    """
    Predict the categories for the test data and evaluate the results. 
    """
    test_df["predicted_categories"] = test_df["predicted_categories"].apply(lambda labels: [complete_label(label) for label in labels])
    
    print(f"Results strict evaluation categories for {model_name} on {data}.")
    strict_report_df, strict_df = eval.eval_strict(test_df, "categories", "predicted_categories", mlb)
    strict_report_df.to_csv(f"./cr_results/cat_{model_name}_{data}_filtering{filtering}_evalstrict", index=False)
    draw_cm_strict(strict_df, category_names, model_name, data, "categories", "predicted_categories")

    print(f"Results lenient evaluation categories for {model_name} on {data}.")
    lenient_report_df, lenient_df = eval.eval_lenient(test_df, "categories", "predicted_categories", mlb)
    lenient_report_df.to_csv(f"./cr_results/cat_{model_name}_{data}_filtering{filtering}_evallenient", index=False)
    draw_cm_lenient(lenient_df, category_names, model_name, data, "categories", "predicted_categories")


def level_predictions(df, model_name, data, filtering):
    """
    Predict the levels for the test data and evaluate the results. 
    """
    df = df.copy()
    df = extend_df(df)
    
    results_strict = eval.level_eval_strict(df)
    results_lenient = eval.level_eval_lenient(df)
    
    print(f"Results {filtering} filtering and strict evaluation levels ({model_name}|{data})")
    print(results_strict)
    print(f"Results {filtering} filtering and lenient evaluation levels ({model_name}|{data})")
    print(results_lenient)
    

def time_predictions(test_df, mlb, model_name, data, filtering):
    """
    Predict the time periods for the test data and evaluate the results. 
    """
    print(f"Results strict evaluation relative time for {model_name} on {data}.")
    strict_report_df, strict_df = eval.eval_strict(test_df, "relative_time", "predicted_time", mlb)
    strict_report_df.to_csv(f"./cr_results/time_{model_name}_{data}_filtering{filtering}_evalstrict", index=False)
    draw_cm_strict(strict_df, time_names, model_name, data, "relative_time", "predicted_time")


    print(f"Results lenient evaluation relative time for {model_name} on {data}.")
    lenient_report_df, lenient_df = eval.eval_lenient(test_df, "relative_time", "predicted_time", mlb)
    lenient_report_df.to_csv(f"./cr_results/time_{model_name}_{data}_filtering{filtering}_evallenient", index=False)
    draw_cm_lenient(lenient_df, time_names, model_name, data, "relative_time", "predicted_time")
    
def main(
        model_name = "LLM",
        data = "fewshot",
        input_dir_test="prepared_fewshot_test_data"
    ):
    test_df_strict, test_df_lenient = eval.combine_df(input_dir_test)
    
    test_df_strict.to_csv("./filtered_test_data_llm_strict.csv", index=False)
    test_df_strict = test_df_strict[["conversation_id", "turn", "speaker", "text", "categories", "levels", "relative_time", "predicted_categories", "predicted_levels", "predicted_time", "segment_id", "segment_category"]]
    # Drop rows with NaN for text
    test_df_strict = test_df_strict.dropna(subset=["text"])
    
    test_df_lenient.to_csv("./filtered_test_data_llm_lenient.csv", index=False)
    test_df_lenient = test_df_lenient[["conversation_id", "turn", "speaker", "text", "categories", "levels", "relative_time", "predicted_categories", "predicted_levels", "predicted_time", "segment_id", "segment_category"]]
    # Drop rows with NaN for text
    test_df_lenient = test_df_lenient.dropna(subset=["text"])
    
    categories_mlb = MultiLabelBinarizer(classes=category_names)
    categories_mlb.fit([category_names])
    time_mlb = MultiLabelBinarizer(classes=time_names)
    time_mlb.fit([time_names])
    
    
    for test_df, filtering in zip([test_df_strict, test_df_lenient], ["strict", "lenient"]):
        test_df["predicted_categories"] = test_df["predicted_categories"].apply(lambda x: "['None']" if pd.isna(x) else x)
        test_df["predicted_categories"] = test_df["predicted_categories"].apply(literal_eval)   
        
        test_df["predicted_levels"] = test_df["predicted_levels"].apply(lambda x: "['None']" if pd.isna(x) else x)
        test_df["predicted_levels"] = test_df["predicted_levels"].apply(literal_eval)     
              
        test_df["predicted_time"] = test_df["predicted_time"].apply(lambda x: "['None']" if pd.isna(x) else x) 
        test_df["predicted_time"] = test_df["predicted_time"].apply(literal_eval)
        
        
        category_predictions(test_df, categories_mlb, model_name, data, filtering)
    
        level_predictions(test_df, model_name, data, filtering)
    
        time_predictions(test_df, time_mlb, model_name, data, filtering)
            


if __name__ == "__main__":
    main()

    