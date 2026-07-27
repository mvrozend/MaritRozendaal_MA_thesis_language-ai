import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ast import literal_eval
from sklearn.metrics import classification_report, multilabel_confusion_matrix, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer

          
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
    (f"{og_length - len(conv_df)} duplicate rows removed in strict filtering.")
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
          
def combine_df(input_dir):
    """
        Combine all test files
    """
    dfs_strict = []
    dfs_lenient = []

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, file))
            df_strict = filter_conversation_strict(df, file)
            df_lenient = filter_conversation_lenient(df, file)
            dfs_strict.append(df_strict)
            dfs_lenient.append(df_lenient)

    combined_df_strict = pd.concat(dfs_strict, ignore_index=True)
    combined_df_lenient = pd.concat(dfs_lenient, ignore_index=True)

    return combined_df_strict, combined_df_lenient

def eval_strict(df, gold_col, pred_col, mlb):
    """
        Evaluate the df turn-by-turn.

        :param df: Dataframe
        :param gold_col: String with column name of the gold labels
        :param pred_col: String with column name of the predicted labels
        :param mlb: MultiLabelBinarizer
    """
    df[gold_col] = df[gold_col].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df[pred_col] = df[pred_col].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    
    gold = [[label for label in labels if label not in [None, "None"]] 
             #if isinstance(labels, list) else []
             for labels in df[gold_col]]
    
    pred = [[label for label in labels if label not in [None, "None"]] 
             #if isinstance(labels, tuple) else []
             for labels in df[pred_col]]
             
    y_true = mlb.transform(gold)
    y_pred = mlb.transform(pred)
    
    report = classification_report(
                          y_true,
                          y_pred,
                          target_names=mlb.classes_,
                          output_dict=True,
                          zero_division=0)
                          
    report_df = pd.DataFrame(report).transpose()
    
    return report_df, df
    
def eval_lenient(df, gold_col, pred_col, mlb):
    """
        Evaluate the df within a segment.

        :param df: Dataframe
        :param gold_col: String with column name of the gold labels
        :param pred_col: String with column name of the predicted labels
        :param mlb: MultiLabelBinarizer
    """
    gold_segments = []
    pred_segments = []
    
    for _, segment in df.groupby(["conversation_id", "segment_id"]):
        if gold_col == "categories":
            segment_category = segment["segment_category"].iloc[0]
            
            gold = [[segment_category]]
            
            pred_present = any(segment_category in (literal_eval(labels) if isinstance(labels, str) else labels) 
                               for labels in segment[pred_col])
                               
            pred = [[segment_category]] if pred_present else [[]]
            
            gold_segments.extend(gold)
            pred_segments.extend(pred)
        if gold_col == "relative_time":
            gold = sorted({label for labels in segment[gold_col]
                                 for label in labels
                                 if label not in [None, "None"]})
            pred = sorted({label for labels in segment[pred_col]
                                 for label in labels
                                 if label not in [None, "None"]})
                                 
            gold_segments.append(gold)
            pred_segments.append(pred)
        
    y_true = mlb.transform(gold_segments)
    y_pred = mlb.transform(pred_segments)
    
    report = classification_report(
                          y_true,
                          y_pred,
                          target_names=mlb.classes_,
                          output_dict=True,
                          zero_division=0)
                          
    report_df = pd.DataFrame(report).transpose()
    
    segment_eval_df = pd.DataFrame({gold_col: gold_segments, 
                                    pred_col: pred_segments})
                             
    
    return report_df, segment_eval_df
    
def level_eval_lenient(df):
    """
        Evaluate the levels in the df turn-by-turn.

        :param df: Dataframe
    """
    FP = FP1a = FP1b = FP2a = FP2b = 0
    FN = FN1a = FN1b = FN2a = FN2b = 0
    
    gold_values = []
    pred_values = []
    
    gold = None
    pred = None
    
    for _, segment in df.groupby(["conversation_id", "segment_id"]):
        segment_category = segment["segment_category"].iloc[0]
        
        for i, row in segment.iterrows():
            if row["category"] == segment_category:
                if row["level"] not in [None, "None", -1]:
                    gold = float(row["level"])
                    

            if row["predicted_category"] == segment_category:
                if row["predicted_level"] not in [None, "None", -1]:
                    pred = float(row["predicted_level"])
                    
        gold_cat_present = any(
            row["category"] == segment_category
            and row["category"] not in [None, "None"]
            for _, row in segment.iterrows()
            )
            
        pred_cat_present = any(
            row["predicted_category"] == segment_category
            and row["predicted_category"] not in [None, "None"]
            for _, row in segment.iterrows()
            )
        
        if gold is None and pred is not None:
            FP += 1
            if gold_cat_present and pred_cat_present:
                FP1a += 1
            elif gold_cat_present and not pred_cat_present:
                FP1b += 1
            elif not gold_cat_present and pred_cat_present:
                FP2a += 1
            else:
                FP2b += 1
                
        elif gold is not None and pred is None:
            FN += 1
            if gold_cat_present and pred_cat_present:
                FN1a += 1
            elif gold_cat_present and not pred_cat_present:
                FN1b += 1
            elif not gold_cat_present and pred_cat_present:
                FN2a += 1
            else:
                FN2b += 1
                
        elif gold is not None and pred is not None:
            gold_values.append(gold)
            pred_values.append(pred)
            
    mae = mean_absolute_error(gold_values, pred_values)
    mse = mean_squared_error(gold_values, pred_values)
    rmse = np.sqrt(mse)
    
    return {
        "FP": FP,
        "FN": FN,
        "FP1a": FP1a,
        "FP1b": FP1b,
        "FP2a": FP2a,
        "FP2b": FP2b,
        "FN1a": FN1a,
        "FN1b": FN1b,
        "FN2a": FN2a,
        "FN2b": FN2b,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "n_level_pairs": len(gold_values)
        } 
        
    
def level_eval_strict(df):
    """
        Evaluate the levels in the df within a segment.

        :param df: Dataframe
    """
    FP = FP1a = FP1b = FP2a = FP2b = 0
    FN = FN1a = FN1b = FN2a = FN2b = 0
    
    gold_values = []
    pred_values = []
    
    for _, row in df.iterrows():
        gold = row["level"]
        pred = row["predicted_level"]
            
        gold_cat_present = (row["category"] not in [None, "None"])
        pred_cat_present = (row["predicted_category"] == row["category"])
        
        gold_is_none = gold in [None, "None", -1]
        pred_is_none = pred in [None, "None", -1]
        
        if gold_is_none and not pred_is_none:
            FP += 1
            if gold_cat_present and pred_cat_present:
                FP1a += 1
            elif gold_cat_present and not pred_cat_present:
                FP1b += 1
            elif not gold_cat_present and pred_cat_present:
                FP2a += 1
            else:
                FP2b += 1
                
        elif not gold_is_none and pred_is_none:
            FN += 1
            if gold_cat_present and pred_cat_present:
                FN1a += 1
            elif gold_cat_present and not pred_cat_present:
                FN1b += 1
            elif not gold_cat_present and pred_cat_present:
                FN2a += 1
            else:
                FN2b += 1
                
        elif not gold_is_none and not pred_is_none:
            gold_values.append(float(gold))
            pred_values.append(float(pred))
                
    mae = mean_absolute_error(gold_values, pred_values) if gold_values else np.nan
    mse = mean_squared_error(gold_values, pred_values) if gold_values else np.nan
    rmse = np.sqrt(mse) if gold_values else np.nan
    
    return {
        "FP": FP,
        "FN": FN,
        "FP1a": FP1a,
        "FP1b": FP1b,
        "FP2a": FP2a,
        "FP2b": FP2b,
        "FN1a": FN1a,
        "FN1b": FN1b,
        "FN2a": FN2a,
        "FN2b": FN2b,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "n_level_pairs": len(gold_values)
        } 
        
    
    
    
    
        
    
  
    
    
    
    
    
    
    
    
    