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

def tokenise_categories_function(examples, tokeniser):
    """
    Tokenise the input text for the category model. The input text is the original text of the turn.
    """
    return tokeniser(examples["text"], 
                     truncation=True, 
                     padding="max_length", 
                     max_length=512)

def tokenise_levels_function(examples, tokeniser):
    """
    Tokenise the input text for the level model. The input text is the original text of the turn combined with the categories.
    """
    return tokeniser(examples["combined_text"], 
                     truncation=True,  
                     padding="max_length", 
                     max_length=512)

def tokenise_time_function(examples, tokeniser):
    """
    Tokenise the input text for the time model. The input text is the original text of the turn combined with the categories and levels.
    """
    return tokeniser(examples["double_combined_text"], 
                     truncation=True,  
                     padding="max_length", 
                     max_length=512)

def run_multilabel_predictions(trainer, dataset):
    """
    Run multi-label predictions. The output is a one-hot encoded array of the predicted labels.
    """
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    probabilities = torch.sigmoid(torch.tensor(logits))
    one_hot_predictions = (probabilities > 0.5).int()
    
    empty_rows = one_hot_predictions.sum(dim=1) == 0
    one_hot_predictions[empty_rows, -1] = 1
    
    return one_hot_predictions

def decode_categories(prediction):
    """
    Decode the categories from a one-hot encoded array.
    """
    return [category_names[i] for i, value in enumerate(prediction) if value == 1]

def encode_categories(category_list, all_categories):
    """
    Encode the categories into a one-hot encoded array.
    """
    if isinstance(category_list[0], list):
        category_list = category_list[0]
    return [1 if cat in category_list else 0 for cat in all_categories]

def extend_df(df, cat_col):
    """
        Extend the dataframe such that every row only has one category

        :param df: Dataframe
    """
    expanded_rows = []

    for i, row in df.iterrows():
        categories = row["categories"]
        levels = row["levels"]
        predicted_categories = row["predicted_categories"]

        cat_to_level = dict(zip(categories, levels))

        for j in range(len(row[cat_col])):
            new_row = row.copy()
            new_row["full_turn_id"] = i
            new_row["turn_id"] = f"{i}_{j}"

            if cat_col == "categories":
                new_row["category"] = categories[j]
                new_row["level"] = levels[j]

                if j < len(predicted_categories):
                    new_row["predicted_category"] = predicted_categories[j]
                else:
                    new_row["predicted_category"] = "None"

            elif cat_col == "predicted_categories":
                pred_cat = predicted_categories[j]

                new_row["predicted_category"] = pred_cat
                new_row["level"] = cat_to_level.get(pred_cat, "None")

                if pred_cat in categories:
                    new_row["category"] = pred_cat
                else:
                    new_row["category"] = "None"

            new_row["predicted_level"] = "None"

            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)

def build_levels_input(df, category_type, history_size=5):
    """
        Build the turns with the extra context to feed the level regression model.

        :param df: Dataframe
        :param category_type: String with either cat_gold or cat_pred
        :param history_size: Integer with the size of the history (default=5)
    """
    turn_categories = {}

    for (conv_id, turn_id), turn_rows in df.groupby(["conversation_id", "full_turn_id"], sort=False):
        turn_categories[(conv_id, turn_id)] = "".join(f"[{cat}]" for cat in turn_rows[category_type])

    turn_histories  = {}

    for conv_id, conv_df in df.groupby("conversation_id", sort=False):
        turn_ids = conv_df["full_turn_id"].drop_duplicates().tolist()

        for i, turn_id in enumerate(turn_ids):
            previous_turns = turn_ids[max(0, i-history_size):i]

            history = " ".join(turn_categories[(conv_id, prev_turn)] for prev_turn in previous_turns)

            turn_histories[(conv_id, turn_id)] = history

    return df.apply(lambda row:
                    f"[HISTORY] {turn_histories[(row['conversation_id'], row['full_turn_id'])]} " if history_size > 0 else "" +
                    f"[CURRENT] [{row['category']}] "
                    f"[TEXT] {row['text']}",
            axis=1)

def build_time_input(df, category_type, level_type, history_size=5):
    """
        Build the turns with the extra context to feed the time classification model.

        :param df: Dataframe
        :param category_type: String with either cat_gold or cat_pred
        :param level_type: String with either lev_gold or lev_pred
        :param history_size: Integer with the size of the history (default=5)
    """
    turn_pairs = []
    categories = category_type
    levels = level_type

    for row in df.itertuples():
        pairs = " ".join(f"[{cat}] {level if level is not None or level != '[None]' or level != -1 else 'None'}"
                         for cat, level in zip(row.categories, row.levels))    
        turn_pairs.append(pairs)

    df = df.copy()
    df["pair_text"] = turn_pairs

    histories  = []

    for conv_id, conv_df in df.groupby("conversation_id", sort=False):
        history = []

        for row in conv_df.itertuples():
            histories.append(" [SEP] ".join(history[-history_size:]))
            history.append(row.pair_text)

    df["history"] = histories

    return  ("[HISTORY] " + df["history"] 
            + " [CURRENT] " + df['pair_text']
            + " [TEXT] " + df['text'].astype(str))
         
                
def get_true_labels(df):
    """
    Get the true labels for the level model.
    """
    return [-1 if x is None
    else -1 if isinstance(x, list) and (len(x) == 0 or x[0] is None or x[0] == "None")
    else x[0] if isinstance(x,list)
    else x 
        for x in df["level"]]

def regroup_df(df):
    final_df = (df.groupby("full_turn_id").agg({"conversation_id": "first",
                                                "speaker": "first",
                                                "text": "first",
                                                "categories": "first",
                                                "levels": "first",
                                                "relative_time": "first",
                                                "predicted_categories": "first",
                                                #"category": list,
                                                "predicted_level": list,
                                                #"level": list,
                                                "combined_text": "first",
                                                "segment_id": "first",
                                                "segment_category": "first"
                                                }).reset_index())
    return final_df

def has_no_labels(row):
    return (all(cat == ["None"] for cat in row["categories"])
                and all(level == ["None"] for level in row["levels"]))


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
        gold_labels = [x.strip("'") for x in row[gold_col] if x.strip("'") not in [None, "None"]]
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

def category_predictions(test_df, tokeniser, trainer, mlb, model_name, data, filtering):
    """
    Predict the categories for the test data and evaluate the results. 
    """
    # Create Hugging Face dataset for the category predictions
    categories_test_dataset = Dataset.from_pandas(test_df.drop(columns=["categories", "levels"]))
    categories_test_dataset = categories_test_dataset.map(tokenise_categories_function, batched=True, fn_kwargs={"tokeniser": tokeniser})

    # Run category predictions
    predicted_categories = run_multilabel_predictions(trainer, categories_test_dataset)
    test_df["one_hot_preds"] = predicted_categories.tolist()
    test_df["predicted_categories"] = list(mlb.inverse_transform(predicted_categories))
   
    # Replace None with "None" to ensure it is included in the classes
    test_df["categories"] = test_df["categories"].apply(lambda cats: ["None" if x is None else x for x in cats])
    test_df["one_hot_true"] = list(mlb.transform(test_df["categories"]))
    
    print(f"Results strict evaluation categories for {model_name} on {data}.")
    strict_report_df, strict_df = eval.eval_strict(test_df, "categories", "predicted_categories", mlb)
    strict_report_df.to_csv(f"./cr_results/cat_{model_name}_{data}_filtering{filtering}_evalstrict", index=False)
    draw_cm_strict(strict_df, category_names, model_name, data, "categories", "predicted_categories")

    print(f"Results lenient evaluation categories for {model_name} on {data}.")
    lenient_report_df, lenient_df = eval.eval_lenient(test_df, "categories", "predicted_categories", mlb)
    lenient_report_df.to_csv(f"./cr_results/cat_{model_name}_{data}_filtering{filtering}_evallenient", index=False)
    draw_cm_lenient(lenient_df, category_names, model_name, data, "categories", "predicted_categories")

    return test_df

def level_predictions(test_df, tokeniser, trainer, history_size, model_name, data, filtering):
    """
    Predict the levels for the test data and evaluate the results. 
    """
    base_df = test_df.copy()

    configs = {
        "catpred": ("predicted_category", "predicted_categories"),
        "catgold": ("category", "categories"),
    }

    results = {}

    for name, (cat_col_sn, cat_col_mp) in configs.items():
        df = extend_df(base_df, cat_col_mp)

        df["combined_text"] = build_levels_input(df, cat_col_sn, history_size)

        dataset = Dataset.from_pandas(df[["combined_text"]])

        dataset = dataset.map(tokenise_levels_function, batched=True, fn_kwargs={"tokeniser": tokeniser})

        predictions = trainer.predict(dataset).predictions.squeeze()

        df["predicted_level"] = predictions        
        
        results_strict = eval.level_eval_strict(df)
        results_lenient = eval.level_eval_lenient(df)
        
        print(f"Results {filtering} filtering and strict evaluation levels ({name})")
        print(results_strict)
        print(f"Results {filtering} filtering and lenient evaluation levels ({name})")
        print(results_lenient)
    
        results[name] = regroup_df(df)
        
        results[name].to_csv(f"{model_name}_results/{data}/results_{name}_filtering{filtering}.csv", index=False)

    return results["catpred"], results["catgold"]


def time_predictions(test_df, tokeniser, trainer, mlb, model_name, data, filtering, history_size):
    """
    Predict the time periods for the test data and evaluate the results. 
    """
    configs = {
        "catpred_levpred": ("predicted_categories", "predicted_level"),
        "catpred_levgold": ("predicted_categories", "levels"),
        "catgold_levpred": ("categories", "predicted_level"),
        "catgold_levgold": ("categories", "levels"),
    }

    results = {}

    for name, (cat_col, lev_col) in configs.items():
        df = test_df.copy()
        
        # Replace None with "None" to ensure it is included in the classes
        df["relative_time"] = df["relative_time"].apply(lambda t: ["None" if x is None else x for x in t] if isinstance(t,list) else t)
        
        df["double_combined_text"] = build_time_input(df, cat_col, lev_col, history_size)

        df = df[~df.apply(has_no_labels, axis=1)].copy()

        dataset = Dataset.from_pandas(df.drop(columns=["relative_time", "levels"]))

        dataset = dataset.map(tokenise_time_function, batched=True, fn_kwargs={"tokeniser": tokeniser})

        preds = run_multilabel_predictions(trainer, dataset)

        df["one_hot_preds_time"] = preds.tolist()

        df["predicted_time"] = list(mlb.inverse_transform(preds.numpy()))
       
        # Replace None with "None" to ensure it is included in the classes
        df["relative_time"] = df["relative_time"].apply(lambda t: ["None" if x is None else x for x in t])

        df["one_hot_true_time"] = list(mlb.transform(df["relative_time"]))

        results[name] = {"df": df, "preds": preds}

    base_df = results["catpred_levpred"]["df"]
    time_true = np.array(base_df["one_hot_true_time"].tolist()) #mlb.transform(base_df["relative_time"])

    for name, result in results.items():
    
        print(f"Results strict evaluation relative time ({name}) for {model_name} on {data}.")
        strict_report_df, strict_df = eval.eval_strict(df, "relative_time", "predicted_time", mlb)
        strict_report_df.to_csv(f"./cr_results/time_{model_name}_{data}_filtering{filtering}_evalstrict", index=False)
        draw_cm_strict(strict_df, time_names, model_name, data, "relative_time", "predicted_time")
    
        print(f"Results lenient evaluation relative time ({name}) for {model_name} on {data}.")
        lenient_report_df, lenient_df = eval.eval_lenient(df, "relative_time", "predicted_time", mlb)
        lenient_report_df.to_csv(f"./cr_results/time_{model_name}_{data}_filtering{filtering}_evallenient", index=False)
        draw_cm_lenient(lenient_df, time_names, model_name, data, "relative_time", "predicted_time")
        
        strict_df.to_csv(f"{model_name}_results/{data}/result_time_{name}_filtering{filtering}_evalstrict.csv", index=False)
        lenient_df.to_csv(f"{model_name}_results/{data}/results_time_{name}_filtering{filtering}_evallenient.csv", index=False)
        
    #return results["catpred_levpred"]["df"], results["catpred_levgold"]["df"], results["catgold_levpred"]["df"], results["catgold_levgold"]["df"]

def main(
        model_name = "medRoBERTa",
        data = "alldata",
        history_size = 5
    ):
    categories_tokeniser = AutoTokenizer.from_pretrained(f"path_to_model_categories") 
    categories_model = AutoModelForSequenceClassification.from_pretrained(f"path_to_model_categories", num_labels=len(category_names))    
    categories_trainer = Trainer(model=categories_model)
    #level_tokeniser = AutoTokenizer.from_pretrained(f"path_to_model_levels")
    #level_model = AutoModelForSequenceClassification.from_pretrained(f"path_to_model_levels", num_labels=1)
    level_tokeniser = AutoTokenizer.from_pretrained(f"path_to_model_levels")
    level_model = AutoModelForSequenceClassification.from_pretrained(f"path_to_model_levels", num_labels=1)
    level_trainer = Trainer(model=level_model)
    if data == "convdata":
          time_tokeniser = AutoTokenizer.from_pretrained(f"path_to_model_time")
          time_model = AutoModelForSequenceClassification.from_pretrained(f"path_to_model_time", num_labels=len(time_names))
          time_trainer = Trainer(model=time_model)
          
    input_dir_test = r"prepared_test_data"
    test_df_strict, test_df_lenient = eval.combine_df(input_dir_test)
    
    test_df_strict.to_csv("./filtered_test_data_all_strict.csv", index=False)
    test_df_strict = test_df_strict[["conversation_id", "turn", "speaker", "text", "categories", "levels", "relative_time", "segment_id", "segment_category"]]
    # Drop rows with NaN for text
    test_df_strict = test_df_strict.dropna(subset=["text"])
    
    test_df_lenient.to_csv("./filtered_test_data_all_lenient.csv", index=False)
    test_df_lenient = test_df_lenient[["conversation_id", "turn", "speaker", "text", "categories", "levels", "relative_time", "segment_id", "segment_category"]]
    # Drop rows with NaN for text
    test_df_lenient = test_df_lenient.dropna(subset=["text"])

    categories_mlb = MultiLabelBinarizer(classes=category_names)
    categories_mlb.fit([category_names])
    time_mlb = MultiLabelBinarizer(classes=time_names)
    time_mlb.fit([time_names])
    
    for test_df, filtering in zip([test_df_strict, test_df_lenient], ["strict", "lenient"]):
      categories_test_df = category_predictions(test_df, categories_tokeniser, categories_trainer, categories_mlb, model_name, data, filtering)
  
      level_test_df_catpred, level_test_df_catgold = level_predictions(categories_test_df, level_tokeniser, level_trainer, history_size, model_name, data, filtering)
      
      if data == "convdata":          
          time_predictions(level_test_df_catpred, time_tokeniser, time_trainer, time_mlb, model_name, data, filtering, history_size)             
  
if __name__ == "__main__":
    main()

    