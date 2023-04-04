import yaml
from pathlib import Path
import torch
import pandas as pd

RELATION_FORM_MAP = {
    "Member Of:P463": "\"{}\" is a member of \"{}\"",
    "Owned By:P127":  "\"{}\" is owned by \"{}\"",
    "Located In:P463":  "\"{}\" is located in \"{}\"",
    # "Candidate:P726":  "\"{}\" is a candidate for \"{}\"",
}

DEVICE = torch.device("cuda")


def get_current_schema_versions(d):
    most_recent_map = {}
    for filename in d.iterdir():
        filename = str(filename)
        if filename.startswith("ce"):
            continue
        with open(filename) as f:
            schema = yaml.safe_load(f)
        schema_name = schema["name"].lower()
        creation_date = schema["creation_date"]
        if schema_name in most_recent_map:
            stored_filename, stored_schema, stored_creation_date = most_recent_map[schema_name]
            if stored_creation_date < creation_date:
                print(f"Replacing {stored_filename} with newer {filename}")
                most_recent_map[schema_name] = (filename, schema, creation_date)
        else:
            most_recent_map[schema_name] = (filename, schema, creation_date)
    most_recent_map = {k: most_recent_map[k] for k in sorted(most_recent_map.keys())}
    current_schemas = [v[1] for v in most_recent_map.values()]
    return current_schemas


def get_event_arg_pairs(schema):
    event_arg_pairs = []
    for step in schema["steps"]:
        if "subschema" in step:
            continue
        event_text = step["id"]
        args = [arg["refvar"] for arg in step["slots"]]
        # temp solution for initial testing
        if len(args) < 2:
            continue
        arg_pairs = []
        for i in range(len(args) - 1):
            for j in range(i+1, len(args)):
                if args[i] != args[j]:
                    arg_pairs.append((args[i], args[j]))
                    arg_pairs.append((args[j], args[i]))
        sample = {"event": event_text, "arg_pairs": arg_pairs}
        event_arg_pairs.append(sample)
    return event_arg_pairs

current_schemas = get_current_schema_versions(Path("data/submitted_schemas_03_14_23"))
import random
import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
from src.paths import DATA_DIR, MODELS_DIR
from src.models.transformer import Transformer

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenizer_max_length", type=int, default=128)
    parser.add_argument("--model_type", type=str, default="microsoft/deberta-large-mnli")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--valid_interval", type=float, default=1.0)
    parser.add_argument("--use_random_negatives", action="store_true")
    parser.add_argument("--use_hard_negatives", action="store_true")
    parser.add_argument("--gradient_accumulations", default=1, type=int)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    return args

args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_type)
config = AutoConfig.from_pretrained(args.model_type)
state_dict = torch.load(args.checkpoint)["state_dict"]
model = Transformer(args.model_type)
model.load_state_dict(state_dict)
model.to(device=DEVICE)
softmax = torch.nn.Softmax(dim=1)
idx2label = {v: k for k, v in config.label2id.items()}

# random_seed = random.Random(0)
# random_schemas = random_seed.sample(current_schemas, 10)
random_schemas = current_schemas
df_map = {}
for schema in tqdm(random_schemas):

    event_arg_pairs = get_event_arg_pairs(schema)
    outputs = []
    with torch.no_grad():
        for event_arg_pair in event_arg_pairs:
            event_text = event_arg_pair["event"]
            arg_pairs = event_arg_pair["arg_pairs"]
            for arg_pair in arg_pairs:
                for relation, prompt in RELATION_FORM_MAP.items():
                    filled_prompt = prompt.format(*arg_pair)
                    tokenizer_input = [(event_text, filled_prompt)]
                    tokenized_inputs = tokenizer(
                        tokenizer_input,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=128,
                    )
                    tokenized_inputs = {k: v.to(device=DEVICE) for k, v in tokenized_inputs.items()}
                    logits = model(tokenized_inputs).logits.detach().cpu()
                    softmax_logits = softmax(logits)
                    results = [event_text, arg_pair[0], relation, arg_pair[1]] +  softmax_logits.numpy().flatten().tolist()[1:]
                    outputs.append(results)

    columns = ["event", "arg0", "relation", "arg1", "neutral", "entailment"]
    sorted_outputs = [output for output in sorted(outputs, key=lambda x: -1 * x[-1])]
    df = pd.DataFrame(sorted_outputs, columns=columns)
    name = schema["name"]
    name = name if "." not in name else name.split(".", 1)[0]
    df_map[name[:30]] = df


# writer = pd.ExcelWriter("outputs/relation_suggestion_scores.xlsx", engine='xlsxwriter')
# for schema_id, df in df_map.items():
#     df.to_excel(writer, sheet_name=schema_id)
# writer.close()

# export 
import numpy as np
results = [df.to_numpy() for df in df_map.values()]
filtered_results = []
for result in results:
    suggested_relations = set()
    for row in result:
        sentence, arg0, relation, arg1, neutral_score, entailment_score = row
        if "GRAPH" in sentence or "PREDICT" in sentence or "MATCH" in sentence:
            continue
        relation_tuple = (arg0, relation, arg1)
        if relation_tuple in suggested_relations:
            continue
        suggested_relations.add(relation_tuple)
        filtered_results.append(row)

full_array = np.concatenate(filtered_results).reshape(-1, 6)

import random
columns = ["event", "arg0", "relation", "arg1", "neutral", "entailment"]
for num in [10]:
    for i in [0.7, 0.8, 0.9]:
        less_than_array = full_array[full_array[:,-1] < (i + 0.1)]
        filtered_array = less_than_array[less_than_array[:,-1] >= i]
        all_sampled_rows = []
        for relation in RELATION_FORM_MAP.keys():
            random_seed = random.Random(0)
            relation_filtered_array = filtered_array[filtered_array[:,2] == relation]
            this_num = num
            while True:
                try:
                    sampled_rows = random_seed.sample(relation_filtered_array.tolist(), this_num) # 10 samples from each relation
                    break
                except ValueError:
                    this_num -= 1
                    print(f"Trying smaller value than {this_num} for {relation} at threshold {i}")
                    if this_num == 0:
                        break
            for row in sampled_rows:
                all_sampled_rows.append(list(row))
        df_i = pd.DataFrame(all_sampled_rows, columns=columns)
        df_i.to_csv(f"sampled_{i}_{num}.csv", index=False)
