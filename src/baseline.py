#%%
import pandas as pd
from utils.clean_data import ReadTSV
from utils.training import count_parameters
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from datasets import Dataset
import torch as t
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

#%%
if (t.cuda.is_available()):
    device= "cuda"
else:
    device = "cpu"

#%%
annotation_path = '../data/nlp_annots.tsv'
data = ReadTSV(annotation_path)

data_df = data.clean_data()
#%% Select task: metaphor, simile, alliteration

task = "metaphor"
data_df.loc[data_df['device'] != task, 'device']= 'literal'

#%%

labels_list = data_df['device']
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

#%% Loop, K fold cross validation

for i, (train_index, test_index) in enumerate(skf.split(data_df, labels_list)):

    train_fold = data_df.iloc[train_index]
    val_fold = data_df.iloc[test_index]

    model_checkpoint="bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    train_dataset = Dataset.from_pandas(train_fold)
    eval_dataset = Dataset.from_pandas(val_fold)
    TARGET_COLUMN = "device"

    train_dataset = train_dataset.class_encode_column(TARGET_COLUMN)
    eval_dataset = eval_dataset.class_encode_column(TARGET_COLUMN)

    train_dataset = train_dataset.rename_column(TARGET_COLUMN, "labels")
    eval_dataset = eval_dataset.rename_column(TARGET_COLUMN, "labels")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    train_dataset = train_dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), 
        batched=True)

    eval_dataset = eval_dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), 
        batched=True)

    columns_to_return = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns_to_return)
    eval_dataset.set_format(type="torch", columns=columns_to_return)

    targets = np.array(train_dataset["labels"])
    unique_labels, counts = np.unique(targets, return_counts=True)
    total = len(targets)
    class_weights = (1.0 / counts)

    sample_weights = []
    for tar in targets:
        sample_weights.append(class_weights[tar])

    weights_tensor = t.DoubleTensor(sample_weights)

    gen = t.Generator()
    gen.manual_seed(42)

    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(train_dataset), replacement=True, generator=gen)

    num_labels = train_dataset.features["labels"].num_classes
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, problem_type="single_label_classification")

    for param in model.base_model.parameters():
        param.requires_grad = False

    for param in model.bert.pooler.parameters():
        param.requires_grad = True

    param_counts = count_parameters(model)
    print("Model Parameter Counts: ")
    for k, v in param_counts.items():
        print(f"{k}: {v:,}")

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, shuffle=False)
    eval_loader=DataLoader(eval_dataset, batch_size=16)

    model.to(device)
    model.train()
    optimizer=AdamW(model.parameters(), lr=2e-5)

    for epoch in range(7):
        for batch in train_loader:

            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}")
        model.eval()
        all_preds = []
        all_labels = []
        with t.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = t.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        print(classification_report(all_labels, all_preds, target_names=["literal", "metaphor"]))

        t.cuda.empty_cache()

#%%