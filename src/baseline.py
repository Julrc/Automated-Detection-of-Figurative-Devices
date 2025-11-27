#%%
import pandas as pd
from utils.split_data import ReadTSV
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from datasets import Dataset
#%%
annotation_path = '../data/nlp_annots.tsv'
data = ReadTSV(annotation_path)

training_df, test_df= data.split_data()
#%%
model_checkpoint="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#%%
# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

train_dataset = Dataset.from_pandas(training_df)
eval_dataset = Dataset.from_pandas(test_df)

TARGET_COLUMN = "device"

train_dataset = train_dataset.class_encode_column(TARGET_COLUMN)
eval_dataset = eval_dataset.class_encode_column(TARGET_COLUMN)

train_dataset = train_dataset.rename_column(TARGET_COLUMN, "labels")
eval_dataset = eval_dataset.rename_column(TARGET_COLUMN, "labels")

#%%
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

columns_to_return = ["input_ids", "token_type_ids", "attention_mask", "labels"]
train_dataset.set_format(type="torch", columns=columns_to_return)
eval_dataset.set_format(type="torch", columns=columns_to_return)

# %%
num_labels = train_dataset.features["labels"].num_classes
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, problem_type="single_label_classification")

# %%

for param in model.base_model.parameters():
    param.requires_grad = False

for param in model.bert.pooler.parameters():
    param.requires_grad = True

# %%

def count_parameters(model):
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "Total": total_params,
        "Trainable": trainable_params,
        "Frozen": total_params - trainable_params,
    }

param_counts = count_parameters(model)
print("Model Parameter Counts: ")
for k, v in param_counts.items():
    print(f"{k}: {v:,}")
# %%

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

#%%

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

#%%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

#%%
trainer.train()
#%%
