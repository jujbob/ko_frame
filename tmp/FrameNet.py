import os
from datasets import load_dataset
from datasets import Dataset, Value, ClassLabel, Features
import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



raw_datasets = load_dataset("conll2003")

###
label_set = set()
tokens = []
pos_tags = []
token = []
pos_tag = []
with open('data/pkfn_clean_accumulated.tsv') as file:
  for line in file:
    line = line.strip()    
    if len(line) == 0:
      tokens.append(token)
      pos_tags.append(pos_tag)
      token = []
      pos_tag = []
    elif line.startswith("#"):
      continue
    else:
      words = line.split("\t")
      label_set.add(words[4])
      token.append(words[1])
      pos_tag.append(words[4])

label_list = list(label_set)

label_encoding_dict = {}
index_encdoing_dict = {}
for i in range(len(label_list)):
  label_encoding_dict[label_list[i]] = i
  index_encdoing_dict[i] = label_list[i]

train_df = pd.DataFrame({'tokens':tokens, 'ner_tags': pos_tags})
ClassLabels = ClassLabel(num_classes=len(label_list), names=label_list)
def map_label2id(example):
    label_list = []
    for label in example['ner_tags']:
        label_list.append(ClassLabels.str2int(label))
        
    example['ner_tags'] = label_list
    return example

train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(map_label2id, batched=True)
train_dataset.features['ner_tags'] = ClassLabels
ner_feature = raw_datasets["train"].features["ner_tags"]
ner_feature.feature.names=index_encdoing_dict
label_names = ner_feature.feature.names

model_checkpoint = "snunlp/KR-BERT-char16424"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)



data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metric = evaluate.load("seqeval")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

args = TrainingArguments(
    "bert-finetuned-ner-ime",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)



model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
###
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()