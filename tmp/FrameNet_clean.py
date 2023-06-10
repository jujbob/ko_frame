import os
from datasets import load_dataset, Dataset, ClassLabel
import pandas as pd
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

def set_environment_variables():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_raw_datasets():
    return load_dataset("conll2003")

def process_data(file_path):
    label_set = set()
    tokens = []
    pos_tags = []
    token = []
    pos_tag = []
    with open(file_path) as file:
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
    return tokens, pos_tags, list(label_set)

def create_label_dicts(label_list):
    label_encoding_dict = {label_list[i]: i for i in range(len(label_list))}
    index_encdoing_dict = {i: label_list[i] for i in range(len(label_list))}
    return label_encoding_dict, index_encdoing_dict

def prepare_dataset(tokens, pos_tags, label_list):
    train_df = pd.DataFrame({'tokens': tokens, 'ner_tags': pos_tags})
    ClassLabels = ClassLabel(num_classes=len(label_list), names=label_list)

    def map_label2id(example):
        example['ner_tags'] = [ClassLabels.str2int(label) for label in example['ner_tags']]
        return example

    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(map_label2id, batched=True)
    train_dataset.features['ner_tags'] = ClassLabels
    return train_dataset

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = examples["ner_tags"]
    new_labels = [align_labels_with_tokens(labels, tokenized_inputs.word_ids(i)) for i, labels in enumerate(all_labels)]
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def map_datasets(raw_datasets, tokenizer):
    return raw_datasets.map(
        lambda examples: tokenize_and_align_labels(tokenizer, examples),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

def compute_metrics(eval_preds, label_names):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    all_metrics = evaluate.load("seqeval").compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def create_training_args():
    return TrainingArguments(
        "bert-finetuned-ner-ime",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
        )

def create_model(model_checkpoint, id2label, label2id):
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

def train_model(model, tokenized_datasets, data_collator, compute_metrics, tokenizer, args):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    return trainer.train()

def main():
    set_environment_variables()
    raw_datasets = load_raw_datasets()
    tokens, pos_tags, label_list = process_data('data/pkfn_clean_accumulated.tsv')
    label_encoding_dict, index_encdoing_dict = create_label_dicts(label_list)
    train_dataset = prepare_dataset(tokens, pos_tags, label_list)
    ner_feature = raw_datasets["train"].features["ner_tags"]
    ner_feature.feature.names=index_encdoing_dict
    label_names = ner_feature.feature.names
    model_checkpoint = "snunlp/KR-BERT-char16424"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_datasets = map_datasets(raw_datasets, tokenizer)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics_func = lambda eval_preds: compute_metrics(eval_preds, label_names)
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    args = create_training_args()
    model = create_model(model_checkpoint, id2label, label2id)
    train_model(model, tokenized_datasets, data_collator, compute_metrics_func, tokenizer, args)

if __name__ == "__main__":
    main()