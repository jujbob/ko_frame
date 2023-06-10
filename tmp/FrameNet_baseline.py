#!/usr/bin/env python
# coding: utf-8

# # Token classification (PyTorch)
#  - From : https://huggingface.co/learn/nlp-course/chapter7/2

# Install the Transformers, Datasets, and Evaluate libraries to run this notebook.

# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[3]:


#get_ipython().system('pip install datasets evaluate transformers[sentencepiece]')
#get_ipython().system('pip install accelerate')
# To run the training on TPU, you will need to uncomment the following line:
# !pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
#get_ipython().system('apt install git-lfs')


# You will need to setup git, adapt your email and name in the following cell.

# In[4]:


#get_ipython().system('git config --global user.email "jujbob@gmail.com"')
#get_ipython().system('git config --global user.name "jujbob"')


# You will also need to be logged in to the Hugging Face Hub. Execute the following and enter your credentials.

# In[5]:


from huggingface_hub import notebook_login
#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
notebook_login()


# In[6]:


from datasets import load_dataset
from datasets import Dataset, Value, ClassLabel, Features
import pandas as pd

raw_datasets = load_dataset("conll2003")


# In[7]:


raw_datasets


# In[10]:


raw_datasets["train"]


# In[9]:


raw_datasets["train"]["pos_tags"]


# In[8]:


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


# In[9]:


index_encdoing_dict


# In[10]:


list(train_df.iloc[:7]["tokens"])


# In[11]:


train_df


# In[12]:


train_dataset = Dataset.from_pandas(train_df)


# In[13]:


train_dataset


# In[14]:


ClassLabels = ClassLabel(num_classes=len(label_list), names=label_list)


# In[15]:


ClassLabels


# In[16]:


def map_label2id(example):
    label_list = []
    for label in example['ner_tags']:
        label_list.append(ClassLabels.str2int(label))
        
    example['ner_tags'] = label_list
    return example


# In[17]:


train_dataset = train_dataset.map(map_label2id, batched=True)


# In[18]:


train_dataset["ner_tags"]


# In[19]:


train_dataset.features['ner_tags'] = ClassLabels


# In[20]:


ClassLabels


# In[21]:


raw_datasets["train"] = train_dataset


# In[22]:


raw_datasets["train"].features


# In[23]:


raw_datasets["train"][0]


# In[24]:


raw_datasets["train"][0]["tokens"]


# In[25]:


raw_datasets["train"][0]["ner_tags"]


# In[26]:


ner_feature = raw_datasets["train"].features["ner_tags"]
ner_feature


# In[27]:


ner_feature.feature


# In[28]:


ner_feature.feature.names=index_encdoing_dict


# In[29]:


label_names = ner_feature.feature.names
label_names


# In[30]:


words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1, len(line1))
print(line2, len(line2))


# In[31]:


from transformers import AutoTokenizer

model_checkpoint = "snunlp/KR-BERT-char16424"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[32]:


tokenizer.is_fast


# In[33]:


inputs = tokenizer(train_dataset[0]["tokens"], is_split_into_words=True)
inputs.tokens()


# In[34]:


inputs.word_ids()


# In[35]:


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


# In[36]:


labels = train_dataset[0]["ner_tags"]
print(labels)


# In[37]:


word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))


# In[38]:


tmp = align_labels_with_tokens(labels, word_ids)
tmp


# In[39]:


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


# In[40]:


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)


# In[41]:


raw_datasets


# In[42]:


tokenized_datasets


# In[43]:


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# In[44]:


batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
batch["labels"]


# In[45]:


#get_ipython().system('pip install seqeval')


# In[46]:


import evaluate

metric = evaluate.load("seqeval")


# In[47]:


labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
labels


# In[48]:


predictions = labels.copy()
predictions[2] = "O"
metric.compute(predictions=[predictions], references=[labels])


# In[49]:


import numpy as np


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


# In[50]:


id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


# In[51]:


print(len(id2label), len(label2id))


# In[52]:


from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)


# In[53]:


model.config.num_labels


# In[55]:


from huggingface_hub import notebook_login
#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
#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
notebook_login()


# In[56]:


from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner-ime",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)


# In[57]:


from transformers import Trainer

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


# In[ ]:


trainer.push_to_hub(commit_message="Training complete by jujbob!!")


# In[ ]:


tokenized_datasets["train"][0]


# In[ ]:


from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)


# In[ ]:


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)


# In[ ]:


from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)


# In[ ]:


from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


# In[ ]:


from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# In[ ]:


from huggingface_hub import Repository, get_full_repo_name

model_name = "bert-finetuned-ner-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name


# In[ ]:


output_dir = "bert-finetuned-ner-accelerate"
repo = Repository(output_dir, clone_from=repo_name)


# In[ ]:


def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


# In[ ]:


from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )


# In[ ]:


accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)


# In[ ]:


from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

