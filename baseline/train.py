#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=4, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-6, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=2, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--train_data_path', type=str, help='Load path of training data')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    with open(args.train_data_path) as f:
        train_data = [json.loads(line) for line in f]

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    labels = []
    input_ids = []
    input_att_msks = []

    for count, item in enumerate(train_data):
        print(count, len(train_data))
        context = item["context"]
        candidate = item["candidate"]
        reference = item["reference"]
        question = item["question"]
        label = int(item["score"])

        combo = candidate + " [SEP] " + reference + " [SEP] " + question
        input_encodings_dict = tokenizer(combo, truncation=True, max_length=512, padding="max_length")
        input_ids.append(input_encodings_dict['input_ids'])
        input_att_msks.append(input_encodings_dict['attention_mask'])
        labels.append(label)


    # Convert to torch tensors
    labels = torch.tensor(labels)
    labels = labels.long().to(device)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    input_att_msks = torch.tensor(input_att_msks)
    input_att_msks = input_att_msks.long().to(device)

    # Create the DataLoader for training set.
    train_data = TensorDataset(input_ids, input_att_msks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model = ElectraForSequenceClassification.from_pretrained(electra_large).to(device)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    # weight_decay = 0.01
                    )

    loss_values = []

    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0.1*total_steps,
                                                num_training_steps = total_steps)


    for epoch in range(args.n_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_att_msks = batch[1].to(device)
            b_labs = batch[2].to(device)
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, labels=b_labs)
            loss = outputs[0]
            total_loss += loss.item()
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    file_path = args.save_path+'electra_AE_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)