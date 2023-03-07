#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import datetime

from transformers import ElectraTokenizer

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--predictions_save_path', type=str, help='Load path to which predicted values')

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

    # Choose device
    device = get_default_device()

    with open(args.test_data_path) as f:
        test_data = [json.loads(line) for line in f]

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)
    
    input_ids = []
    input_att_msks = []

    for count, item in enumerate(test_data):
        print(count, len(test_data))
        context = item["context"]
        candidate = item["candidate"]
        reference = item["reference"]
        question = item["question"]
        combo = candidate + " [SEP] " + reference + " [SEP] " + question
        input_encodings_dict = tokenizer(combo, truncation=True, max_length=512, padding="max_length")
        input_ids.append(input_encodings_dict['input_ids'])
        input_att_msks.append(input_encodings_dict['attention_mask'])

    # Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    input_att_msks = torch.tensor(input_att_msks)
    input_att_msks = input_att_msks.long().to(device)

    ds = TensorDataset(input_ids, input_att_msks)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    logits = []
    count = 0
    for inp_id, att_msk in dl:
        print(count)
        count+=1
        inp_id, att_msk = inp_id.to(device), att_msk.to(device)
        with torch.no_grad():
            outputs = model(input_ids=inp_id, attention_mask=att_msk)
        curr_logits = outputs[0]
        logits += curr_logits.detach().cpu().numpy().tolist()
    logits = np.asarray(logits)
    np.save(args.predictions_save_path + "logits_all.npy", logits)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)