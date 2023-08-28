"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
input_file_path2 = os.path.join(os.path.dirname(__file__), 'input2.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
if not os.path.exists(input_file_path2):
    data_url2 = 'https://www.gutenberg.org/cache/epub/71496/pg71496.txt'
    with open(input_file_path2, 'w') as f2:
        f2.write(requests.get(data_url2).text)

with open(input_file_path, 'r') as f:
    data = f.read()
with open(input_file_path2, 'r') as f2:
    data2 = f2.read()
print(f"length of shakespeare dataset in characters: {len(data):,}")
print(f"length of non-shakespeare dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")


# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# get all the unique characters that occur in this text
chars2 = sorted(list(set(data2)))
vocab_size2 = len(chars2)
print("all the unique characters:", ''.join(chars2))
print(f"vocab size: {vocab_size2:,}")

chars = set(chars).union(set(chars2))

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print("data len: " + str(len(data)))
print("data[:5]: " + str(data[:50]))

train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
train_data2 = data2[:int(n2*0.9)]
val_data2 = data2[int(n2*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
train_ids2 = encode(train_data2)
val_ids2 = encode(val_data2)
print(f"train2 has {len(train_ids2):,} tokens")
print(f"val2 has {len(val_ids2):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# export to bin files
train_ids2 = np.array(train_ids2, dtype=np.uint16)
val_ids2 = np.array(val_ids2, dtype=np.uint16)
train_ids2.tofile(os.path.join(os.path.dirname(__file__), 'train2.bin'))
val_ids2.tofile(os.path.join(os.path.dirname(__file__), 'val2.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

