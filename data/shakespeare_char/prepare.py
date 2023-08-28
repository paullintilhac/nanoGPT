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

train_data = encode(train_data)


print("class train_x: " + str(type(train_dat)))
print("class of each col: " + str(type(train_dat[0])))
print("dim(valDat)")
val_x = np.concatenate(val_dat[0].values,axis = 0)
print("val_x shape after concat: " + str(val_x.shape))
#val_x  = np.concatenate( val_dat[0], axis=0 )
val_n_tot = len(val_x)
# val_x = torch.tensor(val_x)
val_x = torch.tensor(val_x.reshape(val_n,int(val_n_tot/val_n)))

val_y = torch.tensor(val_dat[1].to_numpy())
print("val x shape: " + str(val_x.shape))
print("val y shape: " + str(val_y.shape))

torch.save(val_x,"val_x.pt")
torch.save(val_y,"val_y.pt")

train_x  = np.concatenate( train_dat[0].values, axis=0 )
train_n_tot = len(train_x)
train_x = torch.tensor(train_x.reshape(train_n,int(train_n_tot/train_n)))
train_y = torch.tensor(train_dat[1].to_numpy())

train_x = train_x[:50000]
train_y = train_y[:50000]
val_x = val_x[:10000]
val_y = val_y[:10000]

print("train x shape: " + str(train_x.shape))
print("train y shape: " + str(train_y.shape))
torch.save(train_x,"train_x.pt")
torch.save(train_y,"train_y.pt")

