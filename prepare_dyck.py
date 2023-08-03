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
import pandas as pd
import torch
# download the dyck dataset
test_path1 =  '~/Downloads/Dataset/data/dyck-clean-val-rep.txt'
test_path2 = '~/Downloads/Dataset/data/dyck-corrupted-val-rep.txt'
train_path1 =  '~/Downloads/Dataset/data/dyck-clean-train-rep.txt'
train_path2 =  '~/Downloads/Dataset/data/dyck-corrupted-train-rep.txt'
MAX_LEN = 128
chars = ["#","(",")","{","}"] # this needs to be replaced with a set function
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

train_dat1 = pd.read_csv(test_path1,header=None)
train_dat2 = pd.read_csv(test_path2,header=None)
val_dat1 = pd.read_csv(train_path1,header=None)
val_dat2 = pd.read_csv(train_path2,header=None)
train_dat1[1] = 1
train_dat2[1] = 0
val_dat1[1] = 1
val_dat2[1]=0


train_dat = pd.concat([train_dat1,train_dat2])
val_dat = pd.concat([val_dat1,val_dat2])

train_n = len(train_dat)
val_n = len(val_dat)

#strip off spaces and END
def stripApply(x):
    return x.replace(" ","")
def replaceEnd(x):
    return x.replace("END","")

train_dat[0] = train_dat[0].apply(replaceEnd)
val_dat[0] = val_dat[0].apply(replaceEnd)
train_dat[0] = train_dat[0].apply(stripApply)
val_dat[0] = val_dat[0].apply(stripApply)

#get max lengths
train_dat["length"] = train_dat.apply(lambda x: len(x[0]),axis = 1)
val_dat["length"] = val_dat.apply(lambda x: len(x[0]),axis = 1)
max_train_len = np.max(train_dat["length"])
max_val_len = np.max(val_dat["length"])

print("max train len: " + str(max_train_len))
print("max val len: " + str(max_val_len))

train_dat[0] = train_dat[0].str.pad(width=max_train_len+1,side = "left",fillchar="#")
val_dat[0] = val_dat[0].str.pad(width=max_train_len+1,side = "left",fillchar="#")

print("train_dat: " + str(train_dat[:100]))
#get max lengths
train_dat["length"] = train_dat.apply(lambda x: len(x[0]),axis = 1)
val_dat["length"] = val_dat.apply(lambda x: len(x[0]),axis = 1)
max_train_len = np.max(train_dat["length"])
max_val_len = np.max(val_dat["length"])

print("max train len: " + str(max_train_len))
print("max val len: " + str(max_val_len))



#prepare to convert to tensor
train_dat[0]=train_dat[0].apply(encode)
train_dat[0]=train_dat[0].apply(np.array)
val_dat[0]=val_dat[0].apply(encode)
val_dat[0]=val_dat[0].apply(np.array)




val_x  = np.concatenate( val_dat[0], axis=0 )
val_n_tot = len(val_x)
val_x = tprcj/temspr(val_x.reshape(val_n,int(val_n_tot/val_n)))
val_y = torch.tensor(val_dat[1].to_numpy())
torch.save(val_x,"val_x.pt")
torch.save(val_y,"val_y.pt")

train_x  = np.concatenate( train_dat[0], axis=0 )
train_n_tot = len(train_x)
train_x = torch.tensor(train_x.reshape(train_n,int(train_n_tot/train_n)))
train_y = torch.tensor(train_dat[1].to_numpy())
torch.save(train_x,"train_x.pt")
torch.save(train_y,"train_y.pt")

