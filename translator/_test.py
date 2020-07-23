import torch
from torch.autograd import Variable
import numpy as np
from transformer import SelfAttentionMask

if 0:
  device = torch.device('cuda', 0)
  selfattnmask = SelfAttentionMask(device)
  #print (selfattnmask.weights)

def subsequent_mask(size):
  "Mask out subsequent positions."
  attn_shape = (1, size, size)
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  # return subsequent_mask == 0
  return torch.from_numpy(subsequent_mask) == 0

if 0:
  print (subsequent_mask(10)[0])

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

V = 11
for batch in data_gen(V, 30, 20):
  print (batch.src)
  print(batch.src_mask)
  print (batch.trg)
  print(batch.trg_mask)

  print (batch.src)
  print (batch.trg)
  print (batch.src.size())
  print (batch.src_mask.size())
  print (batch.trg.size())
  print (batch.trg_mask.size())
  
  mask = batch.trg_mask
  scores = torch.from_numpy(np.random.randint(1, V, size=(30, 9, 9)))
  print (scores)
  print (scores.size())
  scores = scores.masked_fill(mask == 0, -1e3)
  print (scores)
 
  assert False