import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pickle
import os
import numpy as np
"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""
from embedding_factory import get_embedding
class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension,embedding_type):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = get_embedding(emb_size, emb_dimension) #, sparse=True
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension) # , sparse=True

        initrange = 1.0 / self.emb_dimension
        if type(self.u_embeddings) == nn.Embedding:
            init.uniform_(self.u_embeddings.weight.data,    -initrange, initrange)
        else:
            print("any init needed? !!!")
            pass
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward_embedding(self,pos_u,time=None):
        emb_u = self.u_embeddings(pos_u)
        return emb_u

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, path):
        if not os.path.exists(path):
            os.mkdir(path)
        file_name = os.path.join(path,"vectors.txt")
        if type(self.u_embeddings) == nn.Embedding:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.get_weights().cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
        pickle.dump( id2word,open("{}/dict.pkl".format(path),"wb"))

    def load_embeddings(self, id2word, path):
        
        file_name = os.path.join(path,"vectors.txt")
        word2id = {value:key for key,value in id2word.items()}


        with open(file_name) as f:
            for i,line in enumerate(f):
                if i==0:
                    vocab_size,emb_dimension = [int(item) for item in line.split()]
                    embeddings= np.zeros([vocab_size,emb_dimension])
                else:
                    tokens = line.split()
                    word, vector = tokens[0], [float(num_str) for num_str in tokens[1:]]
                    embeddings[word2id[word]]=vector
        self.u_embeddings.weight = torch.nn.Parameter(torch.from_numpy(embeddings).float())
        other_embeddings_pkl = "{}/para.pkl".format(path)

        

