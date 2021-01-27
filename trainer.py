import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SkipGramModel
from data_reader import DataReader, Word2vecDataset

import os
import argparse
import pickle
import numpy as np


parser = argparse.ArgumentParser(description='parameter information')

parser.add_argument('--text', dest='text', type=str,default= "input.txt", help='text dataset')
parser.add_argument('--output', dest='output', default= "output" , type=str, help='output dir to save embeddings')
parser.add_argument('--log_step', dest='log_step', default= 1 , type=int, help='log_step')
parser.add_argument('--from_scatch', dest='from_scatch', default= 1 , type=int, help='from_scatch or not')
parser.add_argument('--batch_size', dest='batch_size', default= 1 , type=int, help='batch_size')
parser.add_argument('--emb_dimension', dest='emb_dimension', default= 100 , type=int, help='emb_dimension')
parser.add_argument('--verbose', dest='verbose', default= 0, type=int, help='verbose')
parser.add_argument('--lr', dest='lr', default= 0.001, type=float, help='learning rate')

args = parser.parse_args()

import numpy as np
import heapq

def keep_top(arr,k=3): 
    smallest = heapq.nlargest(k, arr)[-1]  # find the top 3 and use the smallest as cut off
    arr[arr < smallest] = 0 # replace anything lower than the cut off with 0
    return arr


def read_embeddings_from_file(file_name):
    embedding_dict = dict()
    with open(file_name) as f:
        for i,line in enumerate(f):
            if i==0:
                vocab_size,emb_dimension = [int(item) for item in line.split()]
                # embeddings= np.zeros([vocab_size,emb_dimension])
            else:
                tokens = line.split()
                word, vector = tokens[0], [float(num_str) for num_str in tokens[1:]]
                embedding_dict[word] = vector
    return embedding_dict

class Word2VecChecker:
    def __init__(self,path = "output", model_file_name ="vectors.txt" ):
        # for time_type in os.listdir(path):
        #     if ".DS_Store" in time_type:
                # continue
        # subpath = os.path.join(path,model_file_name)
        self.embedding_dict = read_embeddings_from_file(os.path.join(path,model_file_name)) # repeating loading, maybe reading a config file is better if you saved
   
        self.skip_gram_model = SkipGramModel(len(self.embedding_dict), args.emb_dimension)
        self.id2word = pickle.load(open(os.path.join(path, "dict.pkl"),"rb"))
        self.skip_gram_model.load_embeddings(self.id2word,path)
            

            # print(embeddings)
    def check_word353(self,word_sim_353_text="path for wordsim353"):
        return




class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.01, min_count=25,embedding_type ="default"):

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
    

        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=0, collate_fn=dataset.collate)

        

        self.output_file_name = os.path.join(output_file,embedding_type)
        if not os.path.exists(output_file):
            os.mkdir(output_file)
        if not os.path.exists(self.output_file_name):
            os.mkdir(self.output_file_name)

        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr


        print(args)

        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension,embedding_type)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            print("using cuda and GPU ....")
            self.skip_gram_model.cuda()

        # load_path = "{}/{}".format(self.output_file_name)
        if not args.from_scatch and os.path.exists(self.output_file_name):

            print("loading parameters  ....")
            self.skip_gram_model.load_embeddings(self.data.id2word,self.output_file_name)

    def train(self):
        print(os.path.join(self.output_file_name,"log.txt"))
        with open("{}/log.txt".format(self.output_file_name,"log.txt"),"w") as f: 
            for iteration in range(self.iterations):

                print("Iteration: " + str(iteration + 1))
                # optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
                optimizer = optim.Adam(self.skip_gram_model.parameters(), lr=self.initial_lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

                running_loss = 0.0
                for i, sample_batched in enumerate(tqdm(self.dataloader)):

                    if len(sample_batched[0]) > 1:

                        pos_u = sample_batched[0].to(self.device)
                        pos_v = sample_batched[1].to(self.device)
                        neg_v = sample_batched[2].to(self.device)
                        # print(pos_u.shape)
                        
                        scheduler.step()
                        optimizer.zero_grad()

                        loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                        # print(loss)
                        loss.backward()
                        optimizer.step()

                        running_loss = loss.item() #running_loss * 0.9 + loss.item() * 0.1
                        if  i % args.log_step == 0: # i > 0 and
                            f.write("Loss in {} steps: {}\n".format(i,str(running_loss)))
                        # if i > 1000:
                            if args.verbose:
                                print("Loss in {} steps: {}\n".format(i,str(running_loss)))

                        #     exit()

                self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    for embedding_type in ["tr","tt","ket","ketxs", "row_col_plus","row_col_mul"]:
        w2v = Word2VecTrainer(input_file=args.text, output_file=args.output,batch_size =args.batch_size,initial_lr = args.lr,embedding_type= embedding_type)
        w2v.train()
    # checker = Word2VecChecker()

