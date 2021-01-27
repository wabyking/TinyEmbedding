import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3
import math

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)



class TTEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 voc_size=None,
                 emb_size=None,
                 auto_shapes=None,
                 auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 d=3,
                 tt_rank=8,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TTEmbedding, self).__init__()
        self.ori_voc_size = voc_size
        self.ori_emb_size = emb_size
        if auto_shapes:
            voc_quantization = t3.utils.suggest_shape(
                voc_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            emb_quantization = t3.utils.auto_shape(
                emb_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [voc_quantization, emb_quantization]
            self.shape = shape

        else:
            self.shape = shape

        if init is None:
            if shape is None:
                raise ValueError('if init is not provided,'
                                 ' please specify shape')
        else:
            self.shape = init.raw_shape


        if init is None:
            init = t3.glorot_initializer(self.shape, tt_rank=tt_rank)

        self.tt_matrix = init.to_parameter()
        self.parameters = self.tt_matrix.parameter

        # for p in self.parameters():
        #    p.name = 'tt_core'

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = self.shape[0]
        self.emb_quant = self.shape[1]

        self.padding_idx = padding_idx
        self.naive = naive

    def get_weights(self): # why twice
        # waby write, just for a try
        if self.naive:
            self.full = t3.naive_full(self.tt_matrix)
        else:
            self.full = self.tt_matrix.full()
        return self.full[:self.ori_voc_size, :self.ori_emb_size]

    def forward(self, x):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        # x_ind = t3.ind2sub(self.voc_quant, x)
        # rows = t3.gather_rows(self.tt_matrix, x_ind)
        #
        # rows = rows.view(x.shape[0], -1)
        if self.naive:
            self.full = t3.naive_full(self.tt_matrix)
        else:
            self.full = self.tt_matrix.full()
        rows = self.full[x]

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        return rows.to(x.device)
    def initialize(self, embeddings, steps=1000):
        if type(embeddings) != torch.Tensor:
            embeddings = torch.Tensor(embeddings) #.cuda
        # print([p.size() for p in self.parameters()])
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
        es = EarlyStopping(patience=6)
        for i in range(steps):
            # print(self.get_weights())
            loss = torch.mean((self.get_weights() - embeddings)**2)
            #loss = torch.mean(torch.abs(self.get_weights() - embeddings) )
            loss.backward()
            if es.step(loss):
                print("early stoped")
                break
            print("compressing word embeddings with loss {}".format(loss.item()))
            optimizer.step()

class TREmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 voc_size=None,
                 emb_size=None,
                 auto_shapes=None,
                 auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 d=3,
                 tt_rank=8,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TREmbedding, self).__init__()

        self.ori_voc_size = voc_size
        self.ori_emb_size = emb_size
        if auto_shapes:
            voc_quantization = t3.utils.suggest_shape(
                voc_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            emb_quantization = t3.utils.auto_shape(
                emb_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [voc_quantization, emb_quantization]
            self.shape = shape

        else:
            self.shape = shape

        if init is None:
            if shape is None:
                raise ValueError('if init is not provided,'
                                 ' please specify shape')
        else:
            self.shape = init.raw_shape


        if init is None:
            init = t3.glorot_initializer_tr(self.shape, tr_rank=tt_rank)

        self.tr_matrix = init.to_parameter()
        self.parameters = self.tr_matrix.parameter

        # for p in self.parameters():
        #    p.name = 'tt_core'

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = self.shape[0]
        self.emb_quant = self.shape[1]

        self.padding_idx = padding_idx
        self.naive = naive

        # waby rite it, just first try
    def get_weights(self):  # why twice
        # waby write, just for a try
        if self.naive:
            self.full = t3.naive_full(self.tt_matrix)
        else:
            self.full = self.tr_matrix.full()
        return self.full[:self.ori_voc_size, :self.ori_emb_size]

    def forward(self, x):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        # x_ind = t3.ind2sub(self.voc_quant, x)
        # rows = t3.gather_rows(self.tr_matrix, x_ind)
        #
        # rows = rows.view(x.shape[0], -1)
        if self.naive:
            self.full = t3.naive_full(self.tr_matrix)
        else:
            self.full = self.tr_matrix.full()
        rows = self.full[x]

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        return rows.to(x.device)

    def initialize(self, embeddings, steps=1000):
        if type(embeddings) != torch.Tensor:
            embeddings = torch.Tensor(embeddings) #.cuda
        # print([p.size() for p in self.parameters()])
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
        es = EarlyStopping(patience=6)
        for i in range(steps):
            # print(self.get_weights())
            loss = torch.mean((self.get_weights() - embeddings)**2)
            #loss = torch.mean(torch.abs(self.get_weights() - embeddings) )
            loss.backward()
            if es.step(loss):
                print("early stoped")
                break
            print("compressing word embeddings with loss {}".format(loss.item()))
            optimizer.step()



class TTLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=8, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy', naive=False
                 ):
        super(TTLinear, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)

        self.shape = shape
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter
        if naive:
            self.mm_op = t3.naive_dense_tt_matmul
        else:
            self.mm_op = t3.dense_tt_matmul
        if bias:
            self.bias = torch.nn.Parameter(1e-3 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        if self.bias is None:
            return self.mm_op(x, weight)
        else:
            return self.mm_op(x, weight) + self.bias


class TRLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=8, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy', naive=False
                 ):
        super(TRLinear, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer_tr(shape, tr_rank=tt_rank)

        self.shape = shape
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter
        if naive:
            self.mm_op = t3.naive_dense_tr_matmul
        else:
            raise ValueError('Not implemented, use naive option.')
        if bias:
            self.bias = torch.nn.Parameter(1e-3 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        if self.bias is None:
            return self.mm_op(x, weight)
        else:
            return self.mm_op(x, weight) + self.bias