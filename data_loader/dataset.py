import copy
import json
import sys
import os
os.chdir(sys.path[0])
import torch
from dgl import graph as DGLGraph
from tqdm import tqdm
from utils.ops import load_default_identifiers, initialize_batch, debug

def convert_examples_to_features(code, tokenizer):
    block_size = 512
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_ids

class DataEntry:# 这边就是一个完整的图数据，包括节点特征，边，边类型
    def __init__(self, datset, num_nodes, features, edges, target,code):
        self.dataset = datset
        self.code = code  # code sequence
        self.num_nodes = num_nodes
        self.target = target
        self.graph = DGLGraph(([], []))
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})   ##
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edges(s, t, data={'etype': torch.LongTensor([etype_number])})

class DataSet:
    def __init__(self, train_src, valid_src, test_src, batch_size, tokenizer, n_ident=None, g_ident=None, l_ident=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.tokenizer = tokenizer
        self.n_ident, self.g_ident, self.l_ident = load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset(train_src, valid_src, test_src)
        self.initialize_dataset()

    def initialize_dataset(self):

        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, train_src, valid_src=None, test_src=None):
        debug('Reading Train File!')
        
        with open(train_src,"r") as fp:
            for i, line in enumerate(tqdm(fp)):
                entry = json.loads(line)
                example = DataEntry(datset=self, num_nodes=len(entry['full_graph'][self.n_ident]), features=entry['full_graph'][self.n_ident],
                                    edges=entry['full_graph'][self.g_ident], target=entry['full_graph'][self.l_ident][0][0],code=convert_examples_to_features(entry['func'], self.tokenizer)) #这里的target是label
                                  
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                    debug('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
        
        if valid_src is not None:
            debug('Reading Validation File!')
            
            with open(valid_src,"r") as fp:
                for i, line in enumerate(tqdm(fp)):
                    entry = json.loads(line)
                    example = DataEntry(datset=self, num_nodes=len(entry['full_graph'][self.n_ident]),
                                        features=entry['full_graph'][self.n_ident],
                                        edges=entry['full_graph'][self.g_ident], target=entry['full_graph'][self.l_ident][0][0],code=convert_examples_to_features(entry['func'], self.tokenizer),)
                    self.valid_examples.append(example)
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                for i, line in enumerate(tqdm(fp)):
                    entry = json.loads(line)
                    example = DataEntry(datset=self, num_nodes=len(entry['full_graph'][self.n_ident]),
                                        features=entry['full_graph'][self.n_ident],
                                        edges=entry['full_graph'][self.g_ident], target=entry['full_graph'][self.l_ident][0][0],code=convert_examples_to_features(entry['func'], self.tokenizer),)
                    self.test_examples.append(example)


    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size, shuffle=False)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size, shuffle=False)
        
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        code = [e.code for e in taken_entries]
        labels = [e.target for e in taken_entries]
        gs = [e.graph for e in taken_entries]
        hs = [e.features for e in taken_entries]
        # batch_graph = GGNNBatchGraph() #创建一个 GGNNBatchGraph 对象，这可能是一个用于表示图神经网络批次图的类。这个对象将用于存储和管理多个图的集合。
        # for entry in taken_entries:
        #     batch_graph.add_subgraph(copy.deepcopy(entry.graph))

        return gs, hs, code, torch.FloatTensor(labels)

    def get_next_train_batch(self):

        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
