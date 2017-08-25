import numpy as np
from collections import Counter
from itertools import chain
from keras.preprocessing.sequence import pad_sequences
import re

SOS = '^'
EOS = '$'

def encode(poem,c2id):
    return [c2id[c] for c in poem]

def decode(mystery,id2c):
    return [id2c[i] for i in mystery]

class dataLoader(object):
    def __init__(self,yan=None):
        raw_poems_file = "data/quan_tang_shi.txt"
        self.poems = self.preprocessing(raw_poems_file,yan)
        self.c2id, self.id2c = self.build_vocabulary(self.poems)
        self.encoded_poems = [encode(poem,self.c2id) for poem in self.poems]
        self.encoded_poems_length,self.encoded_poems = list(map(list,
            zip(*sorted(
                map(lambda x:(len(x),x),self.encoded_poems)))))

    def preprocessing(self,fn,yan):
        f = open(fn,'r',encoding='utf8')
        poems = []
        for poem in f:
            line = poem.strip()
            if len(line)<16 or len(line)>2000:
                continue
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem or '：' in poem or ':' in poem:
                continue
            line = line.replace('？','。').replace('！','。').replace(' ','')
            if not yan is None:
                sline = line.replace('，','。').split('。')[:-1]
                if list(set(map(len,sline)))!=[yan]:
                    continue
            line = SOS+line+EOS
            poems.append(line)
        f.close()
        return poems

    def build_vocabulary(self,poems,th=10):
        vocab = Counter(chain(*poems))
        rares = list(filter(lambda x:x[1]<th, vocab.most_common()))
        c2id = {v[0]:k for k,v in enumerate(vocab.most_common())}
        unkown_id = len(vocab)-len(rares)
        for k in rares:
            c2id[k[0]] = unkown_id
        id2c = {v:k for k,v in c2id.items() if v<unkown_id}
        id2c[unkown_id] = "?"

        c2id[" "] = unkown_id+1
        id2c[unkown_id+1] =" "
        return c2id,id2c

    #iterate
    def __call__(self, is_fixed_length,batch_size,nb_epoch,max_length=None):
        self.current_epoch = 1
        self.nb_epoch = nb_epoch
        self.nb_chunk = len(self.encoded_poems) // batch_size

        if is_fixed_length:
            raw = pad_sequences(self.encoded_poems, maxlen=max_length + 1,
                                padding='post', truncating='post', value=self.c2id[' '])
            X = raw[:, :-1]
            y = raw[:, 1:]
            X_len = np.array(list(map(
                lambda x: min(x, max_length), self.encoded_poems_length)))
            for epoch in range(1, nb_epoch + 1):
                self.current_epoch = epoch
                order = np.random.permutation(len(X))
                X = X[order]
                y = y[order]
                X_len = X_len[order]
                for i in range(self.nb_chunk):
                    yield X[i * batch_size:(i + 1) * batch_size], \
                          y[i * batch_size:(i + 1) * batch_size], \
                          X_len[i * batch_size:(i + 1) * batch_size]

        else:
            X = []
            X_len = []
            y = []
            for i in range(self.nb_chunk):
                x = self.encoded_poems[i * batch_size:(i + 1) * batch_size]
                x_len = self.encoded_poems_length[i * batch_size:(i + 1) * batch_size]
                raw = pad_sequences(x, maxlen=max(x_len) + 1, padding='post',
                                    truncating='post', value=self.c2id[' '])
                X.append(raw[:, :-1])
                y.append(raw[:, 1:])
                X_len.append(np.array(x_len))

            for epoch in range(1, nb_epoch + 1):
                self.current_epoch = epoch
                order = np.random.permutation(self.nb_chunk)
                for i in order:
                    yield X[i], y[i], X_len[i]