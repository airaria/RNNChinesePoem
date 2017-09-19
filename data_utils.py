import numpy as np
from collections import Counter,defaultdict
from itertools import chain
from keras.preprocessing.sequence import pad_sequences
import re
from copy import copy

SOS = '^'
EOS = '$'

def soft_onehot(arr,k):
    ret = np.zeros(k,dtype=np.float32)
    ret[arr]=1./len(arr)
    return ret

def encode(poem,c2id):
    return [c2id[c] for c in poem]

def encode_pingze(poem,yun_dict):
    '''
    Only encode pingze now.
    '''
    return np.array([soft_onehot(list(set(yun_dict[c][0])), k=3) for c in poem])

def encode_yun(poem,yun_dict,num_yun):
    return np.array([yun_dict['^'][1][0]]
                    +[yun_dict[poem[i-1]][1][0] for i in range(len(poem)) if poem[i]=='\n']
                    +[yun_dict['$'][1][0]])
    #return np.array([soft_onehot(list(yun_dict[poem[i-1]][1]),k=num_yun) for i in range(len(poem)) if poem[i]=='\n'])


def decode(mystery,id2c):
    return ''.join([id2c[i] for i in mystery])

class dataLoader(object):
    def __init__(self,yan=None):
        raw_poems_file = "data/quan_tang_shi.txt"
        ping = "data/ping.txt"
        ze = "data/ze.txt"

        self.yun_dict = self.load_pingze(ping,ze)
        self.poems = self.preprocessing(raw_poems_file,yan)


        self.c2id, self.id2c = self.build_vocabulary(self.poems)
        self.encoded_poems = [encode(poem,self.c2id) for poem in self.poems]
        self.encoded_pingze = [encode_pingze(poem,self.yun_dict) for poem in self.poems]
        num_yun = max(chain.from_iterable(map(lambda x: x[1], self.yun_dict.values()))) + 1
        self.encoded_yun = [encode_yun(poem,self.yun_dict,num_yun) for poem in self.poems]

        self.encoded_poems_length,self.encoded_poems,\
        self.encoded_pingze, self.encoded_yun,self.encoded_yun_length = \
            zip(*sorted(
                map(lambda i:(len(self.encoded_poems[i]),self.encoded_poems[i],
                              self.encoded_pingze[i],self.encoded_yun[i],len(self.encoded_yun[i])),
                    range(len(self.encoded_poems))),key=lambda triple:triple[0]))



    def load_pingze(self,ping,ze):
        yun_dict = defaultdict(list)
        with open(ping,'r',encoding='utf8') as file_ping:
            for ip, line in enumerate(file_ping):
                for c in line.strip():
                    yun_dict[c].append((1, ip+1))
        with open(ze,'r',encoding='utf8') as file_ze:
            for iz,line in enumerate(file_ze):
                for c in line.strip():
                    yun_dict[c].append((2,iz+1+ip+1))
        for k,v in yun_dict.items():
            yun_dict[k] = tuple(zip(*v))
        yun_dict['^'] = ((0,),(iz+ip+3,))
        yun_dict['$'] = ((0,),(iz+ip+4,))
        yun_dict.default_factory = lambda :((0,),(0,))
        return yun_dict

    def preprocessing(self,fn,yan):
        f = open(fn,'r',encoding='utf8')
        poems = []
        for poem in f:
            line = poem.strip()
            if len(line)<16 or len(line)>2000:
                continue
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem or '：' in poem or ':' in poem:
                continue
            line = line.replace('？','\n').replace('！','\n').replace(' ','').replace('，','\n').replace('。','\n')
            if not yan is None:
                sline = line.split('\n')[:-1]
                aa = list(set(map(len,sline)))
                if aa!=[yan]:
                    continue
            line = SOS+line+EOS
            poems.append(line)
        f.close()
        return poems

    def build_vocabulary(self,poems,th=10):
        vocab = Counter(chain(*poems))
        rares = list(filter(lambda x:x[1]<th, vocab.most_common()))

        vocab_new = copy(vocab)
        vocab_new['?'] = sum(list(map(lambda x:x[1],rares)))
        for k,v in rares:
            del vocab_new[k]

        c2id = {v[0]:k for k,v in enumerate(vocab_new.most_common())}
        for k,v in vocab.items():
            if not k in c2id:
                c2id[k] = c2id['?']
        id2c = {v:k for k,v in c2id.items() if k in vocab_new}

        c2id[" "] = len(id2c)
        id2c[len(id2c)] =" "
        return c2id,id2c

    #iterate
    def __call__(self, is_fixed_length,batch_size,nb_epoch,max_length=None):
        self.current_epoch = 1
        self.nb_epoch = nb_epoch
        self.nb_chunk = len(self.encoded_poems) // batch_size

        if is_fixed_length:
            raw = pad_sequences(self.encoded_poems, maxlen=max_length + 1,
                                padding='post', truncating='post', value=self.c2id[' '])
            raw_pz = pad_sequences(self.encoded_pingze,maxlen=max_length+1,
                                padding='post',truncating='post',value=[1.,0,0])

            X = raw[:, :-1]
            y = raw[:, 1:]
            X_len = np.array(list(map(
                lambda x: min(x, max_length), self.encoded_poems_length)))

            X_pz = raw_pz[:,:-1]
            #y_yun = raw_pz[:,1:]


            for epoch in range(1, nb_epoch + 1):
                self.current_epoch = epoch
                order = np.random.permutation(len(X))
                X = X[order]
                y = y[order]
                X_len = X_len[order]
                X_pz = X_pz[order]
                #y_yun = y_yun[order]


                for i in range(self.nb_chunk):
                    yield X[i * batch_size:(i + 1) * batch_size], \
                          y[i * batch_size:(i + 1) * batch_size], \
                          X_len[i * batch_size:(i + 1) * batch_size],\
                          X_pz[i * batch_size:(i + 1) * batch_size]
                          #y_yun[i * batch_size:(i + 1) * batch_size]

        else:
            X = []
            X_len = []
            y = []
            X_pz = []
            #y_yun = []
            for i in range(self.nb_chunk):
                x = self.encoded_poems[i * batch_size:(i + 1) * batch_size]
                x_len = self.encoded_poems_length[i * batch_size:(i + 1) * batch_size]
                x_pz = self.encoded_pingze[i*batch_size:(i+1)*batch_size]

                raw = pad_sequences(x, maxlen=max(x_len) + 1, padding='post',
                                    truncating='post', value=self.c2id[' '])
                raw_pz = pad_sequences(x_pz,maxlen=max(x_len)+1,
                                        padding='post',truncating='post',value=[1,0,0])

                X.append(raw[:, :-1])
                y.append(raw[:, 1:])
                X_len.append(np.array(x_len))
                X_pz.append(raw_pz[:,:-1])
                #y_yun.append(raw_pz[:,1:])


            for epoch in range(1, nb_epoch + 1):
                self.current_epoch = epoch
                order = np.random.permutation(self.nb_chunk)
                for i in order:
                    yield X[i], y[i], X_len[i],X_pz[i] #,y_yun[i]
