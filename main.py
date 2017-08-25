import os
import optparse
import tensorflow as tf
from data_utils import *
from utils import get_session
from Model import RNNmodel

if __name__ == '__main__':
    sess = get_session()
    poemdata = dataLoader()

    vocab_size = len(poemdata.id2c)
    embedding_size = 256
    rnn_units = 128
    rnn_layers = 2
    grad_clip = 10
    save_every_n = 1000
    log_every_n = 20
    sample_every_n = 50
    learning_rate = 0.005
    dropout_rate = 0.0
    save_dir = 'saved_model'
    batch_size = 64
    num_epochs = 30
    maxlen = 100
    mode = 'train'
    start_chars = ''

    model = RNNmodel(vocab_size=vocab_size,
                     embedding_size=embedding_size,
                     rnn_units=rnn_units,
                     rnn_layers=rnn_layers,
                     grad_clip=grad_clip,
                     save_every_n=save_every_n,
                     log_every_n=log_every_n,
                     learning_rate=learning_rate,
                     dropout_rate=dropout_rate)
    #model.save_graph(sess)

    if mode == 'train':
        model.train(sess,poemdata,log_every_n,save_every_n,sample_every_n,
                    save_dir=save_dir,is_fixed_length=True,
                    batch_size=batch_size,num_epochs=num_epochs,
                    maxlen=maxlen)

    elif mode == 'sample':
        start_chars = SOS + start_chars
        stop_code = encode(EOS,poemdata.c2id)[0]
        start_codes = encode(start_chars,poemdata.c2id)
        poems_codes = model.sample(sess,start_codes,stop_code,save_dir,nb_poem=10)
        poems = [''.join(decode(poem_codes,poemdata.id2c)) for poem_codes in poems_codes]
        print ('\n'.join(poems))