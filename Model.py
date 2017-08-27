import tensorflow as tf
import os
from model_helper import *
from data_utils import *
from utils import *

class RNNmodel(object):
    def __init__(self,**kwargs):
        self.vocab_size = vocab_size = kwargs['vocab_size']
        self.embedding_size= embedding_size = kwargs['embedding_size']
        self.rnn_units = rnn_units = kwargs['rnn_units']
        self.rnn_layers = rnn_layers = kwargs['rnn_layers']
        self.grad_clip = kwargs['grad_clip']
        self.learning_rate  = kwargs['learning_rate']
        self.dropout = kwargs['dropout_rate']

        with tf.variable_scope("PoemModel"):
            with tf.variable_scope("placeholders"):
                self.x = tf.placeholder(dtype=tf.int32,shape=(None,None),name='input')
                self.y = tf.placeholder(dtype=tf.int32,shape=(None,None),name='target')
                self.xl = tf.placeholder(dtype=tf.int32,shape=(None,),name='length')
                self.xyun = tf.placeholder(dtype=tf.float32,shape=(None,None,3),name='input_yun')

                self.dropout_rate = tf.placeholder(tf.float32,shape=(),name='dropout_rate')

            batch_size = tf.shape(self.x)[0]
            sample_length = tf.shape(self.x)[1]

            with tf.variable_scope("embedding"):
                with tf.device("/cpu:0"):
                    self.embeddings = tf.get_variable(
                        "embeddingTable",[vocab_size,embedding_size],dtype=tf.float32)
                    self.embedded_x = tf.nn.embedding_lookup(self.embeddings,self.x)

            with tf.variable_scope("concat"):
                pre_inputs = tf.concat([self.embedded_x,self.xyun],axis=2)

            rnn = create_rnn_cell(num_units=rnn_units,
                                      num_layers=rnn_layers,
                                      forget_bias=1,
                                      dropout=self.dropout_rate)
            self.initial_state = rnn.zero_state(batch_size,tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(
                rnn,
                pre_inputs,
                sequence_length=self.xl,
                initial_state= self.initial_state)

            with tf.variable_scope("afterRNN"):
                flat_outputs = tf.reshape(outputs,shape=[-1,rnn_units])
                self.flat_logits = tf.layers.dense(flat_outputs,vocab_size)
                self.flat_probs = tf.nn.softmax(self.flat_logits)
                #self.cubic_logits = tf.reshape(self.flat_logits,
                #            shape=(batch_size,sample_length,vocab_size))
                #self.cubic_prob = tf.nn.softmax(self.cubic_logits)

            self.build_loss()
            self.build_train_op()
            self.initialize = tf.global_variables_initializer()
            self.saver = tf.train.Saver()


    def build_loss(self):
        with tf.variable_scope("loss"):
            flat_y = tf.reshape(self.y,shape=(-1,),name='flat_y')
            flat_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = flat_y,
                logits = self.flat_logits,
                name='flat_loss'
            )
            self.loss = tf.reduce_mean(flat_loss,name="scalarloss")

    def build_train_op(self):
        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),tf.constant(self.grad_clip,dtype=tf.float32))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars))


    def train(self,sess,dataset,
              log_every_n,
              save_every_n,
              sample_every_n,
              save_dir,
              is_fixed_length,
              batch_size,
              num_epochs,
              maxlen=None):
        sess.run(self.initialize)
        step = 0
        for X_data,y_data,X_len,X_yun in dataset(is_fixed_length,batch_size,num_epochs,maxlen):
            step += 1
            train_loss,_ = sess.run([self.loss,self.train_op],feed_dict={
                self.x:X_data,self.y:y_data,self.xl:X_len,self.xyun:X_yun,self.dropout_rate:self.dropout})

            if step % log_every_n ==0:
                print ('{}/{} in {}/{}  loss: {:.4f}'\
                       .format((step-1)%(dataset.nb_chunk)+1,dataset.nb_chunk,
                               dataset.current_epoch,
                               dataset.nb_epoch,train_loss))

            if step % sample_every_n == 0:
                a_sample = self.sample(sess,SOS,EOS,dataset)[0]
                print (a_sample)

            if step % save_every_n==0:
                self.saver.save(sess,os.path.join(save_dir,'model'),global_step=step)
        self.saver.save(sess,os.path.join(save_dir,'model'),global_step=step)

    def sample(self,sess,start_words,stop_symbol,dataset,save_dir=None,nb_poem=1):
        start_codes = encode(start_words,dataset.c2id)
        stop_code = dataset.c2id[stop_symbol]
        if not save_dir is None:
            model_file = tf.train.latest_checkpoint(save_dir)
            print(model_file)
            self.saver.restore(sess,model_file)

        x = np.zeros((1, 1))
        xyun = np.zeros((1, 1, 3),dtype=np.float32)
        state = sess.run(self.initial_state, feed_dict={self.x: np.zeros((1, 1)), self.xl: [1]})

        poems_codes = []
        for pt in range(nb_poem):
            poem_codes = start_codes[:]
            last_code = stop_code
            for c in start_codes:
                x[0,0] = c
                xyun[0] = encode_yun(dataset.id2c[c],dataset.yun_dict)
                probs, state = sess.run([self.flat_probs,self.final_state],
                                        feed_dict={self.x:x,self.xl:[1],self.xyun:xyun,
                                                   self.dropout_rate:0,
                                                   self.initial_state:state})
                last_code = sampler(probs,3) #random_choice_top_k(logits,5)
            while last_code!=stop_code and len(poem_codes)<100:
                poem_codes.append(last_code)
                x[0,0] = last_code
                xyun[0] = encode_yun(dataset.id2c[last_code],dataset.yun_dict)
                probs, state = sess.run([self.flat_probs,self.final_state],
                                        feed_dict={self.x:x,self.xl:[1],self.xyun:xyun,
                                                   self.dropout_rate:0,
                                                   self.initial_state:state})
                last_code = sampler(probs,3)
            poems_codes.append(decode(poem_codes,dataset.id2c))
        return poems_codes

    def save_graph(self,sess):
        writer = tf.summary.FileWriter('./graphs',sess.graph)
        writer.close()