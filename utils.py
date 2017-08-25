import tensorflow as tf
import numpy as np
def is_ch(c):
    return "\u4e00"<=c<="\u9fa5" or c in "，。！？,.?!"

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    return session

def sampler(probs,k=None):
    probs = np.squeeze(probs)
    if k is None:
        return np.argmax(probs)
    else:
        top_k = np.argsort(probs)[-k:]
        top_k_p = probs[top_k]/np.sum(probs[top_k])
        return np.random.choice(top_k,p=top_k_p)
