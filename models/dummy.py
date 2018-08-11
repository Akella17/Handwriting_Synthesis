import numpy as np
import tensorflow as tf
from models import unconditional_model_build, conditional_model_build, recognition_model_build

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

strokes = np.load('data/strokes.npy')
stroke = strokes[0]

timesteps = 1200                     # LSTM network is unrolled for 1200 timesteps
hidden_dim = 30 #300
epochs = 10
batch_size = 10
text_len = 70                                # all the text inputs are expanded to this length
char_dims = 78 # len(char_encoding)
num_gaussians_mixturemodel = 20                # number of gaussians in the mixture model to predict outputs
num_gaussians_windowmm = 10                  # number of gaussians in the mixture model to predict 'w'
output_dim = 6*num_gaussians_mixturemodel + 1  # =121
bias = 0

char_encoding = {}
char_decoding = []

def dict_generate():
    with open('data/sentences.txt') as f:
        texts = f.readlines()                  # loads the corresponding text label, size : (6000,<sentence_length>)
    num = np.zeros(1000)                                # 1000 is chosen arbitrarily just as an upperbound to compute
    for text in texts:                                  # the dictionaries for character encodings and decodings
        for ch in text:
            if num[ord(ch)] == 0 :
                num[ord(ch)] = 1
    for ch_num,x in enumerate(num) :
        if x == 1 :
            char_decoding.append(ch_num)
    for i,ch_num in enumerate(char_decoding) :
        char_encoding[ch_num] = i  
        
def dict_encode(text_input):
    dict_generate()
    temp = []
    input_onehot_encoder = OneHotEncoder(n_values=len(char_encoding),sparse = False)
    for ch in text_input :                                              
        temp.append(char_encoding[ord(ch)])         
    temp = np.asarray(temp)
    temp = temp.reshape(len(temp), 1)
    input_onehot_encoded = input_onehot_encoder.fit_transform(temp)       # input_onehot_encoded shape [sentence_length, 78]   
    zer = np.zeros((70-len(input_onehot_encoded),len(char_encoding)))     # zer shape = [70,78] {appending zero(78) columns
                                                                        # won't affect the 'w' calculations}
    return np.concatenate((input_onehot_encoded,zer),0)

def dict_decode(encoded_text):    
    dict_generate()
    out = []
    for i in encoded_text:
        out.append(chr(char_decoding[i]))
    return out

def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer  
    return unconditional_model_build(random_seed)


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer
    return conditional_model_build(dict_encode(text),random_seed)
    


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)
    decoded_text = dict_decode(recognition_model_build(stroke))    
    # Output:
    #   text - str
    return decoded_text # 'welcome to lyrebird'