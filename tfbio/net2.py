import numpy as np

from math import ceil
import io
# activations
import tensorflow as tf

#import matplotlib as mpl
#mpl.use('agg')

import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()

import matplotlib.pyplot as plt



'''
def hidden_conv3D(inp, out_chnls, conv_patch=5, pool_patch=2, name='conv'):
    #hidden_conv3D函数是定义一层卷积神经网络。    
 
    in_chnls = inp.get_shape()[-1].value   
    
    with tf.variable_scope(name):
        w_shape = (conv_patch, conv_patch, conv_patch, in_chnls, out_chnls)
        w = tf.get_variable('w', shape=w_shape,
                            initializer=tf.truncated_normal_initializer(
                                stddev=0.001))
        b = tf.get_variable('b', shape=(out_chnls,), dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv3d(inp, w, strides=(1, 1, 1, 1, 1), padding='SAME',
                            name='conv')
       
        h = tf.nn.relu(conv + b, name='h')

        pool_shape = (1, pool_patch, pool_patch, pool_patch, 1)
        
        h_pool = tf.nn.max_pool3d(h, ksize=pool_shape, strides=pool_shape,
                                  padding='SAME', name='h_pool')
     
    return h_pool
'''

def hidden_fcl(inp, out_size, keep_prob, name='fc'):   
    #hidden_fcl函数是定义一层全连接网络。
    
    assert len(inp.get_shape()) == 2

    in_size = inp.get_shape()[1].value

    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=(in_size, out_size),
                            initializer=tf.truncated_normal_initializer(
                                stddev=(1 / (in_size**0.5))))
        b = tf.get_variable('b', shape=(out_size,), dtype=tf.float32,
                            initializer=tf.constant_initializer(1))

        h = tf.nn.relu(tf.matmul(inp, w) + b, name='h')
        h_drop = tf.nn.dropout(h, keep_prob, name='h_dropout')

    return h_drop

'''
def convolve3D(inp, channels, conv_patch=5, pool_patch=2):
    prev = inp
    i = 0
    for num_channels in channels:
        #下面是调用前面定义的hidden_conv3D函数。循环3次，从而创建3层卷积神经网络。
        output = hidden_conv3D(prev, num_channels, conv_patch, pool_patch,
                               name='conv%s' % i)
        i += 1
        prev = output
    return output
'''

def feedforward(inp, fc_sizes, keep_prob=1.0):#原始1.0
    prev = inp
    i = 0
    for hsize in fc_sizes:
        #下面是调用前面定义的hidden_fcl函数。循环3次，从而创建3层全连接网络。
        output = hidden_fcl(prev, hsize, keep_prob, name='fc%s' % i)
        i += 1
        prev = output
    return output

def sequence_cnn_model(XDinput, XTinput, embedding_size, name='sequence_cnn_model'):
    
    #XDinput = tf.placeholder(tf.int32, shape=[None, 100], name='drug')
    #XTinput = tf.placeholder(tf.int32, shape=[None, 1000], name='protein')
    encode1 = tf.Variable(tf.random_uniform([64+1, embedding_size], -1.0, 1.0))
    encode_smiles = tf.nn.embedding_lookup(encode1, XDinput)
    encode_smiles = tf.layers.conv1d(inputs=encode_smiles, filters=32, kernel_size=4, strides=1, padding='valid', activation=tf.nn.relu) 
    encode_smiles = tf.layers.conv1d(inputs=encode_smiles, filters=64, kernel_size=6, strides=1, padding='valid', activation=tf.nn.relu) 
    encode_smiles = tf.layers.conv1d(inputs=encode_smiles, filters=96, kernel_size=8, strides=1, padding='valid', activation=tf.nn.relu) 
    encode_smiles = tf.reduce_max(encode_smiles, [1], name='pool', keepdims=False)
    #encode_smiles = tf.layers.max_pooling1d(inputs=encode_smiles, pool_size, strides, padding='valid', data_format='channels_last',name=None)
    
    
    encode2 = tf.Variable(tf.random_uniform([25+1, embedding_size], -1.0, 1.0))  
    encode_protein = tf.nn.embedding_lookup(encode2, XTinput)
    encode_protein = tf.layers.conv1d(inputs=encode_protein, filters=32, kernel_size=4, strides=1, padding='valid', activation=tf.nn.relu) 
    encode_protein = tf.layers.conv1d(inputs=encode_protein, filters=64, kernel_size=8, strides=1, padding='valid', activation=tf.nn.relu) 
    encode_protein = tf.layers.conv1d(inputs=encode_protein, filters=96, kernel_size=12, strides=1, padding='valid', activation=tf.nn.relu) 
    encode_protein = tf.reduce_max(encode_protein, [1], name='pool', keepdims=False)
    #encode_protein = tf.layers.max_pooling1d(inputs=encode_protein, pool_size, strides, padding='valid', data_format='channels_last',name=None)
    
    
    sequence = tf.concat([encode_smiles, encode_protein], 1)
    
    
    return sequence
'''    
def make_SB_network(isize=20, in_chnls=19, osize=1, max_smi_len=100, max_seq_len=1000, 
                    conv_patch=5, pool_patch=2, conv_channels=[64, 128, 256],
                    dense_sizes=[1000, 500, 200],
                    lmbda=0.001, learning_rate=1e-5,
                    seed=123):
'''
def make_SB_network(osize=1, max_smi_len=100, max_seq_len=1000, 
                    dense_sizes=[1024, 1024, 512],#原始=[1000, 500, 200]
                    learning_rate=1e-5,#原始learning_rate=1e-5
                    seed=123):
   
    graph = tf.Graph()
    
    
    #定义复合物的输入和结合强度的真实值。
    with graph.as_default():
        np.random.seed(seed)
        tf.set_random_seed(seed)
        with tf.variable_scope('input'):
            #x = tf.placeholder(tf.float32,
            #                   shape=(None, isize, isize, isize, in_chnls),
            #                   name='structure_x')
           
            XD = tf.placeholder(tf.int32, 
                                shape=(None, max_smi_len),
                                name='sequence_xd')
            
            XT = tf.placeholder(tf.int32, 
                                shape=(None, max_seq_len),
                                name='sequence_xt')
            
            t = tf.placeholder(tf.float32, shape=(None, osize), name='affinity')
            
        '''    
        #调用卷积神经网络，并把复合物结构放入卷积神经网络。
        with tf.variable_scope('structure_convolution'):
            structure_cnn = convolve3D(x, conv_channels,
                                 conv_patch=conv_patch,
                                 pool_patch=pool_patch)
        hfsize = isize
        for _ in range(len(conv_channels)):
            hfsize = ceil(hfsize / pool_patch) 
        hfsize = conv_channels[-1] * hfsize**3 
        '''
        
        with tf.variable_scope('sequence_convolution'):
            sequence_cnn = sequence_cnn_model(XD, XT, 128, name='sequence_cnn')
            
        

        with tf.variable_scope('fully_connected'):
            #structure_cnn = tf.reshape(structure_cnn, shape=(-1, hfsize), name='structure_cnn')
            
            #result_cnn = tf.concat([structure_cnn, sequence_cnn], 1)
            result_cnn = sequence_cnn
            print("result_cnn", result_cnn)
            
            prob1 = tf.constant(0.1, name='keep_prob_default')#原始1.0
            keep_prob = tf.placeholder_with_default(prob1, shape=(),
                                                    name='keep_prob')
            #调用全连接网络函数，将卷积神经网络的最终输出作为输入，放入全连接网络。
            h_fcl = feedforward(result_cnn, dense_sizes, keep_prob=keep_prob)
            
            
        

        with tf.variable_scope('output'):
            w = tf.get_variable('w', shape=(dense_sizes[-1], osize),
                                initializer=tf.truncated_normal_initializer(
                                    stddev=(1 / (dense_sizes[-1]**0.5))))
            b = tf.get_variable('b', shape=(osize,), dtype=tf.float32,
                                initializer=tf.constant_initializer(1))
            y = tf.nn.relu(tf.matmul(h_fcl, w) + b, name='prediction')

        with tf.variable_scope('training'):
            global_step = tf.get_variable('global_step', shape=(),
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)
           # global_step在滑动平均、优化器、指数衰减学习率等方面都有用到，这个变量的实际意义非常好理解：
           #代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表。
           #为什么不直接用global_step=tf.Variable(0, trainable=False) ？？？？

            mse = tf.reduce_mean(tf.pow((y - t), 2), name='mse')

          
            cost = mse

            optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer')
            train = optimizer.minimize(cost, global_step=global_step,
                                       name='train')
    #损失函数优化器的minimize()中global_step=global_steps能够提供global_step自动加一的操作。
    
   

    graph.add_to_collection('output', y)
    #graph.add_to_collection('structure_x', x)
    graph.add_to_collection('sequence_xd', XD)
    graph.add_to_collection('sequence_xt', XT)    
    graph.add_to_collection('real', t)
    graph.add_to_collection('kp', keep_prob)
    #add_to_collectio为Graph的一个方法，可以简单地认为Graph下维护了一个字典，
    #key为name,value为list，而add_to_collection就是把变量添加到对应key下的list中

    return graph


def custom_summary_image(mpl_figure):

    if not isinstance(mpl_figure, plt.Figure):
        raise TypeError('mpl_figure must be matplotlib.figure.Figure object,'
                        '%s was given' % type(mpl_figure))

    imgdata = io.BytesIO()
    mpl_figure.savefig(imgdata, format='png')
    imgdata.seek(0)

    width, height = mpl_figure.canvas.get_width_height()

    image = tf.Summary.Image(height=height, width=width, colorspace=3,
                             encoded_image_string=imgdata.getvalue())
    imgdata.close()

    return image


