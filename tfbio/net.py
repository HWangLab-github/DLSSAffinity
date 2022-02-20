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
print('31_&&&&&&&&&&&&&&&')

import matplotlib.pyplot as plt


def hidden_conv3D(inp, out_chnls, conv_patch=5, pool_patch=2, name='conv'):    
    #hidden_conv3D函数是定义一层卷积神经网络。
    print('32_&&&&&&&&&&&&&&&')
    '''  
    
    print('whwinp:',  inp )
    >>>out: 
        whwinp: Tensor("input/structure:0", shape=(?, 21, 21, 21, 19), dtype=float32)
        whwinp: Tensor("convolution/conv0/h_pool:0", shape=(?, 11, 11, 11, 64), dtype=float32)
        whwinp: Tensor("convolution/conv1/h_pool:0", shape=(?, 6, 6, 6, 128), dtype=float32)
        即inp表示的是每层卷积层的输入张量的形状
        
    print('whwout_chnls:',  out_chnls )    
    >>>out:
        whwout_chnls: 64
        whwout_chnls: 128
        whwout_chnls: 256
        即out_chnls表示的是每层卷积层的卷积核数
    '''    
    in_chnls = inp.get_shape()[-1].value   
    #in_chnls输出应该为19， 64， 128
    with tf.variable_scope(name):
        w_shape = (conv_patch, conv_patch, conv_patch, in_chnls, out_chnls)
        '''
        print('w_shape:', w_shape)
        >>>out:
        w_shape: (5, 5, 5, 19, 64)
        w_shape: (5, 5, 5, 64, 128)
        w_shape: (5, 5, 5, 128, 256)
        '''
        w = tf.get_variable('w', shape=w_shape,
                            initializer=tf.truncated_normal_initializer(
                                stddev=0.001))
        b = tf.get_variable('b', shape=(out_chnls,), dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv3d(inp, w, strides=(1, 1, 1, 1, 1), padding='SAME',
                            name='conv')
        #因为strides=(1, 1, 1, 1, 1), padding='SAME',所以输入和输出的复合物大小一样，即21*21*21还是21*21*21
        h = tf.nn.relu(conv + b, name='h')

        pool_shape = (1, pool_patch, pool_patch, pool_patch, 1)
        '''
        为什么pool的卷积核形状为(1, pool_patch, pool_patch, pool_patch, 1)？？？
        '''
        
        h_pool = tf.nn.max_pool3d(h, ksize=pool_shape, strides=pool_shape,
                                  padding='SAME', name='h_pool')
        #因为strides=pool_shape,padding='SAME',
        #如果卷积前图片大小为奇数，则pool层卷积后的图片大小等于（卷积前图片大小+1）/pool_shape 
        #如果卷积前图片大小为偶数，则pool层卷积后的图片大小等于（卷积前图片大小）/pool_shape 
        '''
        print('h:', h)
        >>>out
        h: Tensor("convolution/conv0/h:0", shape=(?, 21, 21, 21, 64), dtype=float32)
        h: Tensor("convolution/conv1/h:0", shape=(?, 11, 11, 11, 128), dtype=float32)
        h: Tensor("convolution/conv2/h:0", shape=(?, 6, 6, 6, 256), dtype=float32)
        即h为pool层的输入，也是上一层卷积层的输出。
        '''
    return h_pool


def hidden_fcl(inp, out_size, keep_prob, name='fc'):
    print('33_&&&&&&&&&&&&&&&')
    """Create fully-connected hidden layer with dropout. This function
    performs transformation:

        y = dropout(relu(wx + b)))

    where x is an input layer and y is an output layer returned by this
    function. All created tensors are in the name scope defined by `name`.

    Parameters
    ----------
    inp: 2D tf.Tensor
        Input tensor
    out_size: int
        Number of neurons in the layer.
    keep_prob: float or 0D tf.Tensor
        Keep probability for dropout layer
    name: str, optional
        Name for the `variable_scope`

    Returns
    -------
    h_drop: 2D tf.Tensor
        Output tensor
    """
    
    
    
    #hidden_fcl函数是定义一层全连接网络。
    
    
    
    assert len(inp.get_shape()) == 2
    #assert函数：其作用是如果它的条件返回错误，则终止程序执行
    '''
    print('inp:', inp)
    >>>out
    inp: Tensor("fully_connected/h_flat:0", shape=(?, 6912), dtype=float32)
    inp: Tensor("fully_connected/fc0/h_dropout/mul:0", shape=(?, 1000), dtype=float32)
    inp: Tensor("fully_connected/fc1/h_dropout/mul:0", shape=(?, 500), dtype=float32)
    '''
    in_size = inp.get_shape()[1].value

    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=(in_size, out_size),
                            initializer=tf.truncated_normal_initializer(
                                stddev=(1 / (in_size**0.5))))
        b = tf.get_variable('b', shape=(out_size,), dtype=tf.float32,
                            initializer=tf.constant_initializer(1))

        h = tf.nn.relu(tf.matmul(inp, w) + b, name='h')
        h_drop = tf.nn.dropout(h, keep_prob, name='h_dropout')
        #dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。

    return h_drop


def convolve3D(inp, channels, conv_patch=5, pool_patch=2):
    print('34_&&&&&&&&&&&&&&&')
    """Create block of 3D convolutional layers with max pooling using
    `hidden_conv3D`. The i'th layer in the block is in name scope called
    'conv<i>' and has channels[i] number of filters.

    Parameters
    ----------
    inp: 5D tf.Tensor
        Input tensor
    channels: array-like, shape = (N,)
        Numbers of filters in convolutions.
    conv_patch: int, optional
        Size of convolution patch
    pool_patch: int, optional
        Size of max pooling patch


    Returns
    -------
    output: 5D tf.Tensor
        Output of last layer
    """

    prev = inp
    i = 0
    for num_channels in channels:
        ''''
        print('channels:', channels)
        >>>out:
            channels: [64, 128, 256]
            channels: [64, 128, 256]
            channels: [64, 128, 256]
        '''
        #下面是调用前面定义的hidden_conv3D函数。循环3次，从而创建3层卷积神经网络。
        output = hidden_conv3D(prev, num_channels, conv_patch, pool_patch,
                               name='conv%s' % i)
        '''
        print('prev:',prev)
        print('num_channels:',num_channels)
        print('conv_patch:', conv_patch)
        print('pool_patch:', pool_patch)
        >>>out
        prev: Tensor("input/structure:0", shape=(?, 21, 21, 21, 19), dtype=float32)
        num_channels: 64
        conv_patch: 5
        pool_patch: 2
        prev: Tensor("convolution/conv0/h_pool:0", shape=(?, 11, 11, 11, 64), dtype=float32)
        num_channels: 128
        conv_patch: 5
        pool_patch: 2
        prev: Tensor("convolution/conv1/h_pool:0", shape=(?, 6, 6, 6, 128), dtype=float32)
        num_channels: 256
        conv_patch: 5
        pool_patch: 2
        '''
        i += 1
        prev = output
    return output


def feedforward(inp, fc_sizes, keep_prob=1.0):
    print('35_&&&&&&&&&&&&&&&')
    
    #print('inp:', inp)
    #np: Tensor("fully_connected/h_flat:0", shape=(?, 6912), dtype=float32)
    
    """Create block of fully-connected layers with dropout using
    `hidden_fcl`. The i'th layer in the block is in name scope called
    'fc<i>' and has fc_sizes[i] number of neurons.

    Parameters
    ----------
    inp: 2D tf.Tensor
        Input tensor
    fc_sizes: array-like, shape = (N,)
        Numbers of neurons in layers.
    keep_prob: float or 0D tf.Tensor, optional
        Keep probability for dropout layer


    Returns
    -------
    output: 2D tf.Tensor
        Output of last layer
    """

    prev = inp
    i = 0
    for hsize in fc_sizes:
        #print('hsize:', hsize)
        #hsize: 1000
        #hsize: 500
        #hsize: 200
        #下面是调用前面定义的hidden_fcl函数。循环3次，从而创建3层全连接网络。
        output = hidden_fcl(prev, hsize, keep_prob, name='fc%s' % i)
        i += 1
        prev = output
    return output

    #想想这个模型的全连接网络是不是只有前馈而没有反馈。！！！！！


def make_SB_network(isize=20, in_chnls=19, osize=1,
                    conv_patch=5, pool_patch=2, conv_channels=[64, 128, 256],
                    dense_sizes=[1000, 500, 200],
                    lmbda=0.001, learning_rate=1e-5,
                    seed=123):
    #make_SB_network函数是主函数，里面所有的计算是以一个batch为单位的，所以定义cost, 
    #正则化L2时，都要对一个batch的计算值求和。
   
    graph = tf.Graph()
    print('36_&&&&&&&&&&&&&&&')
    
    #定义复合物的输入和结合强度的真实值。
    with graph.as_default():
        np.random.seed(seed)
        tf.set_random_seed(seed)
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32,
                               shape=(None, isize, isize, isize, in_chnls),
                               name='structure')
            t = tf.placeholder(tf.float32, shape=(None, osize), name='affinity')
            
        #调用卷积神经网络，并把复合物结构放入卷积神经网络。
        with tf.variable_scope('convolution'):
            h_convs = convolve3D(x, conv_channels,
                                 conv_patch=conv_patch,
                                 pool_patch=pool_patch)
        hfsize = isize
        for _ in range(len(conv_channels)):
            hfsize = ceil(hfsize / pool_patch) 
        hfsize = conv_channels[-1] * hfsize**3 

        with tf.variable_scope('fully_connected'):
            h_flat = tf.reshape(h_convs, shape=(-1, hfsize), name='h_flat')
            
            #print('h_flat:', h_flat)
            # print('aaaaa')
            #print(type(h_flat))
            #print('bbbbbbb')
            #sess = tf.Session()
            #print(sess.run(h_flat))
            #即把卷积神经网络最后输出的结果展成一行数据。
            #************这是我们拆开时想要的部分********************************
            #************这是我们拆开时想要的部分********************************
            #************这是我们拆开时想要的部分********************************
            #************这是我们拆开时想要的部分********************************
            #************这是我们拆开时想要的部分********************************
            #************这是我们拆开时想要的部分********************************

            prob1 = tf.constant(1.0, name='keep_prob_default')
            keep_prob = tf.placeholder_with_default(prob1, shape=(),
                                                    name='keep_prob')
            #调用全连接网络函数，将卷积神经网络的最终输出作为输入，放入全连接网络。
            h_fcl = feedforward(h_flat, dense_sizes, keep_prob=keep_prob)
            
            
        

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

            with tf.variable_scope('L2_cost'):
                # sum over all weights
                all_weights = [
                    graph.get_tensor_by_name('convolution/conv%s/w:0' % i)
                    for i in range(len(conv_channels))
                ] + [
                    graph.get_tensor_by_name('fully_connected/fc%s/w:0' % i)
                    for i in range(len(dense_sizes))
                ] + [w]
                
                #L2正则化
                l2 = lmbda * tf.reduce_sum([tf.reduce_sum(tf.pow(wi, 2))
                                            for wi in all_weights])
                #第一个求和是对一个batch里面的all_weights个权重求和，第二个求和是对一个pocth的
                
            #损失函数cost
            cost = tf.add(mse, l2, name='cost')

            optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer')
            train = optimizer.minimize(cost, global_step=global_step,
                                       name='train')
    #损失函数优化器的minimize()中global_step=global_steps能够提供global_step自动加一的操作。
    
   

    graph.add_to_collection('output', y)
    graph.add_to_collection('input', x)
    graph.add_to_collection('target', t)
    graph.add_to_collection('kp', keep_prob)
    #add_to_collectio为Graph的一个方法，可以简单地认为Graph下维护了一个字典，
    #key为name,value为list，而add_to_collection就是把变量添加到对应key下的list中

    return graph

#定义画柱状图的函数
def custom_summary_histogram(values, num_bins=200):
    print('37_&&&&&&&&&&&&&&&')
    """Create custom summary histogram for given values.
    This function returns tf.HistogramProto object which can be used as a value
    for tf.Summary, e.g.:

    >>> summary = tf.Summary()
    >>> summary.value.add(tag='my_summary',
    ...                   histo=custom_summary_histogram(values, num_bins=200))


    Parameters
    ----------
    values: np.ndarray
        Values to summarize
    num_bins: int, optional
        Number of bins in the histogram


    Returns
    -------
    histogram: tf.HistogramProto
        tf.HistogramProto object with a histogram.
    """

    if not isinstance(values, np.ndarray):
        raise TypeError('values must be an array, %s was given'
                        % type(values))
    if values.dtype.kind not in ['f', 'i']:
        raise ValueError('values must be floats, %s was given'
                         % values.dtype)

    if not isinstance(num_bins, int):
        raise TypeError('num_bins must be int, %s was given'
                        % type(num_bins))
    if num_bins <= 0:
        raise ValueError('num_bins must be positive, %s was given'
                         % num_bins)

    flat = values.flatten()
    hist, bins = np.histogram(flat, bins=num_bins)

    bins_middle = (bins[:-1] + bins[1:]) / 2

    histogram = tf.HistogramProto(min=flat.min(), max=flat.max(),
                                  num=len(flat), sum=flat.sum(),
                                  sum_squares=(flat ** 2).sum(),
                                  bucket_limit=bins_middle, bucket=hist)

    return histogram


#这是画3个set的real和prediction的散点图的
#####################################################
def custom_summary_image(mpl_figure):
    print('38_&&&&&&&&&&&&&&&')

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
#####################################################
    
#定义画特征重要性的函数
def feature_importance_plot(values, labels=None):
    print('39_&&&&&&&&&&&&&&&')
    """Create summary image with bar plot of feature importance.

    Parameters
    ----------
    values: array-like, shape = (N,)
        Values to plot
    labels: array-like, shape = (N,), optional
        Labels associated with the values. If not given, "F1", "F2", etc are
        used as labels.

    Returns
    -------
    image: tf.Summary.Image
        tf.Summary.Image object with a bar plot.
    """

    if not isinstance(values, (list, tuple, np.ndarray)):
        raise TypeError('values must be a 1D sequence')

    try:
        values = np.asarray(values, dtype='float')
    except:
        raise ValueError('values must be a 1D sequence of numbers')

    if values.ndim != 1:
        raise ValueError('values must be a 1D sequence of numbers')

    if labels is None:
        labels = ['F%s' % i for i in range(len(values))]
    elif not isinstance(labels, (list, tuple, np.ndarray)):
        raise TypeError('labels must be a 1D sequence')

    if len(values) != len(labels):
        raise ValueError('values and labels must have equal lengths')

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(y=labels, x=values, ax=ax)
    fig.tight_layout()

    image = custom_summary_image(fig)
    plt.close(fig)

    return image

#定义
def make_summaries_SB(graph, feature_labels=None):
    print('40_&&&&&&&&&&&&&&&')
    """Create summaries for network created with `make_SB_network`"""

    with graph.as_default():
        with tf.variable_scope('net_properties'):
            # weights between input and the first layer
            wconv0 = graph.get_tensor_by_name('convolution/conv0/w:0')#在graph中根据定义的名字，提取对应的变量w。
            #print('wconv0', wconv0) #conv0/w:0", shape=(5, 5, 5, 19, 64), dtype=float32_ref)
            in_chnls = wconv0.shape[-2].value
            #print('in_chnls', in_chnls) #in_chnls 19
            if feature_labels is None:
                feature_labels = ['F%s' % i for i in range(in_chnls)]
                #print('feature_labels', feature_labels) #feature_labels ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18']
            else:
                assert in_chnls == len(feature_labels)
            #特征权重的定义
            feature_weights = tf.split(wconv0, in_chnls, axis=3)
            #print('feature_weights', feature_weights) #[<tf.Tensor 'net_properties/split:0' shape=(5, 5, 5, 1, 64) dtype=float32>, <tf.Tensor 'net_properties/split:1' shape=(5, 5, 5, 1, 64) dtype=float32>,
            #特征重要性的定义,即第一层输入到第一层输出之间的卷积核中的值相加。
#            feature_importance = tf.reduce_sum(tf.abs(wconv0), 
#                                               reduction_indices=[0, 1, 2, 4],
#                                               name='feature_importance')
            

        net_summaries = tf.summary.merge((
            tf.summary.histogram('weights', wconv0),
            *(tf.summary.histogram('weights_%s' % name, value)
              for name, value in zip(feature_labels, feature_weights)),
            tf.summary.histogram('predictions', graph.get_tensor_by_name('output/prediction:0'))
        ))

        training_summaries = tf.summary.merge((
            tf.summary.scalar('mse', graph.get_tensor_by_name('training/mse:0')),
            tf.summary.scalar('cost', graph.get_tensor_by_name('training/cost:0'))
        ))

    return net_summaries, training_summaries

#指定一个文件用来保存图。
class SummaryWriter():
    print('41_&&&&&&&&&&&&&&&')

    def __init__(self, *args, **kwargs):
        """Context manager for tf.summary.FileWriter"""
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        self.writer = tf.summary.FileWriter(*self.args, **self.kwargs)
        return self.writer

    def __exit__(self, *args):
        self.writer.close()
