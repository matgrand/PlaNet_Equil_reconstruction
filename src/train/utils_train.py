import numpy as np
import tensorflow as tf

def gauss_kernel(size=3):
    if size == 3: gaussian_ker = np.array(([1,2,1], [2,4,2], [1,2,1]), dtype='float32')/16
    elif size == 5: gaussian_ker = np.array(([1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]), dtype='float32')/273
    else: raise NotImplementedError 
    gaussian_ker = tf.expand_dims(tf.expand_dims(gaussian_ker[::-1, ::-1],axis = -1),axis = -1)
    return gaussian_ker

def fun_GSoperator_NN_conv_smooth_batch_adaptive(f,laplace_ker_ds,df_dr_ker_ds,gaussian_ker,RR_ds,ZZ_ds):
    f = tf.transpose(f,[3,1,2,0])
    Lpsi = tf.nn.depthwise_conv2d(f,tf.transpose(tf.expand_dims(laplace_ker_ds, axis=-1),[1,2,0,3]),strides=[1, 1, 1, 1],padding='VALID')
    Lpsi = tf.transpose(Lpsi,[3,1,2,0]) # no need to be transposed becaused Laplacian filter is left/rigth symmetric
    Dpsi_dr = tf.nn.depthwise_conv2d(f,tf.transpose(tf.expand_dims(df_dr_ker_ds, axis=-1), [1,2,0,3]),strides=[1, 1, 1, 1],padding='VALID')
    Dpsi_dr = - Dpsi_dr # necessary because nn.depthwise_conv2d filters has to be transposed to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    Dpsi_dr = tf.transpose(Dpsi_dr,[3,1,2,0])
    RR_in = tf.expand_dims(RR_ds[:,1:-1,1:-1],axis=-1)
    Dpsi_dr = tf.math.divide(Dpsi_dr,RR_in)
    GS_ope = Lpsi - Dpsi_dr
    hr = RR_ds[:,1,2] - RR_ds[:,1,1]
    hz = ZZ_ds[:,2,1] - ZZ_ds[:,1,1]
    α = -2*(hr**2 + hz**2)
    hr = tf.expand_dims(tf.expand_dims(tf.expand_dims(hr,axis=-1),axis=-1),axis=-1)
    hz = tf.expand_dims(tf.expand_dims(tf.expand_dims(hz,axis=-1),axis=-1),axis=-1)
    α = tf.expand_dims(tf.expand_dims(tf.expand_dims(α,axis=-1),axis=-1),axis=-1)
    GS_ope = GS_ope*α/(hr**2*hz**2)
    GS_ope = tf.nn.conv2d(GS_ope,gaussian_ker,strides=[1, 1, 1, 1],padding='SAME')
    GS_ope = tf.squeeze(GS_ope,axis = -1)
    return GS_ope

def calc_laplace_df_dr_ker(hr, hz):
    α = -2*(hr**2 + hz**2)
    laplace_ker = np.array(([0, hr**2/α, 0], [hz**2/α, 1, hz**2/α], [0, hr**2/α, 0]))
    df_dr_ker = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr*α)*(hr**2*hz**2)
    return laplace_ker, df_dr_ker