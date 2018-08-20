
import tensorflow as tf
from keras.layers import Input, Lambda,Conv2D,Add,Subtract,BatchNormalization,Reshape
import numpy as np
import keras.backend as K
from keras import Model
from keras.activations import softmax
from keras.initializers import RandomNormal



def hw_flatten(z):
    return  K.reshape(z,shape=( K.int_shape(z)[0],-1,K.int_shape(z)[-1]))



def attention_dense(x,ch,gamma):
    f = Conv2D(ch // 8, (1,1),padding='valid') (x) # [bs, h, w, c']
    g = Conv2D(ch // 8, (1,1),padding='valid') (x)# [bs, h, w, c']
    h =  Conv2D(ch , (1,1),padding='valid') (x) # [bs, h, w, c]
    #f = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
    #g = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
    #h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]

    # N = h * w
    print (K.int_shape(K.transpose(hw_flatten(f)) ))
    s =tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
    #s =K.dot(hw_flatten(g), K.transpose(hw_flatten(f)) ) # # [bs, N, N]
    beta = softmax(s, axis=-1)  # attention map

    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]


    o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
    x = gamma*o + x

    return x




def create_label(ch,H,W):
    label = []
    label.append(np.zeros(shape=[1,H,W,ch]))
    label.append(np.zeros(shape=[1,ch,ch]))
    label.append (y)
    return label

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def get_grams(x):
    grams = []
    #global batch_size
    for i in range(1):
        gram = gram_matrix(x[i])
        gram = K.expand_dims(gram, axis=0)
        grams.append(gram)
    return K.concatenate(grams, axis=0)


def build_model(x,y,ch,gamma):
    x_conv= Conv2D(ch,(1,1),padding='same')(x)
    y_conv= Conv2D(ch,(1,1),padding='same')(y)


    x_att = Lambda(attention_dense,arguments = {'ch':ch,'gamma':gamma})(x_conv)
    y_att = Lambda(attention_dense,arguments = {'ch':ch,'gamma':gamma})(y_conv)
    out_att = Subtract()([x_att,y_att])
    print('out_att', K.int_shape(out_att))
    x_g = Lambda(get_grams)(x_att)
    y_g = Lambda(get_grams)(y_att)
    out_gram = Subtract()([x_g,y_g])

    print('out_gram', K.int_shape(out_gram))

    x_out = Conv2D(3,(1,1),padding='same')(x_att)
    print('x_out', K.int_shape(x_out))
    model = Model([x,y],[out_att,out_gram,x_out])
    model.compile(loss='mse',optimizer='Adam')
    model.summary()


    return model




if __name__ ==  '__main__':
    H=28
    W=28
    ch = 1024
    batch_size =1
    x =Input(batch_shape=(1,H,W,3))
    y =Input(batch_shape=(1,H,W,3))
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))


    model = build_model(x,y,ch,gamma)

    X_data = np.random.normal(size=[1,H,W,3])
    y_data = np.random.normal(size=[1,H,W,3])


    label = create_label(ch,H,W)


    model.fit([X_data,y_data], label , epochs=200)
    print ('\n\n')
    print(model.predict([X_data, y_data]))
