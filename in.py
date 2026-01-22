import numpy as np
import tensorflow as tf 

# making a simple dense layer

class DenseLayer:
    def __init__(self,in_size,out_size,activation):
        self.weights=np.random.randn(in_size,out_size)*0.01
        self.biases=np.zeros((1,out_size))
        self.activation=activation

    def __call__(self,input_data):
        return self.activation(np.matmul(input_data,self.weights)+self.biases)
    
    @property
    def weights_(self):
        return [self.weights,self.biases]
    
# a simple sequential class
class naivesequencial:
    def __init__(self,layers):
        self.layers=layers

    def __class__(self,x):
        for layer in self.layers:
            x=layer(x)
    
    @property
    def waights(self):
        waights=[]
        for layer in self.layers:
            waights+=layer.waights
            return waights
        
model=([
    DenseLayer(in_size=28*28,out_size=16,activation=tf.nn.rrelu),
    DenseLayer(in_size=16,out_size=10,activation=tf.nn.softmax)
])

# making a batch genrater

import math 
class bachgenrater:
    def __init__(self,images,labels,batch_size=128):
        assert len(images)==len(labels)
        self.index=0
        self.images=images
        self.labels=labels
        self.batch_size=batch_size
        self.num_batchs=math.ceil(len(images)/batch_size)

        def next(self):
            images=self.images[self.index:self.index+self.batch_size]
            labels=self.labels[self.index:self.index+self.batch_size]   
            self.index+=self.batch_size
            return images,labels
        
def one_traning_step(model,image_batch,labels_batch):
    # doing forourd pass for in GradientTape
    with tf.GradientTape() as tape:
        prad=model(image_batch)
        pre_sample_loss=tf.losses.SparseCategoricalCrossentropy(labels_batch,prad)
        ave_loss=tf.reduce_mean(pre_sample_loss)

        # computing Gradient using tape
        grad=tape.gradient(ave_loss,model.weights)
        # update the model weights
        updete_weights(grad,model.weights)
        return ave_loss

learning_ratr=1e-3
def updete_waights(grad,weights):
    for g,W in zip(grad,weights):
        W.assign_sub(g*learning_ratr)


