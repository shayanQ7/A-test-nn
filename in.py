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
    DenseLayer(in_size=28*28,out_size=16,activation=tf.nn.relu),
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

learning_ratr=1e-3
def updete_waights(grad,weights):
    for g,W in zip(grad,weights):
        W.assign_sub(g*learning_ratr)


def one_traning_step(model,image_batch,labels_batch):
    # doing forourd pass for in GradientTape
    with tf.GradientTape() as tape:
        prad=model(image_batch)
        pre_sample_loss=tf.losses.SparseCategoricalCrossentropy(labels_batch,prad)
        ave_loss=tf.reduce_mean(pre_sample_loss)

        # computing Gradient using tape
        grad=tape.gradient(ave_loss,model.weights)
        # update the model weights
        updete_waights(grad,model.weights)
        return ave_loss


def fit(model,images,labels,epochs=5,batch_size=128):
    for epoch in range(epochs):
        batch_genrater=bachgenrater(images,labels,batch_size)
        for batch_index in range(batch_genrater.num_batchs):
            image_batch,label_batch=batch_genrater.next()
            loss=one_traning_step(model,image_batch,label_batch)
            if batch_index %100==0:
                print(f"Epoch {epoch+1},Batch {batch_index},Loss:{loss.numpy():.4f}")


# example usage with mnist dataset
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(-1,28*28).astype("float32")/255.0
x_test=x_test.reshape(-1,28*28).astype("float32")/255.0     

fit(model,x_train,y_train,epochs=5,batch_size=128)
