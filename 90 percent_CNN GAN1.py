# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:05:29 2022

@author: ahmed
"""
import numpy as np
import os
import tensorflow as tf
import cv2
from  matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D, Input, InputLayer,Reshape, Conv2DTranspose
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import backend 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import MaxNorm
from sklearn.model_selection import KFold, StratifiedKFold
from numpy import std
from numpy import mean
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from os.path import exists
from tensorflow.keras.utils import plot_model
import sys
from os import makedirs

# from numpy import expand_dims
from numpy import zeros
from numpy import ones
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import LeakyReLU, BatchNormalization, ReLU
# from keras.utils.vis_utils import plot_model
from numpy.random import randn, randint
from matplotlib import pyplot
import matplotlib.image as mpimg

# ============================= Loading Spectrum Images ========================
# defining the input images size    
IMG_WIDTH=64
IMG_HEIGHT=64
subject = "sub_B04"
n_epochs = 400
cnn_batch_size = 9 
cnn_epochs = 500
Ad_times = 1
nfolds = 0
R_nfolds = [ 7,9 ]
seed = 7

subjects = ['sub_B06' , 'sub_B07' , 'sub_B08', 'sub_B09' ]    # , 'sub_B02' , 'sub_B03' , 'sub_B04', 'sub_B05' , 'sub_B06' , 'sub_B07' , 'sub_B08', 'sub_B09', 
cnn_acc = list()
GAN_acc = list()
cnn2_acc = list()
GAN2_acc = list()
total_cnn_acc=list()
total_gan2_acc=list()

img_folder =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram\sec_4\{}'.format(subject) 
img_folder_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram\sec_4\Test\{}'.format(subject)   

def create_dataset(img_folder):       
    img_data_array=[]
    class_name=[]       
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image=mpimg.imread(image_path)
            # image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    # extract the image array and class name
    (img_data, class_name) = (img_data_array,class_name)
    # Create a dictionary for all unique values for the classes
    target_dict={s: v for v, s in enumerate(np.unique(class_name))}
    target_dict
    # Convert the class_names to their respective numeric value based on the dictionary
    target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
    x=tf.cast(np.array(img_data), tf.float64).numpy()
    y=tf.cast(list(map(int,target_val)),tf.int32).numpy()
    return x, y

x_tr,y_tr = create_dataset(img_folder)
x_ev,y_ev = create_dataset(img_folder_test)
x0 = np.concatenate(( x_tr,x_ev ))   
y0 = np.concatenate(( y_tr,y_ev ))  


no_copy = 10 - (len(x0) % 10)
x_copy = np.empty(( no_copy , IMG_HEIGHT, IMG_WIDTH,3)) 
y_copy = np.empty((no_copy ))
for s in range(no_copy):
    x_copy[s] = x0[s]
    y_copy[s] = y0[s]

x =  np.concatenate((x0, x_copy))
y = np.concatenate(( y0, y_copy))

def fold_split(xdata,ydata,folds=10): 
    
    trainX = np.empty((int(folds),int (len(xdata)-(len(xdata)/folds)) , IMG_HEIGHT, IMG_WIDTH,3))
    trainY = np.empty((int(folds),int (len(xdata)-(len(xdata)/folds)) ))
    testX = np.empty((int(folds),int (len(xdata)/folds) , IMG_HEIGHT, IMG_WIDTH,3))
    testY = np.empty((int(folds),int (len(xdata)/folds) ))

    sub_fold = StratifiedKFold(folds, shuffle=True, random_state=2) 
    i=0
    # ## enumerate splits
    for train, cv in sub_fold.split(xdata,ydata):
        # select data for train and test
        trainX[i,:,:,:,:], trainY[i,:], testX[i,:,:,:,:], testY[i,:] = xdata[train], ydata[train], xdata[cv], ydata[cv]
        i+=1
    return trainX, trainY, testX, testY

# for subject in subjects:
#     img_folder =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram\sec_4\{}'.format(subject) 
#     img_folder_test =r'D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_2b GAN\spectrogram\sec_4\Test\{}'.format(subject) 
for nfolds in range(2,3):
# for nfolds in R_nfolds:
    # fix random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print (subject)
    print (nfolds)
   
    
    trainX, trainY, testX, testY = fold_split(x, y)

    def class_imgs(x,y):
        class_count = 0
        for i in range( len(y[nfolds]) ):
                if (y[nfolds,i] == 1 ):
                    class_count += 1
        cl1 = x[nfolds,0:class_count, :,:]
        cl2 = x[nfolds,class_count:, :,:]
        return cl1,cl2
        
    x_img_cl1,x_img_cl2 = class_imgs(trainX, trainY)
    x_test_img_cl1, x_test_img_cl2 = class_imgs(testX, testY)
    
    n_batch = (len(x_img_cl1))//2
    print("batch: ",n_batch)
    

    #%% ================================== GAN ===========================
    makedirs('final_90/{0} GAN_results/fold_{1}/plots {0}'.format(subject,nfolds), exist_ok=True) 
    makedirs('final_90/{0} GAN_results/fold_{1}/models {0}'.format(subject,nfolds), exist_ok=True) 
    def define_discriminator(in_shape=(64,64,3)):
        model = Sequential(name="discriminator")
        # normal
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape, name='conv_1'))
        model.add(LeakyReLU(alpha=0.2, name='Leaky_ReLU_1'))
        # downsample
        model.add(Conv2D(256, (3,3), strides=(2,2), padding='same', name='conv_2'))
        model.add(LeakyReLU(alpha=0.2, name='Leaky_ReLU_2'))
        # downsample
        model.add(Conv2D(512, (3,3), strides=(2,2), padding='same', name='conv_3'))
        model.add(LeakyReLU(alpha=0.2, name='Leaky_ReLU_3'))
        # downsample
        model.add(Conv2D(1024, (3,3), strides=(2,2), padding='same', name='conv_4'))
        model.add(LeakyReLU(alpha=0.2, name='Leaky_ReLU_4'))
    
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
    
    # define the standalone generator model
    def define_generator(latent_dim=100):
        model = Sequential(name="generator")
        model.add(Input(shape= latent_dim  , name='input_layer'))
        model.add(Reshape((1, 1, latent_dim)))

        # n_nodes = 256 * 4 * 4
        # model.add(Dense(n_nodes, input_dim=latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Reshape((4, 4, 256)))

        model.add(Conv2DTranspose(1024, 4, strides=4, padding='same', name='deconv_1'))
    
        # upsample to 8x8
        model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', name='deconv_2'))
        model.add(LeakyReLU(alpha=0.2, name='Leaky_ReLU_1'))
        # upsample to 16x16
        model.add(Conv2DTranspose(512, (4,4), strides=2, padding='same', name='deconv_3'))
        model.add(LeakyReLU(alpha=0.2, name='Leaky_ReLU_2'))
        # upsample to 32x32
        model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', name='deconv_4'))
        model.add(LeakyReLU(alpha=0.2, name='Leaky_ReLU_3'))
        # upsample to 64x64
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', name='deconv_5'))
        model.add(LeakyReLU(alpha=0.2, name='Leaky_ReLU_4'))
        # output layer
        model.add(Conv2D(3, (3,3), activation='tanh', padding='same', name='output_layer'))
        # model.add(Conv2DTranspose(3, (3,3), strides=2, activation='tanh', padding='same'))
        return model
    #=================================================
    # define discriminator model
    model = define_discriminator()
    # summarize the model
    model.summary()
    # plot the model
    plot_model(model, to_file='final_90/{0} GAN_results/fold_{1}/GAN_discriminator_plot.png'.format(subject, nfolds), show_shapes=True, show_layer_names=True)
    # define the generator model
    model = define_generator()
    # summarize the model
    model.summary()
    # plot the model
    plot_model(model, to_file='final_90/{0} GAN_results/fold_{1}/GAN_generator_plot.png'.format(subject, nfolds), show_shapes=True, show_layer_names=True)
    
    # define the combined generator and discriminator model, for updating the generator
    def define_gan(g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect them
        model = Sequential(name="GAN")
        # add generator
        model.add(g_model)
        # add the discriminator
        model.add(d_model)
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.8) #, beta_2 = 0.8
        # opt = SGD(learning_rate=0.002)

        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
    
    # load and prepare training images
    def load_real_samples(cl):
        if cl==1:
            X = x_img_cl1
            #********** GAN Batch number *********
            n_batch = (len(x_img_cl1))//2
           
        else:
            X = x_img_cl2
            cl=2
            n_batch = (len(x_img_cl2))//2            
        return X,cl
    
    # select real samples
    def generate_real_samples(dataset, n_samples):
        # choose random instances
        ix = randint(0, dataset.shape[0], n_samples)
        # retrieve selected images
        X = dataset[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, 1))
        return X, y
    
    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input
    
    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(g_model, latent_dim, n_samples):
        # generate points in latent space
        x_input = generate_latent_points(latent_dim, n_samples)
        # predict outputs
        X = g_model.predict(x_input)
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return X, y  
    
    # create and save a plot of generated images
    def save_plot(examples, epoch, n=4):
        # scale from [-1,1] to [0,1]
        # examples = (examples + 1) / 2.0
        # plot images
        plt.figure(figsize=(64,64))
        for i in range(n * n):
            # define subplot
            pyplot.subplot(n, n, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(examples[i])
        # save plot to file
        pyplot.savefig('final_90/{0} GAN_results/fold_{1}/plots {0}/test2_GAN_batch{2}_cl{3}_{4}.png' .format(subject,nfolds, n_batch, cl, epoch+1))
        pyplot.close()
    
    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(epoch, g_model, latent_dim, n_samples=100):
        # prepare real samples
        X_real, y_real = generate_real_samples(dataset, n_samples)
        # evaluate discriminator on real examples
        _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
        # save plot
        save_plot(x_fake, epoch)
        # save the generator model tile file
        g_model.save('final_90/{0} GAN_results/fold_{1}/models {0}/test2_GAN_batch{2}_cl{3}_{4}.h5'.format(subject,nfolds, n_batch, cl, epoch+1))
    
    # create a line plot of loss for the gan and save to file
    def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.plot(d1_hist, label='d-real')
        pyplot.plot(d2_hist, label='d-fake')
        pyplot.plot(g_hist, label='gen')
        pyplot.legend()
        # plot discriminator accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.plot(a1_hist, label='acc-real')
        pyplot.plot(a2_hist, label='acc-fake')
        pyplot.legend()
        # save plot to file
        pyplot.grid()
        pyplot.savefig('final_90/{0} GAN_results/fold_{1}/test2_plot_line_GAN {2}_cl{3}_loss.png' .format(subject,nfolds, n_batch, cl))
        pyplot.close()
    
    # train the generator and discriminator
    def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs= n_epochs, n_batch= n_batch):
        # calculate the number of batches per epoch
        bat_per_epo = int(dataset.shape[0] / n_batch)
        # calculate the total iterations based on batch and epoch
        n_steps = bat_per_epo * n_epochs
        # calculate the number of samples in half a batch
        half_batch = int(n_batch / 2)
        # prepare lists for storing stats each iteration
        d_r_hist, d_f_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                X_real, y_real = generate_real_samples(dataset, half_batch)
                # update discriminator model weights
                d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
                # generate 'fake' examples
                X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
                # update discriminator model weights
                d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
                # prepare points in latent space as input for the generator
                X_gan = generate_latent_points(latent_dim, n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
            # record history
            d_r_hist.append(d_loss1)
            d_f_hist.append(d_loss2)
            g_hist.append(g_loss)
            a1_hist.append(d_acc1)
            a2_hist.append(d_acc2)
            # evaluate the model performance, everry batch
            if (i+1) % 50 == 0 and (i+1) >= 200:
                summarize_performance(i, g_model, latent_dim)
        plot_history(d_r_hist, d_f_hist, g_hist, a1_hist, a2_hist)
             
    # load image data
    dataset,cl = load_real_samples(1)
    print(dataset.shape)
    # size of the latent space
    latent_dim = 200
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    
    #%% train model
    train(g_model, d_model, gan_model, dataset, latent_dim)
    
    dataset,cl = load_real_samples(2)
    print(dataset.shape)
    # size of the latent space
    latent_dim = 200
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    
    train(g_model, d_model, gan_model, dataset, latent_dim)   
    
#%% ============================= Genrator =====================================
    # example of loading the generator model and generating images
    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input
    
    # plot sample the generated images
    def create_plot(examples, n):
        # plot images
        for i in range(n * n):
            # define subplot
            plt.subplot(n, n, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i, :, :])
        plt.show()
    
    cl=1
    epochs=  n_epochs         
    batch = n_batch
    # make folder for results
    makedirs('final_90/{0} GAN_results/fold_{1}/GAN_dataset/CL{2}'.format(subject,nfolds, cl), exist_ok=True) 
        
    # create GAN  images    
    def create_GAN_plot(examples):
        # plot images
        plt.figure(figsize=(64,64))
        for i in range(len(examples)):
            # define subplot
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i])
            plt.savefig('final_90/{0} GAN_results/fold_{1}/GAN_dataset/CL{2}/cl{2}_t{3}.jpg'.format(subject,nfolds , cl , i), bbox_inches= 'tight', pad_inches= 0)
            plt.close()
        
    # load model
    model = load_model('final_90/{0} GAN_results/fold_{1}/models {0}/test_GAN_batch{2}_cl{3}_{4}.h5'.format(subject,nfolds, batch, cl, epochs))
    # generate images
    latent_points = generate_latent_points(200,len(x_img_cl1) * Ad_times )   
    # generate images
    X_gan_lat = model.predict(latent_points)    
    # plot the result
    create_GAN_plot(X_gan_lat)
    
    cl=2
    tf.random.set_seed(seed)
    np.random.seed(seed)
    epochs= n_epochs         
    batch = n_batch
    makedirs('final_90/{0} GAN_results/fold_{1}/GAN_dataset/CL{2}'.format(subject, nfolds, cl), exist_ok=True) 
    model = load_model('final_90/{0} GAN_results/fold_{1}/models {0}/test_GAN_batch{2}_cl{3}_{4}.h5'.format(subject, nfolds, batch, cl, epochs))
    # generate images
    latent_points = generate_latent_points(200, len(x_img_cl2) *  Ad_times)
    # generate images
    X_gan_lat = model.predict(latent_points)    
    # plot the result
    create_GAN_plot(X_gan_lat)
    
    #% ============================= GAN DATA Loading ============================================
    GAN_data= r'final_90/{0} GAN_results/fold_{1}/GAN_dataset'.format(subject, nfolds)  
    
    # ======================= GAN DATA ============================
    x_gan, y_gan =create_dataset(GAN_data)
   
    RD_AD = len(trainX[nfolds])  + Ad_times* len(trainX[nfolds]) 
    x_train = np.empty((int (RD_AD) , IMG_HEIGHT, IMG_WIDTH,3))
    y_train = np.empty((int (RD_AD) ))
    
    x_train = np.concatenate((trainX[nfolds], x_gan[0: Ad_times*(len(trainX[nfolds])//2)], x_gan[len(x_gan)//2: len(x_gan)//2 + Ad_times*( len(trainX[nfolds])//2) ] ))   
    y_train = np.concatenate((trainY[nfolds], y_gan[0: Ad_times*(len(trainY[nfolds])//2)], y_gan[len(y_gan)//2: len(y_gan)//2 + Ad_times*( len(trainY[nfolds])//2) ] ))  
    print ("Subject Data= ",len(trainX[nfolds]) )    

    #%% ================================== CNN model =================================
    Drp1= 2
    Drp2= 2
    Drp3= 4
    Drp4= 4
       
    def create_cnn(): #dropout_rate1=0.0, dropout_rate2=0.0 , momentum=0, weight_constraint=0, 
        model= Sequential()
        model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,3)) )     # dropout 1
        model.add(Conv2D(8, kernel_size=(3,3), padding='same', activation= 'relu' 
                          , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(Drp2/10))                                             # dropout 2
        model.add(Conv2D(8, (3,3), padding='same' ,activation= 'relu' 
                          , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   )) # kernel_regularizer=l2(0.001),
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(Drp3/10))                                             # dropout 3
        model.add(Flatten())
        # model.add(Dropout(0.2)) 
        model.add(Dense(100, activation= 'relu' , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3) ) )  
        model.add(Dropout(Drp4/10))                                             # dropout 4
        model.add(Dense(2, activation= 'softmax' , kernel_initializer='he_uniform' )) 
        opt = SGD(learning_rate=0.0001, momentum=0.99)
        # opt = Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.8) #, beta_2 = 0.8

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
        return model
    model = create_cnn()
    model.summary()
    
    def create_cnn2(): #dropout_rate1=0.0, dropout_rate2=0.0 , momentum=0, weight_constraint=0, 
        model= Sequential()
        model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,3)) )     # dropout 1
        model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation= 'relu' 
                          , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
        model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation= 'relu' 
                          , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(Drp2/10))                                             # dropout 2
        model.add(Conv2D(32, (3,3), padding='same' ,activation= 'relu' 
                          , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   )) # kernel_regularizer=l2(0.001),
        model.add(Conv2D(32, (3,3), padding='same' ,activation= 'relu' 
                          , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(Drp2/10))                                             # dropout 3
        model.add(Conv2D(64, (3,3), padding='same' ,activation= 'relu' 
                          , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(Drp3/10))           
        model.add(Flatten())
        model.add(Dense(128, activation= 'relu' , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3) ) )  
        model.add(Dropout(Drp3/10))                                             # dropout 4
        model.add(Dense(2, activation= 'softmax' , kernel_initializer='he_uniform' )) 
        # opt = SGD(learning_rate=0.0001, momentum=0.99)
        opt = Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.8) #, beta_2 = 0.8

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
        return model
    #%% ============================= Training ====================================
    # #GAN-CNN Model training:
    def model_training(x_data, y_data ,save_dir, sel_mod ,fig_title):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print ("Training Data= ",len(x_data) )    
        model = sel_mod
        # 1-Times generated data:
        mcg = ModelCheckpoint(save_dir, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
        history = model.fit(x_data, y_data, epochs=cnn_epochs , batch_size=cnn_batch_size, verbose=0 
                            ,validation_data=(testX[nfolds],testY[nfolds]), callbacks=[mcg] )
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(fig_title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower right')
        plt.grid()
        plt.show()
        plt.close()
        
    # model_training(x_train, y_train,'final_90/CNN_GAN/ep{3}_{0}_GAN_{1}_CNN2_model_f{2}_ad.h5'.format( subject, Ad_times, nfolds, epochs), create_cnn2(), '{1}_GAN_CNN2 Model accuracy fold {2} --> AD={0} \n'.format( Ad_times, subject, nfolds))        
    # model_training(x_train, y_train,'final_90/{0} GAN_results/fold_{2}/CNN_GAN/ep{3}_batch{4}_{0}_GAN_{1}_CNN_model.h5'.format( subject, Ad_times, nfolds), create_cnn(), '{1}_GAN_CNN Model accuracy fold {2}--> AD={0} \n'.format( Ad_times, subject, nfolds))    

    # model_training(trainX[nfolds], trainY[nfolds],'final_90/{0} GAN_results/fold_{1}/CNN/{0}_CNN_model.h5'.format( subject, nfolds), create_cnn(), '{0}_CNN Model accuracy fold {1} --  \n'.format(  subject, nfolds))        
    # model_training(trainX[nfolds], trainY[nfolds],'final_90/CNN_GAN/{0}_CNN2_model_f{1}_ad.h5'.format( subject, nfolds), create_cnn2(), '{0}_CNN2 Model accuracy fold {1} --  \n'.format(  subject, nfolds))        
   


#%%======================================= 10 fold test with different batch no. ==========================
#     def CNN_GAN_test(cnn=2):
#         scores = list()
    
#         # for f in range(0,10):
#         if cnn == 2 :
#         # load the saved model
#             model = load_model('final_90/CNN_GAN//ep{3}_batch{4}_{0}_GAN_{1}_CNN2_model_f{2}.h5'.format( subject, Ad_times, nfolds, epochs, batch))
#             test_loss, test_acc= model.evaluate(testX[nfolds],testY[nfolds],verbose=0)
#             print('GAN Test: ',nfolds,' fold Accuracy',test_acc)
#             scores.append(test_acc)
#             # model.save('final_90/CNN_GAN//ep{3}_batch{4}_{0}_GAN_{1}_CNN2_model_f{2}.h5'.format( subject, Ad_times, f, epochs, batch))
#         elif cnn == 1:
#             model = load_model('final_90/{0} GAN_results/fold_{2}/CNN_GAN/GAN_CNN models/{0}_GAN_{1}_CNN_model.h5'.format( subject, Ad_times, nfolds))
#             test_loss, test_acc= model.evaluate(testX[nfolds],testY[nfolds],verbose=0)
#             print('Test: ',nfolds,' fold Accuracy',test_acc)
#             scores.append(test_acc)
#         else:
#             model = load_model('final_90/CNN_GAN/ep{2}_batch{3}_{0}_CNN2_model_f{1}.h5'.format( subject, nfolds, epochs, batch))
#             test_loss, test_acc= model.evaluate(testX[nfolds],testY[nfolds],verbose=0)
#             print('Test: ',nfolds,'Test fold Accuracy',test_acc)
#             scores.append(test_acc)
#             # model.save('final_90/CNN_GAN//ep{2}_batch{3}_{0}_CNN2_model_f{1}.h5'.format( subject, f, epochs, batch))
    
#         # print('>>>> {0} folds Accuracy: mean={1} std={2}, n={3}' .format (subject, mean(scores)*100, std(scores)*100, len(scores)))
#         # print ('*************************************')
#         ## box and whisker plots of results
#         #plt.boxplot(scores)
#         #plt.show()
#         #plot_model(model, show_shapes=True, expand_nested=True)
#         return scores
    
#     # sub_gan_acc = CNN_GAN_test(cnn=1)
#     # total_gan_acc.append(sub_gan_acc)
#     cnn2_folds_acc = CNN_GAN_test(cnn=3)
    
#     GAN2_folds_acc = CNN_GAN_test(cnn=2)
#     # total_gan2_acc.append(sub_gan2_acc)
    
#     #  print('**** GAN Accuaracy: mean=%.3f std=%.3f, n=%d' % (mean(total_gan_acc)*100, std(total_gan_acc)*100, len(total_gan_acc)))
#     #  print('**** Enhancement: mean=%.3f std=%.3f' % (mean(GAN_acc)*100 - mean(cnn_acc)*100, std(GAN_acc)*100 - std(cnn_acc)*100))
    
#     # print('**** GAN2 Accuaracy: mean=%.3f std=%.3f, n=%d' % (mean(total_gan2_acc)*100, std(total_gan2_acc)*100, len(total_gan2_acc)))
# print('**** Enhancement: mean=%.3f std=%.3f' % (mean(GAN2_folds_acc)*100 - mean(cnn2_folds_acc)*100, std(GAN2_folds_acc)*100 - std(cnn2_folds_acc)*100))

#%%======================================= 10 fold test ==========================
# # for subject in subjects:
def CNN_GAN_test(cnn=2):
    scores = list()

    for f in range(0,2):
        if cnn == 2 :
        # load the saved model
            model = load_model('final_90/CNN_GAN/ep{3}_{0}_GAN_{1}_CNN2_model_f{2}.h5'.format( subject, Ad_times, f, epochs))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('\nTest: ',f,' fold Accuracy',test_acc)
            scores.append(test_acc)
            # model.save('final_90/CNN_GAN/{0}_GAN_{1}_CNN2_model_f{2}_.h5'.format( subject, Ad_times, f, epochs, batch))
        elif cnn == 1:
            model = load_model('final_90/{0} GAN_results/fold_{2}/CNN_GAN/GAN_CNN models/{0}_GAN_{1}_CNN_model.h5'.format( subject, Ad_times, f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('\nTest: ',f,' fold Accuracy',test_acc)
            scores.append(test_acc)
        else:
            model = load_model('final_90/CNN_GAN/{0}_CNN2_model_f{1}.h5'.format( subject, f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('\nTest: ',f,'CNN_ fold Accuracy',test_acc)
            scores.append(test_acc)
           

    print('\n >>>> {0} folds Accuracy: mean={1} std={2}, n={3}' .format (subject, mean(scores)*100, std(scores)*100, len(scores)))
    print ('*************************************')
    ## box and whisker plots of results
    #plt.boxplot(scores)
    #plt.show()
    #plot_model(model, show_shapes=True, expand_nested=True)
    return scores

# sub_gan_acc = CNN_GAN_test(cnn=1)
# total_gan_acc.append(sub_gan_acc)
# cnn2_folds_acc = CNN_GAN_test(cnn=3)

# GAN2_folds_acc = CNN_GAN_test(cnn=2)
# total_gan2_acc.append(sub_gan2_acc)

#  print('**** GAN Accuaracy: mean=%.3f std=%.3f, n=%d' % (mean(total_gan_acc)*100, std(total_gan_acc)*100, len(total_gan_acc)))
#  print('**** Enhancement: mean=%.3f std=%.3f' % (mean(GAN_acc)*100 - mean(cnn_acc)*100, std(GAN_acc)*100 - std(cnn_acc)*100))

# print('**** GAN2 Accuaracy: mean=%.3f std=%.3f, n=%d' % (mean(total_gan2_acc)*100, std(total_gan2_acc)*100, len(total_gan2_acc)))

# print('**** Enhancement: mean=%.3f std=%.3f' % (mean(GAN2_folds_acc)*100 - mean(cnn2_folds_acc)*100, std(GAN2_folds_acc)*100 - std(cnn2_folds_acc)*100))

