from forse.tools.nn_tools import *
from forse.tools.img_tools import *
from forse.tools.mix_tools import *
from keras.models import Sequential, Model, load_model
from keras.layers import UpSampling2D, Conv2D, Activation, BatchNormalization
from keras.layers import Reshape, Dense, Input
from keras.layers import LeakyReLU, Dropout, Flatten, ZeroPadding2D
from keras.optimizers import Adam
from keras import losses
import numpy as np
import os
from keras import backend as K

# Modified from: 
# https://github.com/eriklindernoren/Keras-GAN.git

class DCGAN:
     def __init__(self, output_directory, img_size):
         self.img_size = img_size
         self.channels = 1
         self.kernel_size = 5
         self.output_directory = output_directory

     def smooth_accuracy(self, y_true, y_pred):
         return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

     def build_generator(self):
         img_shape = (self.img_size[0], self.img_size[1], self.channels)
         model = Sequential()
         model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same")) # 64x64x64
         model.add(LeakyReLU(alpha=0.2))
         model.add(BatchNormalization(momentum=0.5))
         model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same", strides=2)) #32x32x128
         model.add(LeakyReLU(alpha=0.2))
         model.add(BatchNormalization(momentum=0.5))
         model.add(Conv2D(256, kernel_size=self.kernel_size, padding="same", strides=2)) #16x16x256
         model.add(LeakyReLU(alpha=0.2))
         model.add(BatchNormalization(momentum=0.5))
         model.add(UpSampling2D())
         model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
         model.add(LeakyReLU(alpha=0.2))
         model.add(BatchNormalization(momentum=0.5))
         model.add(UpSampling2D())
         model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
         model.add(LeakyReLU(alpha=0.2))
         model.add(BatchNormalization(momentum=0.5))
         model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
         model.add(Activation("tanh"))
         img_in = Input(shape=img_shape)
         img_out = model(img_in)
         return Model(img_in, img_out)

     def build_discriminator(self):
         img_shape = (self.img_size[0], self.img_size[1], self.channels)
         model = Sequential()
         model.add(Conv2D(64, kernel_size=self.kernel_size, strides=1, input_shape=img_shape, padding="same"))
         model.add(LeakyReLU(alpha=0.2))
         model.add(BatchNormalization(momentum=0.5))
         model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))
         model.add(BatchNormalization(momentum=0.5))
         model.add(Conv2D(256, kernel_size=self.kernel_size, strides=2, padding="same"))
         model.add(LeakyReLU(alpha=0.2))
         model.add(Flatten())
         model.add(Dropout(0.25))
         model.add(Dense(1, activation='sigmoid'))
         img = Input(shape=img_shape)
         validity = model(img)
         return Model(img, validity)

     def build_gan(self):
         img_shape = (self.img_size[0], self.img_size[1], self.channels)
         optimizer = Adam(0.0002, 0.5)
         self.discriminator = self.build_discriminator()
         self.discriminator.compile(loss='binary_crossentropy',
                                        optimizer=optimizer,
                                        metrics=['accuracy'])
         self.generator = self.build_generator()
         self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
         z = Input(shape=img_shape)
         img = self.generator(z)
         self.discriminator.trainable = False
         valid = self.discriminator(img)
         self.combined = Model(z, valid)
         self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

     def train(self, epochs, patches_file, batch_size=32, save_interval=100, seed=4324):
         self.build_gan()
         X_train, X_test, Y_train, Y_test = load_training_set(patches_file, seed=seed)
         print("Training Data Shape: ", X_train.shape)
         half_batch = batch_size // 2
         accs = []
         self.discriminator.summary()
         for epoch in range(epochs):
             ind_batch = np.random.randint(0, X_train.shape[0], batch_size)
             g_loss = self.combined.train_on_batch(X_train[ind_batch], np.ones((batch_size, 1)))
             target_real = np.ones((half_batch, 1))
             target_fake = np.zeros((half_batch, 1))
             idxX = np.random.randint(0, X_train.shape[0], half_batch)
             idxY = np.random.randint(0, X_train.shape[0], half_batch)
             imgs = Y_train[idxY]
             gen_imgs = self.generator.predict(X_train[idxX])
             d_loss_real = self.discriminator.train_on_batch(imgs, target_real)
             d_loss_fake = self.discriminator.train_on_batch(gen_imgs, target_fake)
             if epoch % (save_interval) == 0:
                 print(epoch)
                 gen_imgs_test = self.generator.predict(X_test)
                 save_path = self.output_directory + "/models"
                 if not os.path.exists(save_path):
                     os.makedirs(save_path)
                 self.discriminator.save(save_path + '/discrim_'+str(epoch)+'.h5')
                 self.generator.save(save_path + '/generat_'+str(epoch)+'.h5')
         self.discriminator.save(save_path + '/discrim_'+str(epoch)+'.h5')
         self.generator.save(save_path + '/generat_'+str(epoch)+'.h5')
