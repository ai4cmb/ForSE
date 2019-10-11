from keras.models import Sequential, Model, load_model
from keras.layers import UpSampling2D, Conv2D, Activation, BatchNormalization
from keras.layers import Reshape, Dense, Input
from keras.layers import LeakyReLU, Dropout, Flatten, ZeroPadding2D
from keras.optimizers import Adam
import numpy as np
import os

def split_train_sets(xraw):
    nstamps = xraw.shape[-1]
    npix = xraw.shape[0]
    nchans = 1
    ntrains = int(nstamps *  4./5.)
    ntests = int(nstamps * 1./5.)
    train = xraw[:,:,:ntrains].T.reshape(ntrains,npix,npix, 1)
    test = xraw[:,:,-ntests:].T.reshape(ntests,npix,npix,  1)
    return train, test

def divide_image(image):
    test_set_zoom = np.zeros((25, 64, 64))
    ind = 0
    for xax in range(5):
        for yax in range(5):
            test_set_zoom[ind, :, :] = image[64*xax:64*(xax+1), 64*yax:64*(yax+1)]
            ind = ind+1
    return test_set_zoom

def unify_image(images):
    one_big = np.zeros((320, 320))
    ind = 0
    for xax in range(5):
        for yax in range(5):
            one_big[64*xax:64*(xax+1), 64*yax:64*(yax+1)] = images[ind, :, :, 0]
            ind = ind+1
    return one_big


class DCGAN:
    def __init__(self, output_directory, img_size):
        self.img_size = img_size
        self.channels = 1
        self.kernel_size = 5
        #self.discriminator_path = discriminator_path
        #self.generator_path = generator_path
        self.output_directory = output_directory

    def build_generator(self):
        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same")) # 64x64x64
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same", strides=2)) #32x32x128
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(256, kernel_size=self.kernel_size, padding="same", strides=2)) #16x16x256
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())  #32x32x128
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())  #64x64x64
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("tanh"))
        img_in = Input(shape=img_shape)
        img_out = model(img_in)
        return Model(img_in, img_out)

    def build_discriminator(self):
        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=1, input_shape=img_shape, padding="same"))  # 192x256 -> 96x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))  # 48x64 -> 24x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=2, padding="same"))  # 24x32 -> 12x16
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        img = Input(shape=img_shape)
        validity = model(img)
        return Model(img, validity)

    def build_gan(self):
        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        optimizer = Adam(0.0002, 0.5)

        #if os.path.exists(self.discriminator_path) and os.path.exists(self.generator_path):
        #    self.discriminator = load_model(self.discriminator_path)
        #    self.generator = load_model(self.generator_path)
        #    print("Loaded models...")
        #else:
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

    def load_training_set(self, patch_path):
        Y,X = np.load(patch_path+'training_set_1000patches_10x10def_T_hr_1deg_lr_5deg_Npix64_set2.npy')
        Y = Y-X
        #Y = np.log10(np.transpose(Y[:1000]))
        #X = np.log10(np.transpose(X[:1000]))
        Y = np.transpose(Y[:1000])
        X = np.transpose(X[:1000])
        for i in range(Y.shape[-1]):
            Y[:,:,i] = 2*(Y[:,:,i]-Y[:,:,i].min())/(Y[:,:,i].max()-Y[:,:,i].min())-1
            X[:,:,i] = 2*(X[:,:,i]-X[:,:,i].min())/(X[:,:,i].max()-X[:,:,i].min())-1
            #Y[:,:,i] = 2*(Y[:,:,i]-Y[:,:,:].min())/(Y[:,:,:].max()-Y[:,:,:].min())-1
            #X[:,:,i] = 2*(X[:,:,i]-Y[:,:,:].min())/(Y[:,:,:].max()-Y[:,:,:].min())-1
        x_train, x_test = split_train_sets(X)
        y_train, y_test = split_train_sets(Y)
        return x_train, x_test, y_train, y_test
    
    def load_training_set_zoomed(self, patch_path):
        Ys, Xs = np.load(patch_path+'training_set_1000patches_10x10def_T_hr_1deg_lr_5deg_Npix64_set2.npy')
        Yl, Xl = np.load(patch_path+'training_set_1000patches_10x10def_T_hr_1deg_lr_5deg_Npix320_set2.npy')
        Xl_zoom = np.zeros((25*len(Yl), 64, 64))
        for i in range(40):
            Xl_zoom[i*25:(i+1)*25] = divide_image(Yl[i])
        X, Y = Xl_zoom, Ys-Xs
        #Y = np.log10(np.transpose(Y[:1000]))
        #X = np.log10(np.transpose(X[:1000]))
        Y = np.transpose(Y[:1000])
        X = np.transpose(X[:1000])
        for i in range(Y.shape[-1]):
            Y[:,:,i] = 2*(Y[:,:,i]-Y[:,:,i].min())/(Y[:,:,i].max()-Y[:,:,i].min())-1
            X[:,:,i] = 2*(X[:,:,i]-X[:,:,i].min())/(X[:,:,i].max()-X[:,:,i].min())-1
            #Y[:,:,i] = Y[:,:,i]/np.std(Xs[i])
            #X[:,:,i] = X[:,:,i]/np.std(X[:,:,i])
        x_train, x_test = split_train_sets(X)
        y_train, y_test = split_train_sets(Y)
        return x_train, x_test, y_train, y_test

    def train(self, epochs, batch_size=32, save_interval=100, patch_path='/global/homes/k/krach/scratch/NNforFG/training_set/'):
        self.build_gan()
        X_train, X_test, Y_train, Y_test = self.load_training_set(patch_path)
        #X_train, X_test, Y_train, Y_test = self.load_training_set_zoomed(patch_path)
        print("Training Data Shape: ", X_train.shape)
        half_batch = batch_size // 2
        accs = []
        for epoch in range(epochs):
            ind_batch = np.random.randint(0, 800, batch_size)
            # Train Generator
            g_loss = self.combined.train_on_batch(X_train[ind_batch], np.ones((batch_size, 1)))
            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = Y_train[idx]
            # Sample noise and generate a half batch of new images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            gen_imgs = self.generator.predict(X_train[idx])
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            acc = [d_loss_real[1], d_loss_fake[1]]
            accs.append(acc)
            # Print progress

            # If at save interval => save generated image samples, save model files
            if epoch % (save_interval) == 0:
                print(f"{epoch} [D loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
                save_path = self.output_directory + "/models"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                accs_to_save = np.array(accs)
                self.discriminator.save(save_path + '/discrim_'+str(epoch)+'.h5')
                self.generator.save(save_path + '/generat_'+str(epoch)+'.h5')
                np.save(save_path + '/acc_dreal_dfake_'+str(epoch)+'.npy', accs_to_save)
        self.discriminator.save(save_path + '/discrim_'+str(epoch)+'.h5')
        self.generator.save(save_path + '/generat_'+str(epoch)+'.h5')
        np.save(save_path + '/acc_dreal_dfake_'+str(epoch)+'.npy', accs_to_save)



if __name__ == '__main__':

    dcgan = DCGAN(output_directory='./tests/', img_size=(64, 64))
    dcgan.train(epochs=500000, batch_size=32, save_interval=5000)
