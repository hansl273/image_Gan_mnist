import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3),  activation='relu', padding='same')(input_img)
x = MaxPool2D((2, 2), padding='same')(x) # 1/4로 이미지 사이즈 축소
x = Conv2D(8, (3, 3),  activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3),  activation='relu', padding='same')(x)
encoded = MaxPool2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3),  activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3),  activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3),  activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3),  activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
print(conv_x_train.shape)
print(conv_x_test.shape)

noise_facter =0.5
x_train_noisy = conv_x_train + noise_facter * np.random.normal(
    loc=0.0, scale=1.0, size=conv_x_train.shape)
x_test_noisy = conv_x_test + noise_facter * np.random.normal(
    loc=0.0, scale=1.0, size=conv_x_test.shape)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fit_hist = autoencoder.fit(x_train_noisy, conv_x_train,
                           epochs=10, batch_size=128,
                           validation_data=(x_test_noisy, conv_x_test))

decoded_img = autoencoder.predict(x_test_noisy[:10])

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i+1)
    plt.imshow(x_test_noisy[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()