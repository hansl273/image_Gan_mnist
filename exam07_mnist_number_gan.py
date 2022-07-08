# -*- coding: utf-8 -*-
"""exam07_MNIST_Number_Gan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gLrRql_oknvTllA03hI42TTy8_XuVXC8
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

OUT_DIR = './OUT_img'
img_shape = (28,28,1)
epoch = 100000
batch_size = 128
noise = 100
sample_interval = 100
row = col = 4

(X_train, Y_train), (_, _) = mnist.load_data() 
print(X_train.shape)

MY_NUMBER = 2 
X_train = X_train[Y_train == MY_NUMBER]

a = [1,2,3,4,5,6]
a = np.array(a) # 논리 연산자로 인덱싱
b= a[[True, False, True, False,True, False]]
b

df = pd.DataFrame({'age':[20,30,40,50]}) # 논리 연산자로 인덱싱
df

df[df.age < 30]

df.age < 30

_, axs = plt.subplots(row, col, figsize=(row, col),
                              sharey=True, sharex=True)
cnt = 0
for i in range(row):
    for j in range(col):
        axs[i, j].imshow(X_train[cnt, :, :], cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
plt.show()

X_train = X_train / 127.5 -1  # -1, 255로 나누면 1,  최소값 -1 최대값 1
# X_train = X_train.reshape(-1, 28 28, 1) 아래와 동일 차원
X_train = np.expand_dims(X_train, axis=3) # 차원은 0 1 2 3
print(X_train.shape)

real = np.ones((batch_size, 1))  
fake = np.zeros((batch_size, 1))

generator_model = Sequential() #생성자
generator_model.add(Dense(256 * 7 * 7, input_dim=noise))
generator_model.add(Reshape((7, 7,256)))
generator_model.add(Conv2DTranspose(128, kernel_size=3, strides = 2, padding='same'))
generator_model.add(BatchNormalization()) # 중간중간 스켈링
generator_model.add(LeakyReLU(alpha=0.01)) # 시그모이드 기울기 손실을 렐루로 해결, - 값을 제거하지 않기 위해

generator_model.add(Conv2DTranspose(64, kernel_size=3, strides = 1, padding='same'))
generator_model.add(BatchNormalization()) # 중간중간 스켈링
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(1, kernel_size=3, strides = 2, padding='same'))
generator_model.add(Activation('tanh'))
generator_model.summary()
#https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/

lrelu = LeakyReLU(alpha=0.01)
discriminator_model = Sequential() # 판별자
discriminator_model.add(Flatten(input_shape = img_shape))
discriminator_model.add(Dense(128, activation = lrelu))
discriminator_model.add(Dense(1, activation = 'sigmoid')) # 이진분류
discriminator_model.summary() # 성능을 넘 좋게 만드는것은 좋지 않음. acc 0.5 수준으로 
# lrelu = LeakyReLU(alpha=0.01)
# discriminator_model = Sequential()
# discriminator_model.add(Flatten(input_shape=img_shape))
# discriminator_model.add(Dense(128, activation=lrelu))
# discriminator_model.add(Dense(1, activation='sigmoid'))
# discriminator_model.summary()

discriminator_model.compile(loss = "binary_crossentropy",
                            optimizer='adam', metrics=['accuracy'])
discriminator_model.trainable=False

gan_model = Sequential()
gan_model.add(generator_model)  # 생성자
gan_model.add(discriminator_model) # 판별자
discriminator_model.trainable=False
gan_model.summary()

gan_model.compile(loss = "binary_crossentropy", optimizer='adam')

for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size)  # 0 이면 60000, batch_size=128
    real_imgs = X_train[idx]

    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator_model.predict(z)

    d_hist_real = discriminator_model.train_on_batch(real_imgs,real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs,fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    z = np.random.normal(0,1,(batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real)
    # 간모델 학습 시 생성자만 학습
    
    if itr % sample_interval == 0 : 
        print('%d [D loss : %f, acc.:%0.2f%%] [G loss : %f]'%(
            itr, d_loss, d_acc *100, gan_hist))
        # 판별자 오차와 생성자 오차 역의 관계
        # 생성자 오차는 사기를 진실로, 판별자는 사기를 사기로 판명
        row = col = 4
        z = np.random.normal(0,1,(row * col, noise))
        fake_imgs = generator_model.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5 
        _, axs = plt.subplots(row, col, figsize=(row,col),
                              sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cnt, :, :, 0 ], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        plt.show()
        generator_model.save('./generator_mnist_{}_LEJ.h5'.format(MY_NUMBER))