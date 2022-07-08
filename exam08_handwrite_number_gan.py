# -*- coding: utf-8 -*-
"""exam08_handwrite_Number_Gan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tRXjfrLYWgPxKahK217AkPUgSnB5dX9P
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

model = load_model('./models/generator_mnist_0_KDY.h5')
model.summary()

z = np.random.normal(0,1,(1,100))
fake_img = model.predict(z)
fake_imgs = 0.5 * fake_img + 0.5
plt.imshow(fake_img.reshape(28,28))
plt.show()

number_CAN_models = []
for i in range(10):
    number_CAN_models.append(
        load_model('./models/generator_mnist_{}.h5'.format(i)))

value = 12345
numbers = list(str(value))
print(numbers)

imgs = []
for i in numbers:
    i = int(i)
    z = np.random.normal(0,1,(1,100))
    fake_img = number_CAN_models[i].predict(z)
    imgs.append(fake_img.reshape(28,28))
    # plt.imshow(fake_img.reshape(28,28))
    # plt.show()

img = imgs[0]
for i in range(1, len(imgs)):
    img = np.append(img, imgs[i], axis=1)
# plt.summer()
plt.imshow(img, cmap = 'gray_r')
plt.axis('off')
plt.show()















