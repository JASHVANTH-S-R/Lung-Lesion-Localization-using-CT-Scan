#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D,Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D

main_dir = "./Dataset"
train_dir = os.path.join(main_dir, "train")
test_dir = os.path.join(main_dir, "test")

train_covid_dir = os.path.join(train_dir, "COVID")
train_normal_dir = os.path.join(train_dir, "non-COVID")

test_covid_dir = os.path.join(test_dir, "COVID")
test_normal_dir = os.path.join(test_dir, "non-COVID")

train_covid_names = os.listdir(train_covid_dir)
train_normal_names = os.listdir(train_normal_dir)

test_covid_names = os.listdir(test_covid_dir)
test_normal_names = os.listdir(test_normal_dir)

rows = 4
columns = 4
fig = plt.gcf()
fig.set_size_inches(20,20)

covid_img = [os.path.join(train_covid_dir, filename) for filename in train_covid_names[0:8]]
normal_img = [os.path.join(train_normal_dir, filename) for filename in train_normal_names[0:8]]

print(covid_img)
print(normal_img)

merged_img = covid_img + normal_img

for i, img_path in enumerate(merged_img):
  title = img_path.split("/", 1)[1]
  plot = plt.subplot(rows, columns, i+1)
  plot.axis("Off")
  img = mpimg.imread(img_path)
  plot.set_title(title, fontsize = 11)
  plt.imshow(img, cmap= "gray")

plt.show()

dgen_train = ImageDataGenerator(rescale = 1./255,
                                validation_split = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

dgen_validation = ImageDataGenerator(rescale = 1./255,
                                     )

dgen_test = ImageDataGenerator(rescale = 1./255,
                              )

train_generator = dgen_train.flow_from_directory(train_dir,
                                                 target_size = (150, 150), 
                                                 subset = 'training',
                                                 batch_size = 32,
                                                 class_mode = 'binary')
validation_generator = dgen_train.flow_from_directory(train_dir,
                                                      target_size = (150, 150), 
                                                      subset = "validation", 
                                                      batch_size = 32, 
                                                      class_mode = "binary")
test_generator = dgen_test.flow_from_directory(test_dir,
                                               target_size = (150, 150), 
                                               batch_size = 32, 
                                               class_mode = "binary")


print("Class Labels are: ", train_generator.class_indices)
print("Image shape is : ", train_generator.image_shape)

model = Sequential()
model.add(Conv2D(32, (5,5), padding = "same", activation = "relu", input_shape = train_generator.image_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (5,5), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(train_generator, 
                    epochs = 10, 
                    validation_data = validation_generator)

model_json = model.to_json()
with open(r"./Model/lung_model.json", "w") as json_file:
    json_file.write(model_json)

from keras.preprocessing import image
import smtplib, ssl

import imghdr
from email.message import EmailMessage


img_path = './test/y3.png'
img = image.load_img(img_path, target_size = (150,150,3))
images = image.img_to_array(img)
images = np.expand_dims(images, axis = 0)
prediction = model.predict(images)

message=prediction 
if prediction == 0:
    port = 587 
    smtp_server = "smtp.gmail.com"
    sender_email = "gts2021to2022@gmail.com"
    receiver_email = "arunpotter9383@gmail.com"
    password = "20212022"
    message = """    Subject: COVID-19 Report

    Your COVID report is Positive, So Consult a doctor as soon as possible"""

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo() 
        server.starttls(context=context)
        server.ehlo()  
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    print("Kindly Check your mail")
elif prediction == 1:
    port = 587 
    smtp_server = "smtp.gmail.com"
    sender_email = "gts2021to2022@gmail.com"
    receiver_email = "arunpotter9383@gmail.com"
    password = "20212022"
    message = """    Subject: COVID-19 Report

    Congratulations, Your COVID report is Negative """

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo() 
        server.starttls(context=context)
        server.ehlo()  
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    print("Kindly Check your mail")
else:
    print("Image is Not Clear")

