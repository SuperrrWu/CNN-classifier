import os
from PIL import Image
import numpy as np
from keras import utils as np_utils
from keras import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Conv2D
from keras.optimizers import Adam
FilePath="E:/Python/人工智能/水果识别/"
Fruitype=["梨","葡萄","芒果","苹果"]
type_counter=0
for type in Fruitype:
    file_counter=0
    subfolder=os.listdir(FilePath+type)
    for i in subfolder:
    file_counter+=1
    os.rename(FilePath+type+"/"+i,FilePath+type+"/"+str(type_counter)+"_"+str(file_counter)+"_"+type+".jpg")
    type_counter+=1


for type in Fruitype:
    files=os.listdir(FilePath+type)
    for i in files:
        img_open=Image.open(FilePath+type+"/"+i)
        a=img_open.convert("RGB")
        resized_img=a.resize((100,100),Image.LINEAR)
        resized_img.save(os.path.join("E:/Python/人工智能/水果识别/training_set",os.path.basename(i)))

Training_set_img=[]
Training_set_label=[]
for file in os.listdir("E:/Python/人工智能/水果识别/training_set"):
    file_image=Image.open("E:/Python/人工智能/水果识别/training_set"+"/"+file)
    np_array=np.array(file_image)
    Training_set_img.append(np_array)
    Training_set_label.append(file.split("_")[0])
Training_img=np.array(Training_set_img)
Training_label=np.array( Training_set_label)
print(Training_img.shape)
print(Training_label.shape)
categories=4
batch_size=94
number_batch=50
Training_label=np_utils.to_categorical(Training_label,4)
#print(Training_label)
Training_img=Training_img.astype("float32")
Training_img=Training_img/255
print(Training_img)
model = Sequential()
model.add(Conv2D(
    input_shape=(100,100,3),
    filters=32,
    kernel_size=(5,5),
    padding="same"
))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(2,2),padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))#output_shape=(25,25,64)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dense(4))
model.add(Activation("softmax"))
adam=Adam(lr=0.0001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=Training_img,y=Training_label,epochs=number_batch,batch_size=batch_size,verbose=1)
model.save("./fruitfinder.h5")




