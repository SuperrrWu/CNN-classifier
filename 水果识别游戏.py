from keras.models import load_model
from PIL import Image
import numpy as np

print("请输入你想要预测的图片filepath")
path = input()
img_open = Image.open(path)
a = img_open.convert("RGB")
a.show()
c=input("请输入你所看到的水果")
resized_img = a.resize((100, 100), Image.LINEAR)
test_set_img = resized_img
test_set_img = np.array(test_set_img)
test_set_img = np.expand_dims(test_set_img, axis=0)
print(test_set_img.shape)
model=load_model("C:/Users/Oliver/PycharmProjects/untitled7/fruitfinder.h5")
pre_results=model.predict_classes(test_set_img).astype('int')
pre_results=np.sum(pre_results)
Fruitype=["梨","葡萄","芒果","苹果"]
if c==Fruitype[pre_results]:
    print("你猜对了")
else:
    print("你猜错了，正确答案是",Fruitype[pre_results])

# C:/Users/Oliver/Desktop/水果识别/苹果/u=1178032958,140788891&fm=26&gp=0.jpg
#C:/Users/Oliver/Desktop/水果识别/芒果/u=1212044359,1729645316&fm=72.jpg