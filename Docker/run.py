from PIL import Image
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=False,
	help="path to output image")
args = vars(ap.parse_args())
features  = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']
#print("waiting for the camera to initialize")
model=tf.keras.models.load_model('mymodelcelebface.h5')# load model
image_name=args['input']

img = Image.open(image_name)
img.load()
img=img.resize((170,170),resample=0) 
x = image.img_to_array(img)
x = np.expand_dims(img, axis=0)

images = np.vstack([x])
classes = model.predict(x, batch_size=1)
classes=classes[0]

top=np.argsort(classes)[-11:][::-1]#print top 10 most confident predication irrespective of gender
for i in range(1,11):
    if top[i]==35:  #biased feature dropped
        continue
    perc=str(round((classes[top[i]]*100),2))
    print(str(features[top[i]])+"="+perc+"%")
    
        
img.show("output",args['output'])
print("terminated")
# Break the loop