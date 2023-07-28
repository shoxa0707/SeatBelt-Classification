import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import argparse
from resolution import resolution
from models import SeatBelt
from torch import load, FloatTensor
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="usage type", type=str, default="tensorflow")
parser.add_argument("-m", "--model", help="model path", type=str, default="models/seatbelt.model")
parser.add_argument("-i", "--image", help="image for classification", type=str)
parser.add_argument("-d", "--device", help="inference device", type=str, default="cpu")
args = parser.parse_args()

# return condition of driver on seatbelt like "Taqilgan" or "Taqilmagan" and etc.
def predict_seatbelt_tensorflow(image, model):
    types = ['Taqilgan', 'Taqilmagan', 'Aniqlanmadi']
    if np.mean(image.shape[:2]) > 300:
        image = cv2.resize(image, (320, 320))
        cv2.imwrite(types[np.argmax(model(np.expand_dims(image, axis = 0)))], image)
    else:
        image = resolution(image)
        image = cv2.resize(image, (320, 320))
        cv2.imwrite(types[np.argmax(model(np.expand_dims(image, axis = 0)))], image)
    
def predict_seatbelt_torch(image, model):
    types = ['Taqilgan', 'Taqilmagan', 'Aniqlanmadi']
    # if the image size is close to 320, we use resolution to the image. Otherwise don't use
    if np.mean(image.shape[:2]) > 300:
        resize = cv2.resize(image, (320, 320)) / 255.0
        tensor = resize.transpose((2, 0, 1))
        tensor = FloatTensor(np.expand_dims(tensor, axis = 0)).to(device)
        cv2.imwrite(types[model(tensor).cpu().data.numpy().argmax()]+'.jpg', image)
    else:
        image = resolution(image)
        resize = cv2.resize(image, (320, 320)) / 255.0
        tensor = resize.transpose((2, 0, 1))
        tensor = FloatTensor(np.expand_dims(tensor, axis = 0)).to(device)
        cv2.imwrite(types[model(tensor).cpu().data.numpy().argmax()]+'.jpg', image)

# using model
if args.type == 'tensorflow':
    model = load_model(args.model)
    with tf.device('/'+args.device+':0'):
        img = cv2.imread(args.image)
        predict_seatbelt_tensorflow(img, model)
elif args.type == 'pytorch':
    model = SeatBelt()
    model.load_state_dict(load(args.model))
    device = 'cuda' if args.device == 'gpu' else 'cpu'
    model.to(device)
    model.eval()
    img = cv2.imread(args.image)
    predict_seatbelt_torch(img, model)
else:
    print('You selected the wrong type, supported values are [tensorflow, pytorch]!!!')
    