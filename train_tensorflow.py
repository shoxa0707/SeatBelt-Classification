from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from resolution import resolution
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="dataset path", type=str, default="Data")
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=16)
parser.add_argument("-e", "--epochs", help="epochs", type=int, default=100)
parser.add_argument("-s", "--save", help="model save path", type=str, default="models/seatbelt.model")
parser.add_argument('--res', action='store_true')
parser.add_argument('--no-res', dest='resolution', action='store_false')
args = parser.parse_args()

data = []
label = []
types = ['train', 'test']
classes = ['Taqilgan', 'Taqilmagan', 'Aniqlanmadi']
print('Loading images...')
for i in range(len(types)):
    for j in range(len(classes)):
        images = glob(args.data+'/'+types[i]+'/'+classes[j]+'/*')
        for img in images:
            rasm = cv2.imread(img)
            if args.res:
                rasm = resolution(rasm)
            rasm = cv2.resize(rasm, (320, 320))
            data.append(rasm)
            label.append(j)
        
data = np.array(data)
label = np.array(label)

label = keras.utils.to_categorical(label, 3)

trainX, testx, trainY, testy = train_test_split(data, label, test_size=0.10, random_state=707)
trainx, valx, trainy, valy = train_test_split(trainX, trainY, test_size=0.18, random_state=707)
print("Dataset distribution:")
print(trainx.shape, trainy.shape)
print(valx.shape, valy.shape)
print(testx.shape, testy.shape)

model = keras.Sequential([
    keras.layers.Rescaling(1/255.0, input_shape=(320,320,3)),
    
    keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
    
    keras.layers.Conv2D(16, (3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(32, (3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(64, (3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(128, (3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(256, (3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(512, (3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

H = model.fit(trainx, trainy, validation_data=(valx, valy), epochs=args.epochs, batch_size=args.batch_size)

model.evaluate(testx,testy)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(args.epochs),H.history['loss'], label="train_loss")
plt.plot(np.arange(args.epochs),H.history['accuracy'], label="train_acc")

plt.plot(np.arange(args.epochs),H.history['val_loss'], label="val_loss")
plt.plot(np.arange(args.epochs),H.history['val_accuracy'], label="val_acc")
plt.legend(loc="upper right")

model.save(args.save, save_format='h5')