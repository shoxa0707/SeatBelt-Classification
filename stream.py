import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import os
import matplotlib.pyplot as plt
from resolution import resolution

    
def predict_seatbelt(image, model):
    print(image.shape, 'predict ichi')
    types = ['Taqilgan', 'Taqilmagan', 'Aniqlanmadi']
    if np.mean(image.shape[:2]) > 300:
        image = cv2.resize(image, (320, 320))
        return types[np.argmax(model.predict(np.expand_dims(image, axis = 0)))]
    else:
        image = resolution(image)
        image = cv2.resize(image, (320, 320))
        print(image.shape)
        return types[np.argmax(model.predict(np.expand_dims(image, axis = 0)))]



model = load_model('models/seatbelt.model')

original_title = '<p style="color:Blue; font-size: 50px;"><strong>Classification and Resolution</strong></p>'
st.markdown(original_title, unsafe_allow_html=True)
st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

option = st.selectbox(
    'What kind of service do you want?',
    ('Seatbelt classification', 'Image super resolution (4x scale)'))

if option == "Seatbelt classification":
    st.title('Seatbelt classification')

    st.header(":red[Upload your image]")
    img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert('RGB')
        img_array = np.array(image)
        out_img = cv2.resize(img_array, (320, 320))
        st.image(out_img)

        if st.button('Predict'):
            try:
                result = predict_seatbelt(img_array, model)
                st.write(f'result: {result}')
            except:
                pass


elif option == "Image super resolution (4x scale)":
    st.title('Image super resolution (4x scale)')
    st.header(":green[Upload your image]")

    img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        res_img = resolution(img_array)
        # convert RGB to BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('resolution.png', res_img)

        fig1 = plt.figure(figsize = (10, 10))
        plt.subplot(121)
        plt.imshow(img_array)
        plt.title('original image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(res_img)
        plt.title('Resolution image')
        plt.axis('off')
        plt.subplots_adjust(wspace=.025, hspace=.025)
        fig1.set_facecolor("black")
        st.pyplot(fig1)

        with open("resolution.png", "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="resolution.png",
                    mime="image/png"
                )
    else:
        try:
            os.remove('resolution.png')
        except:
            pass
        
