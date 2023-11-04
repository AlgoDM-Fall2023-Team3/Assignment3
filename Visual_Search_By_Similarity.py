import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


def resize_image(image, target_size):
    resized_image = cv2.resize(image, target_size)
    return resized_image

def main():
    st.title("Image Classification")

    # Load the pre-trained model
    model = tf.keras.models.load_model("transfer_model.h5")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Resize the image to match the input shape of the model
        image_array = resize_image(img_to_array(image), (224, 224))
        image_array = np.expand_dims(image_array, axis=0)

        # Make a prediction
        class_probs = model.predict(image_array)
        df = pd.DataFrame(class_probs)
        df.columns = ['Dress', 'Hat' , 'Longsleeve' ,'Shoes' ,'T-Shirt']
        
        # Display the class probabilities
        st.write("Class Probabilities:")
        st.write(df)
        
        for index, row in df.iterrows():
            for col in df.columns:
                if row[col] == 1 or row[col] > 0.9:
                    prob_percentage = "{:.2%}".format(row[col])  # Format the percentage
                    st.write(f"Model Classified Image as {col}: {prob_percentage}")

    
if __name__ == "__main__":
    main()
