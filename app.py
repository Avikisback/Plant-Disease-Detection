import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

#Tensorflow prediction
def model_prediction(test_image):
    model  = tf.keras.models.load_model('crop_disease_detection_my_official_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_ind = np.argmax(prediction)
    return result_ind
#sidebar for choosing different pages
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Crop Disease Detecion"])

#Home page
if app_mode=="Home":
    
    image_path = "E:/python/ml/cropskyofficial_logo.jpeg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to Cropsky!
    
    CropSky, a pioneering tech startup, is revolutionizing agriculture industry with AI, robotics, drone, and IoT. We are working on cutting-edge AI/ML 
    models and ROS-based simulations.
                
    One such project is our crop disease detection program which lets you choose any picture from your folders and by running it through our program it tells you the health status of the leaf in the given picture.

    How It Works\n
    - Upload an Image: Navigate to the Crop Disease Detection page and upload a plant image showing signs of disease.\n
    - Processing: Our AI-powered system analyzes the image using advanced machine learning techniques to identify potential diseases.\n
    - Results & Recommendations: Instantly receive a diagnosis along with expert recommendations for further action.\n
                
    Why Choose Us?\n
    - High Accuracy: Built with cutting-edge deep learning models for precise disease detection.\n
    - Easy to Use: A simple and intuitive interface designed for seamless user experience.\n
    - Fast & Reliable: Get quick results, helping you take timely action to protect your crops.\n
                
    Get Started\n
    - Click on the Crop Disease Detection page in the navigation bar to upload an image and see our AI-powered system in action!
""")
elif app_mode=="Crop Disease Detecion":
    st.header("Crop Disease Detecion")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)
    #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please Wait.."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Define Class
            class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
            cures = [
    "Use resistant varieties, apply fungicides (captan, myclobutanil), prune for airflow, remove infected leaves.",
    "Prune infected branches, remove mummified fruit, spray fungicides (copper-based, captan).",
    "Remove nearby junipers, use resistant varieties, apply fungicides (myclobutanil, mancozeb).",
    "Maintain proper watering, fertilization, and pruning for disease prevention.",
    "Mulch properly, avoid overhead watering, ensure good drainage.",
    "Prune for airflow, use neem oil, apply sulfur or potassium bicarbonate.",
    "Regular pruning, avoid excessive nitrogen fertilizers.",
    "Rotate crops, use resistant varieties, apply fungicides (strobilurins, triazoles).",
    "Plant resistant hybrids, remove infected debris, apply fungicides if severe.",
    "Rotate crops, plant resistant hybrids, use fungicides (azoxystrobin, pyraclostrobin).",
    "Maintain soil fertility, weed management, and crop rotation.",
    "Prune infected parts, remove mummies, apply fungicides (captan, mancozeb).",
    "Remove affected vines, avoid excessive pruning wounds, apply fungicides.",
    "Improve ventilation, remove diseased leaves, use copper fungicides.",
    "Proper trellising, avoid excessive watering.",
    "Remove infected trees, control psyllid insects (imidacloprid), use resistant rootstocks.",
    "Use resistant varieties, prune affected twigs, apply copper sprays.",
    "Proper irrigation and fertilization.",
    "Use copper-based sprays, remove infected plants, avoid overhead watering.",
    "Ensure proper spacing and airflow.",
    "Rotate crops, remove infected foliage, apply fungicides (chlorothalonil, mancozeb).",
    "Remove infected plants, use copper fungicides, avoid excess humidity.",
    "Use certified disease-free seeds, maintain proper irrigation.",
    "Mulch properly, ensure good airflow.",
    "Rotate crops, ensure proper drainage.",
    "Use resistant varieties, spray with neem oil, apply sulfur fungicides.",
    "Remove infected leaves, apply copper fungicides, avoid overhead watering.",
    "Ensure proper spacing and remove dead leaves.",
    "Remove infected leaves, apply copper-based sprays, improve airflow.",
    "Rotate crops, remove infected debris, use fungicides (chlorothalonil).",
    "Remove infected plants, apply fungicides (copper sprays, mancozeb).",
    "Ensure good ventilation, apply sulfur or copper-based fungicides.",
    "Prune lower leaves, apply fungicides (chlorothalonil).",
    "Spray with insecticidal soap or neem oil.",
    "Improve airflow, apply copper fungicides.",
    "Control whiteflies, remove infected plants, use resistant varieties.",
    "Remove infected plants, disinfect tools, avoid tobacco use near plants.",
    "Proper crop rotation, remove weeds, and ensure good spacing."]

        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        st.markdown(f"Next Steps:\n{cures[result_index]}")
