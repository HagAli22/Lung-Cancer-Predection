import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Cancer Classification App",
    layout="wide"
)

# Application title
st.title("Cancer Classification System")
st.markdown("---")

# Application description
st.markdown("""
### AI-Powered Cancer Classification Application
This application uses a CNN model to classify medical images into three different categories.
Upload a medical image and the model will classify it for you.
""")

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model/Cancer_Model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(img):
    # Convert image to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize((64, 64))
    
    # Convert to array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization
    
    return img_array

# Prediction
def predict_image(model, img_array):
    predictions = model.predict(img_array)
    return predictions

# Load model
model = load_model()

if model is not None:
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        # Upload image
        uploaded_file = st.file_uploader(
            "Choose a medical image...",
            type=['png', 'jpg', 'jpeg'],
            help="Please upload an image in PNG, JPG, or JPEG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            
            # Image information
            st.write(f"**File Name:** {uploaded_file.name}")
            st.write(f"**Image Size:** {image_pil.size}")
            st.write(f"**Image Type:** {image_pil.format}")
    
    with col2:
        st.subheader("Classification Results")
        
        if uploaded_file is not None:
            with st.spinner('Analyzing image...'):
                try:
                    # Process image
                    processed_image = preprocess_image(image_pil)
                    
                    # Prediction
                    predictions = predict_image(model, processed_image)
                    
                    # Class names (modify according to your dataset)
                    class_names = ['Benign', 'Malignant', 'Normal']  # Update these names
                    
                    # Get predicted class
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class = class_names[predicted_class_index]
                    confidence = float(predictions[0][predicted_class_index] * 100)  # Convert to regular float
                    
                    # Display results
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display main result
                    st.markdown(f"""
                    ### Predicted Result:
                    **{predicted_class}**
                    
                    ### Confidence Level:
                    **{confidence:.2f}%**
                    """)
                    
                    # Display all probabilities
                    st.markdown("### Probability Distribution:")
                    
                    # Create progress bar for each class
                    for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
                        prob_percentage = float(prob * 100)
                        st.write(f"**{class_name}:**")
                        st.progress(float(prob))  # Convert to regular float
                        st.write(f"{prob_percentage:.2f}%")
                        st.write("")
                    
                    # Warning if confidence is low
                    if confidence < 70:
                        st.warning("Low confidence level. Please consult a medical specialist to confirm the result.")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        else:
            st.info("Please upload an image first to start analysis")

    # Additional information
    st.markdown("---")
    st.markdown("""
    ### Important Information:
    - This application is for educational and research purposes only
    - Results are not a substitute for professional medical consultation
    - Please consult a medical specialist for accurate diagnosis
    - Ensure image quality for best results
    """)
    
    # Model statistics (if available)
    with st.expander("Model Information"):
        st.write("**Model Architecture:**")
        st.write("- Number of Convolutional Layers: 3")
        st.write("- Filter Size: 32")
        st.write("- Input Image Size: 64x64 pixels")
        st.write("- Number of Classes: 3")
        st.write("- Activation Functions: ReLU and Softmax")

else:
    st.error("❌ Model file not found. Make sure 'cnn_model.keras' file exists in the same directory as the application.")
    st.info("💡 Make sure to run the training code first and save the model.")

# Application footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>This application was built using Streamlit and TensorFlow</p>
</div>
""", unsafe_allow_html=True)