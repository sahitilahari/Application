# Imported Libraries
import streamlit as st # type: ignore
import google.generativeai as genai # type: ignore
import pytesseract # type: ignore
from PIL import Image # type: ignore
import os
from langchain_google_genai import GoogleGenerativeAI # type: ignore
import pyttsx3 # type: ignore

# Tessaract OCR library to execute read image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# API Key
Gemini_api_key = "AIzaSyCwT6bAjVZ-6TWih4xBFBnX28uiyNlHXlE"
os.environ["Gemini_api_key"] = Gemini_api_key
# Imported LLM
llm = GoogleGenerativeAI(model= "gemini-1.5-pro", api_key=Gemini_api_key)



st.title("🌟 AI Assistant for the Visually Impaired")
st.markdown(
    """
    Welcome to the AI Assistant! This tool helps visually impaired individuals by:
    - Describing scenes in images.
    - Extracting text from images for text-to-speech conversion.
    """
)

# Extract text from image
def extract_text_from_image(image):
    config = r'--psm 6'
    text = pytesseract.image_to_string(image, config=config)
    return text
        

# Generate a scene description using generative AI
def generate_scene_description(input_prompt, image_data):
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()
   
    
# Prepare uploaded image for processing
def image_upload(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type,
                        "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")
    
   
# File uploader
uploaded_file = st.file_uploader("Upload an image!", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption ="Uploaded Image", use_container_width=True)

# Center-aligned buttons
st.markdown("<h3 style='text-align: center;'>Choose an Action</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths to center buttons
with col1:
    st.write("")  # Empty column for spacing
with col2:
    describe_button = st.button(" Describe Scene", use_container_width=True)
    text_extract_button = st.button(" Extract Text", use_container_width=True)
    speech_conv_button = st.button("Text-to-Speech", use_container_width = True)
with col3:
    st.write("")  # Empty column for spacing
# Input prompt for scene understanding
input_prompt = """
You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
1. List of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""

if uploaded_file:
    image_data = image_upload(uploaded_file)
    
    if describe_button:
        with st.spinner("Generating scene description.."):
            response = generate_scene_description(input_prompt, image_data)
            st.subheader("Scene Description")
            st.write(response)
            
    if text_extract_button:
        with st.spinner("Extracting text.."):
            text = extract_text_from_image(image)
            st.subheader("Text Extraction")
            #st.write(text)
            st.markdown(f"```\n{text}\n```")
            
    if speech_conv_button:
        with st.spinner("Converting text into speech..."):
            text = extract_text_from_image(image)
            if text.strip():
                text_to_speech(text)
                st.success("Text to Speech conversion completed!")
            else:
                st.error("No text found in the image.")
