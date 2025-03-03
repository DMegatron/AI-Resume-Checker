from dotenv import load_dotenv
load_dotenv()
import base64
import streamlit as st
import os
import io
import json
from PIL import Image 
import pdf2image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, pdf_content):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content[0]])
    # Assuming the response is structured in a way that can be converted to JSON
    try:
        # Convert the response text to a dictionary (or any other structure)
        response_dict = json.loads(response.text)
    except json.JSONDecodeError:
        # If the response is not in JSON format, wrap it in a dictionary
        response_dict = {"response": response.text}
    return response_dict

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        ## Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())

        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

## Streamlit App

st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
uploaded_file = st.file_uploader("Upload your resume(PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("Tell Me About the Resume")

input_prompt1 = """
You are an experienced Technical Human Resource Manager. Your task is to extract and list the following details from the provided resume:  

- **Name**  
- **Email**  
- **Mobile No**  
- **Address**  
- **Education**  
- **Skills**  
- **Experience**  
- **Languages**  

Ensure the extracted details are structured clearly and accurately. Do not provide any evaluation or analysis.  
"""  

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content)
        st.subheader("The Response is")
        st.json(response)  # Display the response as JSON
    else:
        st.write("Please upload the resume")