from dotenv import load_dotenv
load_dotenv()

import base64
import streamlit as st
import os
import io
import json
import pandas as pd
from PIL import Image
import pdf2image
import google.generativeai as genai


# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_prompt, pdf_content):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_prompt, pdf_content[0]])

    # Debug: Show raw response
    # st.write("Raw Response from Gemini:", response)

    # Extract response text safely
    try:
        response_text = response.candidates[0].content.parts[0].text
    except (AttributeError, IndexError, KeyError):
        st.error("Gemini API response structure is unexpected.")
        return {}

    # Clean response text
    response_text = response_text.replace("json", "", 1).strip()  # Remove "json" prefix if present
    response_text = response_text.strip("`")  # Remove surrounding backticks if any

    # Debug: Show cleaned response
    # st.write("Cleaned Response Text:", response_text)

    # Parse JSON safely
    try:
        parsed_response = json.loads(response_text)
        return parsed_response
    except json.JSONDecodeError as e:
        st.error(f"Error: Response is not in valid JSON format. {e}")
        return {}

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # Encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

## Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])
custom_directory = st.text_input("Select a folder to save the Excel file:", value=r"C:\Users\souja\Desktop\Resume Data")

# Ensure the directory exists
if custom_directory:
    os.makedirs(custom_directory, exist_ok=True)

submit1 = st.button("Process Resume")
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

Ensure the extracted details are returned as a JSON object in the following structure:
{
  "Name": "John Doe",
  "Email": "johndoe@example.com",
  "Mobile No": "+1234567890",
  "Address": "123 Main Street, New York, NY",
  "Education": "Bachelor’s Degree in Computer Science",
  "Skills": "Python, Machine Learning, JavaScript",
  "Experience": "Software Engineer at Google (2019-Present)",
  "Languages": "English, Spanish"
}

Only return valid JSON and no extra text.
"""  

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content)

        # Debugging: Print structured response
        # st.subheader("Extracted Data Response:")
        # st.json(response)  # Show raw response

        extracted_data = response if response else {}

        custom_directory = r"C:\Users\souja\Desktop\Resume Data"  # Change this to your preferred path
        os.makedirs(custom_directory, exist_ok=True)  # Ensure the directory exists
        excel_filename = os.path.join(custom_directory, "resume_data.xlsx")  # Full file path

    if extracted_data:
        # Convert dictionary to DataFrame
        new_data = pd.DataFrame([extracted_data])

        # Check if the Excel file already exists
        if os.path.exists(excel_filename):
            # Load existing data
            existing_data = pd.read_excel(excel_filename, engine="openpyxl")

            # Append new data if it's not already present
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            # If file doesn't exist, create a new DataFrame
            updated_data = new_data

        # Save the updated data back to the same file
        updated_data.to_excel(excel_filename, index=False, engine="openpyxl")

        st.success("✅ Resume data updated in existing Excel file!")

        # Show updated data in Streamlit (for debugging)
        st.dataframe(updated_data)

    else:
        st.warning("❌ No data extracted from the resume.")