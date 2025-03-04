from dotenv import load_dotenv
load_dotenv()

import base64
import streamlit as st
import os
import io
import json
import pandas as pd
import pdfplumber
import google.generativeai as genai
import time  # For rate limiting

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Rate limiting variables
RATE_LIMIT = 500  # Maximum requests per minute
TIME_WINDOW = 60  # Time window in seconds (1 minute)
request_timestamps = []  # Stores timestamps of requests

# Username and password for authentication
# Use st.secrets for Streamlit Cloud, or fallback to environment variables/hardcoded values for localhost
try:
    USERNAME = st.secrets["auth"]["username"]  # For Streamlit Cloud
    PASSWORD = st.secrets["auth"]["password"]  # For Streamlit Cloud
except Exception:
    # Fallback for localhost (use environment variables or hardcoded values)
    USERNAME = os.getenv("USERNAME", "admin")  # Default: "admin"
    PASSWORD = os.getenv("PASSWORD", "password123")  # Default: "password123"

def check_rate_limit():
    """Check if the rate limit has been exceeded."""
    global request_timestamps
    current_time = time.time()

    # Remove timestamps older than the time window
    request_timestamps = [t for t in request_timestamps if current_time - t <= TIME_WINDOW]

    # Check if the number of requests exceeds the limit
    if len(request_timestamps) >= RATE_LIMIT:
        st.error("⚠️ Rate limit exceeded. Please wait a minute before making more requests.")
        return False
    else:
        # Add the current request timestamp
        request_timestamps.append(current_time)
        return True

def get_gemini_response(input_prompt, pdf_content):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_prompt, pdf_content])

        if not response or not hasattr(response, "candidates") or not response.candidates:
            st.warning("Unable to generate a valid response. Please try again.")
            return {}

        try:
            response_text = response.candidates[0].content.parts[0].text
        except (AttributeError, IndexError, KeyError):
            st.warning("Unexpected response format received. Please try again later.")
            return {}

        # Cleaning response text before JSON parsing
        response_text = response_text.replace("json", "", 1).strip().strip("`")

        try:
            parsed_response = json.loads(response_text)
            return parsed_response
        except json.JSONDecodeError:
            st.warning("Received response could not be processed. Please try again later.")
            return {}

    except Exception:
        st.warning("An error occurred while processing your request. Please try again later.")
        return {}

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    else:
        raise FileNotFoundError("No file uploaded")

## Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")

# Username and password authentication
def authenticate(username, password):
    """Check if the provided username and password match the stored credentials."""
    return username == USERNAME and password == PASSWORD

# Login form
with st.sidebar:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

# Check if the user is authenticated
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if login_button:
    if authenticate(username, password):
        st.session_state.authenticated = True
        st.sidebar.success("✅ Login successful!")
    else:
        st.sidebar.error("❌ Invalid username or password.")

# Only show the app content if the user is authenticated
if st.session_state.authenticated:
    uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

    # Initialize session state to store the DataFrame
    if "resume_data" not in st.session_state:
        st.session_state.resume_data = pd.DataFrame()

    # Button to process the resume and extract data
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

    # Button to check ATS score
    submit2 = st.button("Check ATS Score")
    input_prompt2 = """
    You are an advanced Applicant Tracking System (ATS) evaluator with expertise in resume screening and optimization. Your task is to analyze the provided resume and assess its ATS compatibility based on the following key criteria:

    1. **Keyword Relevance**: 
       - Does the resume include industry-specific keywords and job-related terms?
       - Are the keywords used in a natural and effective manner?
       - Are important skills, certifications, and technologies mentioned?

    2. **Formatting & Parsing**: 
       - Is the resume structured in a way that is ATS-friendly (e.g., no tables, graphics, or complex formatting that might hinder parsing)?
       - Are section headings clear and correctly labeled?
       - Is the document in a standard file format (e.g., PDF, DOCX) suitable for ATS?

    3. **Completeness**:
       - Does the resume include all essential sections: Contact Information, Summary, Work Experience, Skills, Education, Certifications (if applicable)?
       - Are job titles and company names clearly mentioned?
       - Are employment dates included and formatted correctly?

    4. **Clarity & Readability**: 
       - Is the language clear, concise, and professional?
       - Are there any grammar or spelling mistakes?
       - Are job descriptions and accomplishments well-articulated?

    After analyzing the resume, provide an ATS compatibility score out of 100 and a detailed breakdown of the evaluation. Highlight what the resume does well and provide constructive feedback on areas for improvement.

    Return the response strictly in the following JSON format:

    {
      "ATS_Score": 85,
      "Strengths": [
        "The resume includes relevant industry keywords and job-specific skills.",
        "The structure is clean and ATS-friendly, with proper section headings."
      ],
      "Improvements": [
        "Work experience descriptions could be more detailed with quantified achievements.",
        "The resume lacks a dedicated certifications section, which could improve ATS matching."
      ],
      "Overall_Feedback": "The resume is well-structured and includes key ATS-friendly elements. However, adding more detailed work descriptions and ensuring all relevant sections are present could further optimize it."
    }

    Only return valid JSON and no extra text.
    """

    if submit1 or submit2:
        if not check_rate_limit():
            st.stop()  # Stop further execution if rate limit is exceeded

        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)

            if submit1:
                response = get_gemini_response(input_prompt1, pdf_content)
                extracted_data = response if response else {}

                if extracted_data:
                    new_data = pd.DataFrame([extracted_data])

                    # Check for duplicates based on "Email" (or another unique field)
                    if not st.session_state.resume_data.empty:
                        # Check if the email already exists in the stored data
                        if extracted_data["Email"] in st.session_state.resume_data["Email"].values:
                            st.warning("⚠️ This resume (based on email) already exists in the database. Skipping duplicate entry.")
                        else:
                            # Append new data to the existing data in session state
                            st.session_state.resume_data = pd.concat([st.session_state.resume_data, new_data], ignore_index=True)
                            st.success("✅ Resume data processed and appended successfully!")
                    else:
                        # If no data exists yet, just add the new data
                        st.session_state.resume_data = new_data
                        st.success("✅ Resume data processed and stored successfully!")

                    st.dataframe(st.session_state.resume_data)

                else:
                    st.warning("❌ No data extracted from the resume.")

            elif submit2:
                response = get_gemini_response(input_prompt2, pdf_content)

                if response:
                    st.subheader("ATS Score and Feedback")
                    st.write(f"**ATS Score:** {response.get('ATS_Score', 'N/A')}")

                    strengths = response.get("Strengths", [])
                    improvements = response.get("Improvements", [])
                    overall_feedback = response.get("Overall_Feedback", "No feedback available.")

                    if strengths:
                        st.write("### Strengths:")
                        for strength in strengths:
                            st.write(f"- {strength}")

                    if improvements:
                        st.write("### Areas for Improvement:")
                        for improvement in improvements:
                            st.write(f"- {improvement}")

                    st.write("### Overall Feedback:")
                    st.write(overall_feedback)
                else:
                    st.error("Failed to generate ATS score and feedback.")
        else:
            st.error("Please upload a PDF file to process.")

    # Download button for the accumulated data
    if not st.session_state.resume_data.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            st.session_state.resume_data.to_excel(writer, index=False, sheet_name="Resume Data")
        output.seek(0)

        st.download_button(
            label="Download Excel File",
            data=output,
            file_name="resume_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Button to clear the stored data
    if st.button("Clear All Data"):
        st.session_state.resume_data = pd.DataFrame()
        st.success("✅ All data has been cleared!")
else:
    st.warning("Please log in to access the app.")