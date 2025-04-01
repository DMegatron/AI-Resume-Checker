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
try:
    USERNAME = st.secrets["auth"]["username"]
    PASSWORD = st.secrets["auth"]["password"]
except Exception:
    USERNAME = os.getenv("USERNAME", "admin")
    PASSWORD = os.getenv("PASSWORD", "password123")

def check_rate_limit():
    """Check if the rate limit has been exceeded."""
    global request_timestamps
    current_time = time.time()
    request_timestamps = [t for t in request_timestamps if current_time - t <= TIME_WINDOW]
    if len(request_timestamps) >= RATE_LIMIT:
        st.error("⚠️ Rate limit exceeded. Please wait a minute before making more requests.")
        return False
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

        response_text = response_text.replace("json", "", 1).strip().strip("`")
        return json.loads(response_text)
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

# Authentication
def authenticate(username, password):
    return username == USERNAME and password == PASSWORD

with st.sidebar:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "resume_data" not in st.session_state:
    st.session_state.resume_data = pd.DataFrame()

if login_button:
    if authenticate(username, password):
        st.session_state.authenticated = True
        st.sidebar.success("✅ Login successful!")
    else:
        st.sidebar.error("❌ Invalid username or password.")

if st.session_state.authenticated:
    # Modified file uploader to accept multiple files
    uploaded_files = st.file_uploader(
        "Upload your resume(s) (PDF)...", 
        type=["pdf"],
        accept_multiple_files=True
    )

    # Initialize session state for batch processing
    if "batch_processing" not in st.session_state:
        st.session_state.batch_processing = False

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
      "Education": "Bachelor's Degree in Computer Science",
      "Skills": "Python, Machine Learning, JavaScript",
      "Experience": "Software Engineer at Google (2019-Present)",
      "Languages": "English, Spanish"
    }

    Only return valid JSON and no extra text.
    """  

    input_prompt2 = """
    You are an advanced Applicant Tracking System (ATS) evaluator with expertise in resume screening and optimization. Your task is to analyze the provided resume and assess its ATS compatibility based on the following key criteria:

    1. **Keyword Relevance & Optimization (25 points)**: 
       - Does the resume include industry-specific keywords and job-related terms?
       - Are the keywords used in a natural and effective manner?
       - Are important skills, certifications, and technologies mentioned?
       - Are keywords strategically placed in headers, bullet points, and summaries?

    2. **Formatting & Parsing (25 points)**: 
       - Is the resume structured in a way that is ATS-friendly?
       - Are there any complex elements that might hinder parsing (tables, graphics, columns)?
       - Are section headings clear, standard, and correctly labeled?
       - Is the font and formatting consistent and readable?
       - Are there any special characters or symbols that might confuse an ATS?

    3. **Completeness & Structure (25 points)**:
       - Does the resume include all essential sections: Contact Information, Summary/Objective, Work Experience, Skills, Education, Certifications?
       - Are job titles, company names, and dates clearly formatted?
       - Is the chronology clear and logical?
       - Are achievements quantified with metrics where possible?
       - Is there a clear career progression visible?

    4. **Clarity, Content & Readability (25 points)**: 
       - Is the language clear, concise, and professional?
       - Are there any grammar or spelling mistakes?
       - Are job descriptions focused on achievements rather than just responsibilities?
       - Is the resume an appropriate length (1-2 pages for most professionals)?
       - Is information prioritized effectively with the most relevant details first?

    After analyzing the resume, provide an ATS compatibility score out of 100 (with subscores for each category) and a detailed breakdown of the evaluation.

    Return the response strictly in the following JSON format:

    {
      "ATS_Score": 85,
      "Category_Scores": {
        "Keyword_Relevance": 22,
        "Formatting_Parsing": 20,
        "Completeness_Structure": 23,
        "Clarity_Content": 20
      },
      "Keyword_Analysis": {
        "Detected_Keywords": ["project management", "agile", "stakeholder communication"],
        "Missing_Common_Keywords": ["PMP certification", "JIRA", "Scrum"],
        "Keyword_Distribution": "Good distribution across resume sections"
      },
      "Format_Analysis": {
        "ATS_Friendly_Elements": ["Clear section headers", "Standard formatting", "Proper use of bullet points"],
        "ATS_Unfriendly_Elements": ["Text in header might be missed", "Tables in skills section"]
      },
      "Structure_Analysis": {
        "Present_Sections": ["Contact Information", "Professional Summary", "Experience", "Skills", "Education"],
        "Missing_Sections": ["Certifications", "Projects"],
        "Section_Quality_Notes": "Experience section well-structured but Education section lacks details"
      },
      "Content_Analysis": {
        "Strengths": ["Strong action verbs", "Quantified achievements", "Clear job titles"],
        "Weaknesses": ["Some bullet points too lengthy", "Technical jargon without explanation"]
      },
      "Strengths": [
        "The resume includes relevant industry keywords and job-specific skills",
        "The structure is clean and mostly ATS-friendly",
        "Achievements are well-quantified with metrics"
      ],
      "Improvements": [
        "Work experience descriptions could be more concise",
        "Add a dedicated certifications section",
        "Remove the table format in the skills section",
        "Consider adding more technical keywords relevant to the industry"
      ],
      "Overall_Feedback": "This resume is well-structured and includes key ATS-friendly elements. The strongest aspect is the quantified achievements in the experience section. To further optimize, focus on improving the formatting issues, adding missing sections, and incorporating more industry-specific keywords."
    }

    Only return valid JSON and no extra text.
    """
    
    input_prompt3 = """
    You are an advanced Applicant Tracking System (ATS) evaluator with expertise in resume screening and optimization. Your task is to analyze how well the provided resume matches the specific job description based on the following key criteria:

    1. **Keyword Matching (25 points)**: 
       - How many of the key skills and qualifications from the job description appear in the resume?
       - Are the important terms from the job description present in the resume?
       - Are the keywords used in a natural and contextually appropriate way?
       - Are the keywords strategically placed in headers, bullet points, and summaries?

    2. **Experience Alignment (25 points)**:
       - Does the work experience in the resume align with the requirements in the job description?
       - Are the years of experience and job roles relevant to what's being asked for?
       - Do the responsibilities and achievements in the resume match what the job requires?
       - Is there evidence of progression in relevant areas?

    3. **Skills Match (25 points)**:
       - What percentage of the required skills in the job description are present in the resume?
       - Are there any critical missing skills that should be added?
       - Are the skills presented clearly and prominently?
       - Is there evidence of proficiency in the required skills?

    4. **Education & Certification Alignment (15 points)**:
       - Does the education background match the job requirements?
       - Are there any required certifications or qualifications that are missing?
       - Are relevant coursework or specialized training highlighted?

    5. **Overall Fit & Presentation (10 points)**:
       - How well does the candidate's profile match the company culture and job requirements?
       - Is the resume formatted in a way that highlights the most relevant information?
       - Does the career trajectory suggest the candidate is appropriate for this role?

    After analyzing the resume against the job description, provide a matching score out of 100 (with subscores for each category) and a detailed breakdown of the evaluation.

    Return the response strictly in the following JSON format:

    {
      "Job_Match_Score": 75,
      "Category_Scores": {
        "Keyword_Matching": 19,
        "Experience_Alignment": 20,
        "Skills_Match": 18,
        "Education_Certification": 12,
        "Overall_Fit": 6
      },
      "Keyword_Analysis": {
        "Matching_Keywords": ["Python", "Machine Learning", "Data Analysis", "SQL", "Data Visualization"],
        "Missing_Keywords": ["TensorFlow", "Cloud Computing", "AWS", "Big Data"],
        "Keyword_Match_Percentage": "65% of key terms matched",
        "High_Priority_Missing_Terms": ["TensorFlow", "AWS"]
      },
      "Experience_Analysis": {
        "Experience_Match": "The 5 years of software development experience matches well with the job requirements",
        "Role_Alignment": "Previous Data Scientist position aligns closely with the target role",
        "Missing_Experience": "No experience mentioned with cloud infrastructure as required in the job description",
        "Experience_Level": "Senior level experience matches the requirement"
      },
      "Skills_Analysis": {
        "Skills_Present": ["Python", "SQL", "Data Analysis", "Statistical Modeling", "Machine Learning"],
        "Critical_Missing_Skills": ["Cloud Computing", "TensorFlow/PyTorch", "Data Pipeline Development"],
        "Skills_Match_Ratio": "8 out of 12 required skills present (67%)",
        "Skill_Proficiency_Evidence": "Projects demonstrating Python and ML skills are well-documented"
      },
      "Education_Analysis": {
        "Education_Match": "The Bachelor's degree in Computer Science matches the minimum requirement",
        "Missing_Qualifications": "The job prefers a Master's degree, which is not present",
        "Relevant_Coursework": "Relevant coursework in machine learning is mentioned, which is a plus"
      },
      "Job_Title_Match": "Current job title 'Data Scientist' is very relevant to the 'Senior Data Scientist' position",
      "Industry_Experience_Match": "Experience in fintech aligns with the target role in financial services",
      "Strengths": [
        "Strong match in programming languages and frameworks",
        "Relevant work experience in similar roles",
        "Project experience demonstrates required technical skills",
        "Education background meets basic requirements"
      ],
      "Improvements": [
        "Add experience with TensorFlow to match the job requirements",
        "Highlight any cloud computing experience (especially AWS)",
        "Emphasize financial data analysis experience more prominently",
        "Consider adding relevant certifications mentioned in the job posting"
      ],
      "Impact_Statements": [
        "Add quantified achievements related to machine learning model performance",
        "Include metrics about data size handled or processing efficiency improvements"
      ],
      "ATS_Optimization_Tips": [
        "Place key missing terms in the summary section",
        "Rephrase experience bullets to include critical missing keywords",
        "Use the exact terminology from the job description where possible"
      ],
      "Overall_Feedback": "The resume shows good alignment with the job description (75% match), particularly in programming experience and general data science skills. The strongest match is in the experience section, with relevant roles in data science. The weakest area is the lack of specific technical skills like TensorFlow and cloud computing experience. To improve the match, focus on adding the missing keywords, highlighting cloud experience if available, and quantifying achievements in terms of business impact. Consider also emphasizing any experience with financial data to better match the industry requirements."
    }

    Only return valid JSON and no extra text.
    """

    # Buttons for different actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        process_resume = st.button("Process Resume(s)")
    
    with col2:
        check_ats = st.button("Check ATS Score")
    
    with col3:
        clear_data = st.button("Clear All Data")

    # Job description input (for job-specific analysis)
    job_description = st.text_area("Paste job description for job-specific analysis:")

    if process_resume or check_ats:
        if not check_rate_limit():
            st.stop()

        if uploaded_files:
            if process_resume:
                st.session_state.batch_processing = True
                progress_bar = st.progress(0)
                processed_count = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            pdf_content = input_pdf_setup(uploaded_file)
                            response = get_gemini_response(input_prompt1, pdf_content)
                            
                            if response:
                                response["Filename"] = uploaded_file.name  # Track source file
                                new_data = pd.DataFrame([response])
                                
                                # Check for duplicates by email
                                if not st.session_state.resume_data.empty:
                                    if response["Email"] not in st.session_state.resume_data["Email"].values:
                                        st.session_state.resume_data = pd.concat(
                                            [st.session_state.resume_data, new_data], 
                                            ignore_index=True
                                        )
                                        processed_count += 1
                                else:
                                    st.session_state.resume_data = new_data
                                    processed_count += 1
                                
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    except Exception as e:
                        st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
                
                st.success(f"Processed {processed_count}/{len(uploaded_files)} resumes successfully!")
                st.session_state.batch_processing = False
            
            elif check_ats and len(uploaded_files) == 1:
                # Original single-file ATS check functionality
                pdf_content = input_pdf_setup(uploaded_files[0])
                response = get_gemini_response(input_prompt2, pdf_content)
                
                if response:
                    st.subheader("ATS Score and Feedback")
                    st.write(f"**ATS Score:** {response.get('ATS_Score', 'N/A')}")
                    
                    st.write("### Strengths:")
                    for strength in response.get("Strengths", []):
                        st.write(f"- {strength}")
                    
                    st.write("### Areas for Improvement:")
                    for improvement in response.get("Improvements", []):
                        st.write(f"- {improvement}")
                    
                    st.write("### Overall Feedback:")
                    st.write(response.get("Overall_Feedback", ""))
            
            elif check_ats and len(uploaded_files) > 1:
                st.warning("⚠️ ATS scoring is only available for single files. Please upload just one file for ATS analysis.")
        
        else:
            st.error("Please upload at least one PDF file")

    # Job-specific analysis (works with single file)
    if job_description and len(uploaded_files) == 1:
        if st.button("Check Job Match Score"):
            pdf_content = input_pdf_setup(uploaded_files[0])
            full_prompt = f"{input_prompt3}\n\nJob Description:\n{job_description}"
            response = get_gemini_response(full_prompt, pdf_content)
            
            if response:
                st.subheader("Job Match Analysis")
                st.write(f"**Match Score:** {response.get('Job_Match_Score', 'N/A')}")
                
                st.write("### Matching Keywords:")
                st.write(", ".join(response.get("Matching_Keywords", [])))
                
                st.write("### Missing Keywords:")
                st.write(", ".join(response.get("Missing_Keywords", [])))
                
                st.write("### Strengths:")
                for strength in response.get("Strengths", []):
                    st.write(f"- {strength}")
                
                st.write("### Improvements:")
                for improvement in response.get("Improvements", []):
                    st.write(f"- {improvement}")
                
                st.write("### Overall Feedback:")
                st.write(response.get("Overall_Feedback", ""))

    # Display and download data
    if not st.session_state.resume_data.empty:
        st.subheader("Processed Resume Data")
        st.dataframe(st.session_state.resume_data)
        
        # Excel download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            st.session_state.resume_data.to_excel(writer, index=False)
        output.seek(0)
        
        st.download_button(
            label="Download Excel",
            data=output,
            file_name="resume_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if clear_data:
        st.session_state.resume_data = pd.DataFrame()
        st.success("All data cleared!")

else:
    st.warning("Please log in to access the app.")