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
if "ats_scores_data" not in st.session_state:
    st.session_state.ats_scores_data = pd.DataFrame(columns=[
        "Name", "Email", "Mobile No", "Filename", "ATS_Score", 
        "Keyword_Score", "Formatting_Score", "Structure_Score", "Content_Score",
        "Job_Match_Score"  # Add job match score column
    ])

if login_button:
    if authenticate(username, password):
        st.session_state.authenticated = True
        st.sidebar.success("✅ Login successful!")
    else:
        st.sidebar.error("❌ Invalid username or password.")

if st.session_state.authenticated:
    # Add user type selection with tabs
    tab1, tab2 = st.tabs(["For Recruiters", "For Job Seekers"])
    
    # RECRUITER SECTION
    with tab1:
        st.header("Resume Analysis for Recruiters")
        st.write("Upload multiple resumes to extract information and compare candidates.")
        
        # Add custom column configuration section
        with st.expander("Custom Data Extraction Settings", expanded=False):
            st.write("Configure additional fields to extract from resumes. Name and Email will always be included.")
            
            # Default fields that are always included
            default_fields = ["Name", "Email"]
            
            # Optional fields with default selection
            optional_fields = {
                "Mobile No": True,
                "Address": True,
                "Education": True,
                "Skills": True,
                "Experience": True,
                "Languages": True
            }
            
            # Custom field input
            custom_fields = []
            
            # Create columns for optional fields
            col1, col2 = st.columns(2)
            
            # Let user toggle optional fields
            with col1:
                for field, default in list(optional_fields.items())[:len(optional_fields)//2]:
                    optional_fields[field] = st.checkbox(f"Extract {field}", value=default)
                    
            with col2:
                for field, default in list(optional_fields.items())[len(optional_fields)//2:]:
                    optional_fields[field] = st.checkbox(f"Extract {field}", value=default)
            
            # Let user add custom fields
            st.subheader("Add Custom Fields")
            custom_field_input = st.text_input("Enter custom field name (e.g., 'Projects', 'Certifications')")
            add_field = st.button("Add Field")
            
            # Store custom fields in session state if not already there
            if "custom_fields" not in st.session_state:
                st.session_state.custom_fields = []
                
            # Add new custom field when button is clicked
            if add_field and custom_field_input and custom_field_input not in st.session_state.custom_fields:
                st.session_state.custom_fields.append(custom_field_input)
                st.success(f"Added field: {custom_field_input}")
                
            # Display and let user remove custom fields
            if st.session_state.custom_fields:
                st.write("Current custom fields:")
                for i, field in enumerate(st.session_state.custom_fields):
                    cols = st.columns([3, 1])
                    cols[0].write(field)
                    if cols[1].button("Remove", key=f"remove_{i}"):
                        st.session_state.custom_fields.remove(field)
                        st.experimental_rerun()
        
        # Multiple file uploader for recruiters
        uploaded_files = st.file_uploader(
            "Upload multiple resumes (PDF)...", 
            type=["pdf"],
            accept_multiple_files=True,
            key="recruiter_files"
        )
        
        # Initialize session state for batch processing
        if "batch_processing" not in st.session_state:
            st.session_state.batch_processing = False
    
        # Build the dynamic input prompt based on selected fields (same as before)
        selected_fields = default_fields.copy()
        
        for field, selected in optional_fields.items():
            if selected:
                selected_fields.append(field)
                
        # Add custom fields
        if hasattr(st.session_state, "custom_fields"):
            selected_fields.extend(st.session_state.custom_fields)
        
        # Construct the extraction prompt dynamically
        extraction_fields_text = ""
        extraction_json_example = "{\n"
        
        for field in selected_fields:
            extraction_fields_text += f"- **{field}**  \n"
            
            # Add example values for the JSON structure
            if field == "Name":
                extraction_json_example += f'  "{field}": "John Doe",\n'
            elif field == "Email":
                extraction_json_example += f'  "{field}": "johndoe@example.com",\n'
            elif field == "Mobile No":
                extraction_json_example += f'  "{field}": "+1234567890",\n'
            elif field == "Address":
                extraction_json_example += f'  "{field}": "123 Main Street, New York, NY",\n'
            elif field == "Education":
                extraction_json_example += f'  "{field}": "Bachelor\'s Degree in Computer Science",\n'
            elif field == "Skills":
                extraction_json_example += f'  "{field}": "Python, Machine Learning, JavaScript",\n'
            elif field == "Experience":
                extraction_json_example += f'  "{field}": "Software Engineer at Google (2019-Present)",\n'
            elif field == "Languages":
                extraction_json_example += f'  "{field}": "English, Spanish",\n'
            else:
                extraction_json_example += f'  "{field}": "Example {field} information",\n'
        
        # Remove the trailing comma and close the JSON example
        extraction_json_example = extraction_json_example.rstrip(",\n") + "\n}"
        
        input_prompt1 = f"""
        You are an experienced Technical Human Resource Manager. Your task is to extract and list the following details from the provided resume:  

        {extraction_fields_text}

        Ensure the extracted details are returned as a JSON object in the following structure:
        {extraction_json_example}

        Only return valid JSON and no extra text.
        """

        # ATS compatibility analysis prompt
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

        # Job description matching prompt
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
            "Consider adding relevant certifications
        """
        
        # Job description input (for job-specific analysis)
        job_description = st.text_area("Paste job description for job-specific analysis (optional):", key="recruiter_jd")
        
        # Buttons for different actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            process_resume = st.button("Process Resume(s)", key="recruiter_process")
        
        with col2:
            clear_data = st.button("Clear All Data", key="recruiter_clear")
        
        # Process resumes for recruiters
        if process_resume:
            if not check_rate_limit():
                st.stop()

            if uploaded_files:
                st.session_state.batch_processing = True
                progress_bar = st.progress(0)
                processed_count = 0
                
                # Create a combined DataFrame for all resume data
                combined_data = pd.DataFrame()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            pdf_content = input_pdf_setup(uploaded_file)
                            
                            # Get basic resume info
                            info_response = get_gemini_response(input_prompt1, pdf_content)
                            
                            # Always get ATS score 
                            ats_response = get_gemini_response(input_prompt2, pdf_content)
                            
                            # If job description is provided, also get job-specific score
                            job_match_score = None
                            if job_description:
                                full_prompt = f"{input_prompt3}\n\nJob Description:\n{job_description}"
                                job_match_response = get_gemini_response(full_prompt, pdf_content)
                                if job_match_response and "Job_Match_Score" in job_match_response:
                                    job_match_score = job_match_response.get("Job_Match_Score")
                            
                            if info_response:
                                # Create a row that combines all data
                                combined_row = info_response.copy()
                                combined_row["Filename"] = uploaded_file.name
                                
                                # Add ATS scores directly to the combined row
                                if ats_response and "ATS_Score" in ats_response:
                                    combined_row["ATS_Score"] = ats_response.get("ATS_Score", 0)
                                    combined_row["Job_Match_Score"] = job_match_score if job_match_score is not None else "N/A"
                                    
                                    # Add category scores if available
                                    if "Category_Scores" in ats_response:
                                        cat_scores = ats_response["Category_Scores"]
                                        combined_row["Keyword_Score"] = cat_scores.get("Keyword_Relevance", 0)
                                        combined_row["Formatting_Score"] = cat_scores.get("Formatting_Parsing", 0) 
                                        combined_row["Structure_Score"] = cat_scores.get("Completeness_Structure", 0)
                                        combined_row["Content_Score"] = cat_scores.get("Clarity_Content", 0)
                                
                                # Add to combined DataFrame
                                new_data = pd.DataFrame([combined_row])
                                
                                # Check for duplicates by email
                                if not combined_data.empty:
                                    if combined_row["Email"] not in combined_data["Email"].values:
                                        combined_data = pd.concat([combined_data, new_data], ignore_index=True)
                                        processed_count += 1
                                else:
                                    combined_data = new_data
                                    processed_count += 1
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    except Exception as e:
                        st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
                
                # Store the combined data in session state
                st.session_state.resume_data = combined_data
                
                st.success(f"Processed {processed_count}/{len(uploaded_files)} resumes successfully!")
                st.session_state.batch_processing = False
            else:
                st.error("Please upload at least one PDF file")

        # Display and export combined data
        if not st.session_state.resume_data.empty:
            st.subheader("Processed Resume Data with ATS Scores")
            st.dataframe(st.session_state.resume_data)
            
            # Export combined data to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                st.session_state.resume_data.to_excel(writer, index=False)
            output.seek(0)
            
            st.download_button(
                label="Download Complete Resume Analysis",
                data=output,
                file_name="resume_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Clear data
        if clear_data:
            st.session_state.resume_data = pd.DataFrame()
            st.success("All data cleared!")
    
    # JOB SEEKER SECTION
    with tab2:
        st.header("Resume Optimization for Job Seekers")
        st.write("Upload your resume and get ATS compatibility scores and improvement suggestions.")
        
        # Single file uploader for job seekers
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF)...", 
            type=["pdf"],
            key="job_seeker_file"
        )
        
        # Job description input for job seekers
        job_description_seeker = st.text_area("Paste the job description you're applying for (optional):", 
                                              key="job_seeker_jd")
        
        # Buttons for job seeker actions
        col1, col2 = st.columns(2)
        with col1:
            check_ats = st.button("Check ATS Compatibility", key="job_seeker_ats")
        with col2:
            check_job_match = st.button("Check Job Match", key="job_seeker_match")
        
        # Process resume for job seekers
        if uploaded_file:
            if check_ats or check_job_match:
                if not check_rate_limit():
                    st.stop()
                    
                try:
                    with st.spinner("Analyzing your resume..."):
                        pdf_content = input_pdf_setup(uploaded_file)
                        
                        # Get basic info
                        info_response = get_gemini_response(input_prompt1, pdf_content)
                        
                        # For ATS check
                        if check_ats:
                            ats_response = get_gemini_response(input_prompt2, pdf_content)
                            
                            if ats_response and "ATS_Score" in ats_response:
                                st.subheader("ATS Compatibility Analysis")
                                
                                # Create columns for score display
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    # Display the overall ATS score prominently
                                    st.metric("ATS Score", f"{ats_response['ATS_Score']}/100")
                                    
                                    # Display category scores
                                    if "Category_Scores" in ats_response:
                                        st.subheader("Category Scores")
                                        cat_scores = ats_response["Category_Scores"]
                                        for category, score in cat_scores.items():
                                            st.metric(category.replace("_", " "), f"{score}/25")
                                
                                with col2:
                                    # Display resume strengths
                                    if "Strengths" in ats_response:
                                        st.subheader("Resume Strengths")
                                        for strength in ats_response["Strengths"]:
                                            st.markdown(f"✅ {strength}")
                                    
                                    # Display suggested improvements
                                    if "Improvements" in ats_response:
                                        st.subheader("Suggested Improvements")
                                        for improvement in ats_response["Improvements"]:
                                            st.markdown(f"🔍 {improvement}")
                                
                                # Display keyword analysis
                                if "Keyword_Analysis" in ats_response:
                                    st.subheader("Keyword Analysis")
                                    keyword_analysis = ats_response["Keyword_Analysis"]
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if "Detected_Keywords" in keyword_analysis:
                                            st.markdown("**Detected Keywords:**")
                                            st.write(", ".join(keyword_analysis["Detected_Keywords"]))
                                    
                                    with col2:
                                        if "Missing_Common_Keywords" in keyword_analysis:
                                            st.markdown("**Missing Common Keywords:**")
                                            st.write(", ".join(keyword_analysis["Missing_Common_Keywords"]))
                                
                                # Display format analysis
                                if "Format_Analysis" in ats_response:
                                    st.subheader("Format Analysis")
                                    format_analysis = ats_response["Format_Analysis"]
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if "ATS_Friendly_Elements" in format_analysis:
                                            st.markdown("**ATS-Friendly Elements:**")
                                            for item in format_analysis["ATS_Friendly_Elements"]:
                                                st.markdown(f"✅ {item}")
                                    
                                    with col2:
                                        if "ATS_Unfriendly_Elements" in format_analysis:
                                            st.markdown("**ATS-Unfriendly Elements:**")
                                            for item in format_analysis["ATS_Unfriendly_Elements"]:
                                                st.markdown(f"⚠️ {item}")
                                
                                # Display overall feedback
                                if "Overall_Feedback" in ats_response:
                                    st.subheader("Overall Feedback")
                                    st.write(ats_response["Overall_Feedback"])
                            else:
                                st.error("Failed to analyze ATS compatibility. Please try again.")
                        
                        # For Job Match check
                        if check_job_match:
                            if job_description_seeker:
                                full_prompt = f"{input_prompt3}\n\nJob Description:\n{job_description_seeker}"
                                job_match_response = get_gemini_response(full_prompt, pdf_content)
                                
                                if job_match_response and "Job_Match_Score" in job_match_response:
                                    st.subheader("Job Match Analysis")
                                    
                                    # Create columns for score display
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        # Display the overall job match score prominently
                                        st.metric("Job Match Score", f"{job_match_response['Job_Match_Score']}/100")
                                        
                                        # Display category scores
                                        if "Category_Scores" in job_match_response:
                                            st.subheader("Category Scores")
                                            cat_scores = job_match_response["Category_Scores"]
                                            for category, score in cat_scores.items():
                                                max_score = 25
                                                if "Education" in category:
                                                    max_score = 15
                                                elif "Overall" in category:
                                                    max_score = 10
                                                st.metric(category.replace("_", " "), f"{score}/{max_score}")
                                    
                                    with col2:
                                        # Display resume strengths
                                        if "Strengths" in job_match_response:
                                            st.subheader("Strengths")
                                            for strength in job_match_response["Strengths"]:
                                                st.markdown(f"✅ {strength}")
                                        
                                        # Display suggested improvements
                                        if "Improvements" in job_match_response:
                                            st.subheader("Suggested Improvements")
                                            for improvement in job_match_response["Improvements"]:
                                                st.markdown(f"🔍 {improvement}")
                                    
                                    # Display keyword match analysis
                                    if "Keyword_Analysis" in job_match_response:
                                        st.subheader("Keyword Analysis")
                                        keyword_analysis = job_match_response["Keyword_Analysis"]
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            if "Matching_Keywords" in keyword_analysis:
                                                st.markdown("**Matching Keywords:**")
                                                st.write(", ".join(keyword_analysis["Matching_Keywords"]))
                                        
                                        with col2:
                                            if "Missing_Keywords" in keyword_analysis:
                                                st.markdown("**Missing Keywords:**")
                                                st.write(", ".join(keyword_analysis["Missing_Keywords"]))
                                                
                                        if "Keyword_Match_Percentage" in keyword_analysis:
                                            st.info(f"**Match Rate:** {keyword_analysis['Keyword_Match_Percentage']}")
                                    
                                    # Display specific ATS tips
                                    if "ATS_Optimization_Tips" in job_match_response:
                                        st.subheader("ATS Optimization Tips")
                                        for tip in job_match_response["ATS_Optimization_Tips"]:
                                            st.markdown(f"💡 {tip}")
                                    
                                    # Display impact statements
                                    if "Impact_Statements" in job_match_response:
                                        st.subheader("Suggested Impact Statements")
                                        for statement in job_match_response["Impact_Statements"]:
                                            st.markdown(f"📈 {statement}")
                                    
                                    # Display overall feedback
                                    if "Overall_Feedback" in job_match_response:
                                        st.subheader("Overall Feedback")
                                        st.write(job_match_response["Overall_Feedback"])
                                else:
                                    st.error("Failed to analyze job match. Please try again.")
                            else:
                                st.warning("Please paste a job description to check the match.")
                    
                except Exception as e:
                    st.error(f"Error analyzing resume: {str(e)}")
else:
    st.warning("Please log in to access the app.")