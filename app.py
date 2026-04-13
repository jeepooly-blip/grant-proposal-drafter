import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
import streamlit as st

# =============================================================================
# API KEY CONFIGURATION - Google Gemini
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except:
        pass

if not GEMINI_API_KEY:
    st.error("API key not configured!")
    st.info("Get your free key from: https://aistudio.google.com/")
    st.stop()

# Configure for Gemini
os.environ["OPENAI_API_KEY"] = "dummy"  # CrewAI requires this but won't use it
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# =============================================================================
# IMPORTS
# =============================================================================

try:
    import PyPDF2
except ImportError:
    st.error("Missing PyPDF2")
    st.stop()

try:
    from crewai import Agent, Crew, Process, Task
except ImportError:
    st.error("Missing crewai")
    st.stop()

# =============================================================================
# FUNCTIONS
# =============================================================================

def extract_text_from_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    
    if file_name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        text_chunks = []
        with open(tmp_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)

        os.unlink(tmp_path)
        return "\n\n".join(text_chunks)
    
    raise ValueError("Please upload PDF or TXT files")

def run_crew(pdf_text: str, company_description: str) -> str:
    # Use Google Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
    )
    
    rfp_analyst = Agent(
        role="Senior Grant RFP Analyst",
        goal="Extract Problem Statement and Requirements from the RFP.",
        backstory="You are a meticulous grant analyst with 15 years of experience.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
    )

    proposal_writer = Agent(
        role="Expert Grant Proposal Writer",
        goal="Write a compelling 300-word Executive Summary for a grant proposal.",
        backstory="You have secured over $50 million in grants.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
    )

    extract_task = Task(
        description=f"RFP TEXT:\n{pdf_text}\n\nExtract: 1. Problem Statement 2. Key Requirements (bullet points)",
        expected_output="Problem Statement and Key Requirements",
        agent=rfp_analyst,
    )

    write_task = Task(
        description=f"Company: {company_description}\n\nWrite a professional 300-word Executive Summary. Include: hook, problem, solution, expected impact, call to action.",
        expected_output="Executive Summary of approximately 300 words",
        agent=proposal_writer,
        context=[extract_task],
    )

    crew = Crew(
        agents=[rfp_analyst, proposal_writer],
        tasks=[extract_task, write_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result)


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Grant Proposal Drafter", page_icon="🎯", layout="centered")

st.title("🎯 Grant Proposal Drafter")
st.markdown("*Free with Google Gemini AI - Generate executive summaries in seconds*")

if GEMINI_API_KEY:
    st.success("✅ API Ready - Generate proposals for FREE!")

uploaded_file = st.file_uploader("Upload Grant RFP (PDF or TXT)", type=["pdf", "txt"])
company_desc = st.text_area("Describe Your Organization", height=150, 
    placeholder="Organization name, mission, key achievements, budget, team size...")

if st.button("🚀 Generate Executive Summary", disabled=(not uploaded_file or not company_desc)):
    with st.spinner("📖 Reading file..."):
        text = extract_text_from_file(uploaded_file)
    
    st.info(f"✅ Extracted {len(text):,} characters. AI is working...")
    
    with st.spinner("🎯 Generating executive summary (10-20 seconds)..."):
        try:
            result = run_crew(text, company_desc)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    
    st.markdown("---")
    st.subheader("📝 Executive Summary")
    st.write(result)
    st.success("Done!")

st.markdown("---")
st.caption("Powered by Google Gemini · CrewAI · Streamlit")
