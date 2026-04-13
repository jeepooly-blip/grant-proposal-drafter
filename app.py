import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
import streamlit as st

# =============================================================================
# API KEY CONFIGURATION
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    except:
        pass

if not GROQ_API_KEY:
    st.error("API key not configured!")
    st.info("Add GROQ_API_KEY to secrets or .env file")
    st.stop()

# Configure for Groq (OpenAI compatible)
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.3-70b-versatile"

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
    rfp_analyst = Agent(
        role="Senior Grant RFP Analyst",
        goal="Extract Problem Statement and Requirements from the RFP.",
        backstory="You are a meticulous grant analyst with 15 years of experience.",
        verbose=True,
        allow_delegation=False,
    )

    proposal_writer = Agent(
        role="Expert Grant Proposal Writer",
        goal="Write a compelling 300-word Executive Summary for a grant proposal.",
        backstory="You have secured over $50 million in grants.",
        verbose=True,
        allow_delegation=False,
    )

    extract_task = Task(
        description=f"RFP TEXT:\n{pdf_text}\n\nExtract: 1. Problem Statement 2. Key Requirements",
        expected_output="Problem Statement and Requirements",
        agent=rfp_analyst,
    )

    write_task = Task(
        description=f"Company: {company_description}\n\nWrite 300-word Executive Summary. Include: hook, problem, solution, impact.",
        expected_output="Executive Summary",
        agent=proposal_writer,
        context=[extract_task],
    )

    crew = Crew(
        agents=[rfp_analyst, proposal_writer],
        tasks=[extract_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    return str(crew.kickoff())


# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Grant Proposal Drafter", page_icon="🎯", layout="centered")

st.title("🎯 Grant Proposal Drafter")
st.markdown("*Free with Groq AI - Generate executive summaries in seconds*")

if GROQ_API_KEY:
    st.success("✅ API Ready - Generate proposals for FREE!")

uploaded_file = st.file_uploader("Upload Grant RFP (PDF or TXT)", type=["pdf", "txt"])
company_desc = st.text_area("Describe Your Organization", height=150)

if st.button("🚀 Generate Executive Summary", disabled=(not uploaded_file or not company_desc)):
    with st.spinner("Processing..."):
        text = extract_text_from_file(uploaded_file)
        result = run_crew(text, company_desc)
    
    st.subheader("📝 Executive Summary")
    st.write(result)
    st.success("Done!")

st.markdown("---")
st.caption("Powered by Groq (Free) · CrewAI · Streamlit · Llama 3.3 70B")
