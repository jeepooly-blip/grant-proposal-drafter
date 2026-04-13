import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st

# =============================================================================
# API KEY CONFIGURATION - WORKS FOR LOCAL & CLOUD
# =============================================================================

# First try to get from .env file (local development)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# If not found, try to get from Streamlit Secrets (cloud deployment)
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    except:
        pass

# If still no key, show error
if not GROQ_API_KEY:
    st.error("API key not configured!")
    st.info("Create a .env file with: GROQ_API_KEY=gsk_your_key_here")
    st.stop()

# Configure Groq for CrewAI
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama-3.3-70b-versatile"

# =============================================================================
# IMPORTS
# =============================================================================

try:
    import PyPDF2
except ImportError:
    st.error("Missing library: run pip install PyPDF2")
    st.stop()

try:
    from crewai import Agent, Crew, Process, Task
except ImportError:
    st.error("Missing library: run pip install crewai")
    st.stop()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_text_from_file(uploaded_file) -> str:
    """Extracts text from either PDF or TXT files"""
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
    """Creates and runs the CrewAI agents"""
    
    rfp_analyst = Agent(
        role="Senior Grant RFP Analyst",
        goal="Carefully read the RFP and extract the Problem Statement and all Requirements.",
        backstory="You are a meticulous grant analyst with 15 years of experience.",
        verbose=True,
        allow_delegation=False,
    )

    proposal_writer = Agent(
        role="Expert Grant Proposal Writer",
        goal="Write a compelling 300-word Executive Summary for a grant proposal.",
        backstory="You are an award-winning proposal writer who has secured over $50 million in grants.",
        verbose=True,
        allow_delegation=False,
    )

    extract_task = Task(
        description=(
            f"Below is the full text of a Grant RFP document:\n\n"
            f"---BEGIN RFP TEXT---\n{pdf_text}\n---END RFP TEXT---\n\n"
            "Extract and structure:\n"
            "1. Problem Statement\n"
            "2. Key Requirements (bullet points)"
        ),
        expected_output="Problem Statement and Key Requirements in markdown format.",
        agent=rfp_analyst,
    )

    write_task = Task(
        description=(
            f"Write a professional 300-word Executive Summary.\n\n"
            f"Company Description:\n{company_description}\n\n"
            "Include: hook, problem, solution, impact, call to action."
        ),
        expected_output="Executive Summary of approximately 300 words.",
        agent=proposal_writer,
        context=[extract_task],
    )

    crew = Crew(
        agents=[rfp_analyst, proposal_writer],
        tasks=[extract_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Grant Proposal Drafter",
    page_icon="🎯",
    layout="centered",
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1a3c5e 0%, #2e6da4 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
}
.free-badge {
    background: #4caf50;
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    display: inline-block;
    font-size: 0.8rem;
    margin-bottom: 1rem;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 1.5rem;
}
.result-box {
    background: #eef4fb;
    border-left: 4px solid #2e6da4;
    padding: 1.5rem;
    border-radius: 0 8px 8px 0;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎯 Grant Proposal Drafter</h1>
    <p>Upload a Grant RFP, describe your company, and let AI draft your Executive Summary.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="free-badge">✨ FREE - Powered by Groq (Llama 3.3 70B)</div>', unsafe_allow_html=True)
st.success("✅ Groq API key detected - Ready to generate proposals!")

# Input section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📄 Step 1: Upload the Grant RFP")
uploaded_file = st.file_uploader(
    "Choose PDF or TXT file",
    type=["pdf", "txt"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🏢 Step 2: Describe Your Organization")
company_desc = st.text_area(
    "Organization Description",
    placeholder="""Example:
Organization: Coastal Resilience Alliance
Mission: Empowering coastal communities to build climate resilience

Key achievements: 
- Trained 500+ community members
- Restored 200 hectares of mangroves
- Secured $1.2M in grants

Annual budget: $850,000
Team: 12 full-time staff, 30 volunteers""",
    height=180,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Run button
run_btn = st.button(
    "🚀 Generate Executive Summary",
    use_container_width=True,
    type="primary",
    disabled=(not uploaded_file or not company_desc.strip()),
)

# Execution
if run_btn:
    with st.spinner("📖 Reading your file..."):
        try:
            file_text = extract_text_from_file(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read file: {exc}")
            st.stop()

    if not file_text.strip():
        st.error("Could not extract text from file.")
        st.stop()

    st.info(f"✅ Extracted {len(file_text):,} characters. AI is working...")
    
    with st.spinner("🤖 Generating your executive summary (10-20 seconds)..."):
        try:
            final_output = run_crew(file_text, company_desc)
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()

    st.markdown("---")
    st.subheader("📝 Generated Executive Summary")
    st.markdown(f'<div class="result-box">{final_output}</div>', unsafe_allow_html=True)
    
    st.text_area("Copy the text below", value=final_output, height=200)
    st.success("✅ Done! Generated with Groq's free AI tier.")

st.markdown("---")
st.caption("Powered by CrewAI · Groq (Free) · Streamlit · Llama 3.3 70B")
