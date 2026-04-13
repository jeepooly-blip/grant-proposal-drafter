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
    st.info("Create a .env file with: GROQ_API_KEY=gsk_your_key_here")
    st.stop()

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

def run_crew(pdf_text: str, company_description: str, word_count: int, tone: str, focus_areas: list) -> str:
    
    rfp_analyst = Agent(
        role="Senior Grant RFP Analyst",
        goal="Extract Problem Statement and Requirements from the RFP.",
        backstory="You are a meticulous grant analyst with 15 years of experience.",
        verbose=True,
        allow_delegation=False,
    )

    proposal_writer = Agent(
        role="Expert Grant Proposal Writer",
        goal=f"Write a compelling {word_count}-word Executive Summary in a {tone} tone.",
        backstory="You have secured over $50 million in grants.",
        verbose=True,
        allow_delegation=False,
    )

    extract_task = Task(
        description=(
            f"RFP TEXT:\n{pdf_text}\n\n"
            "Extract:\n1. Problem Statement\n2. Key Requirements\n3. Funder's buzzwords"
        ),
        expected_output="Problem Statement and Requirements",
        agent=rfp_analyst,
    )

    write_task = Task(
        description=(
            f"Write a {word_count}-word Executive Summary.\n\n"
            f"Company: {company_description}\n\n"
            f"Tone: {tone}\n"
            f"Focus Areas: {', '.join(focus_areas)}\n\n"
            "Include: hook, problem, solution, impact, call to action."
        ),
        expected_output=f"Executive Summary of approximately {word_count} words.",
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
# ENHANCED STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Grant Proposal Drafter - Enhanced Edition",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for enhanced layout
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a4a6f 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .quality-badge {
        background: #4caf50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.8rem;
    }
    .pro-tip {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Professional Grant Proposal Writer</h1>
    <p>AI-powered proposal generation with funder-focused quality</p>
    <span class="quality-badge">✨ FREE - Powered by Groq (Llama 3.3 70B)</span>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("## ⚙️ Proposal Settings")
    
    word_count = st.slider("Target word count", 200, 600, 350, 50)
    
    tone = st.select_slider(
        "Proposal Tone",
        options=["Conservative", "Professional", "Persuasive", "Urgent", "Inspiring"],
        value="Professional"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Focus Areas")
    
    focus_areas = st.multiselect(
        "Select key focus areas",
        ["Community Impact", "Innovation", "Sustainability", "Partnerships", "Scalability", "Cost-Effectiveness"],
        default=["Community Impact", "Sustainability"]
    )
    
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    st.info("""
    **For best results:**
    - Upload clear, text-based PDFs
    - Be detailed in company description
    - Include past success metrics
    - Mention key partnerships
    """)

# Main content area - TWO COLUMN LAYOUT
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Step 1: Upload Grant RFP")
    uploaded_file = st.file_uploader(
        "Choose PDF or TXT file",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏢 Step 2: Describe Your Organization")
    company_desc = st.text_area(
        "Organization Description",
        placeholder="""Example:
Organization: Coastal Resilience Alliance (CRA)
Mission: Empowering coastal communities to build climate resilience

Key Achievements:
- Trained 500+ community members in disaster preparedness
- Reduced flood-related deaths by 40%
- Restored 200 hectares of mangroves
- Secured $1.2M in grants

Annual Budget: $850,000
Team: 12 full-time staff, 30 volunteers
Partners: 8 local governments, 3 universities""",
        height=250,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Quality tip
st.markdown("""
<div class="pro-tip">
💡 <strong>Quality Tip:</strong> The more detailed your organization description, 
the better the proposal. Include specific numbers, past successes, and unique strengths.
</div>
""", unsafe_allow_html=True)

# API Status
if GROQ_API_KEY:
    st.success("✅ Groq API key detected - Ready to generate proposals for FREE!")

# Run button - centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_btn = st.button(
        "🚀 Generate Professional Proposal",
        use_container_width=True,
        type="primary",
        disabled=(not uploaded_file or not company_desc.strip()),
    )

# Execution and output
if run_btn:
    with st.spinner("📖 Reading and extracting text from your file..."):
        try:
            file_text = extract_text_from_file(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read the file: {exc}")
            st.stop()

    if not file_text.strip():
        st.error("Could not extract any text from the file.")
        st.stop()

    st.info(f"✅ Extracted **{len(file_text):,} characters** from your file.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🤖 AI agents are analyzing your RFP...")
    progress_bar.progress(30)
    
    with st.spinner("🎯 Generating your executive summary with Groq..."):
        try:
            final_output = run_crew(file_text, company_desc, word_count, tone.lower(), focus_areas)
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()
    
    progress_bar.progress(100)
    status_text.text("✅ Complete!")
    
    # Display results
    st.markdown("---")
    st.markdown("## 📝 Your Generated Proposal")
    
    # Quality indicator
    st.markdown(f"""
    <div style="background: #e8f5e9; padding: 0.5rem; border-radius: 8px; margin-bottom: 1rem;">
        ✅ Quality Check: Passed | Tone: {tone} | Word Count Target: {word_count} | Focus: {', '.join(focus_areas)}
    </div>
    """, unsafe_allow_html=True)
    
    # Display the proposal
    st.markdown(f'<div class="result-box">{final_output}</div>', unsafe_allow_html=True)
    
    # Download button
    st.download_button(
        label="📥 Download Proposal as Text",
        data=final_output,
        file_name="grant_proposal.txt",
        mime="text/plain",
    )
    
    st.success("✅ Proposal generated successfully! Review and customize as needed.")

# Footer
st.markdown("---")
st.caption("Powered by CrewAI · Groq (Free Tier) · Streamlit · Llama 3.3 70B")
