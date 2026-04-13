# =============================================================================
# GRANT PROPOSAL DRAFTER — Streamlit + CrewAI (Final Clean Version)
# =============================================================================

import os
import tempfile

# ── LOAD KEYS ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── FORCE SET TAVILY API KEY (Windows Fix) ───────────────────────────────────
# This ensures the key is available even if .env doesn't load properly
os.environ["TAVILY_API_KEY"] = "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"

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import streamlit as st

# ── PDF text extraction ───────────────────────────────────────────────────────
try:
    import PyPDF2
except ImportError:
    st.error("Missing library: run `pip install PyPDF2`")
    st.stop()

# ── CrewAI core ───────────────────────────────────────────────────────────────
try:
    from crewai import Agent, Crew, Process, Task
    from crewai_tools import TXTSearchTool
except ImportError:
    st.error("Missing library: run `pip install crewai crewai-tools`")
    st.stop()

# ── TAVILY SEARCH TOOL SETUP ──────────────────────────────────────────────────
TAVILY_AVAILABLE = False
TavilySearchResults = None

try:
    from crewai_tools import TavilySearchResults
    # Test if the API key works by trying to create an instance
    test_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=1)
    TAVILY_AVAILABLE = True
    print("✅ Tavily search is successfully initialized")
except ImportError:
    st.warning("Tavily package not installed. Run: pip install tavily-python")
except Exception as e:
    st.warning(f"Tavily initialization issue: {str(e)}")


# =============================================================================
# HELPER: Extract text from an uploaded PDF
# =============================================================================

def extract_pdf_text(uploaded_file) -> str:
    """
    Saves the uploaded Streamlit file to a temp location,
    reads all pages with PyPDF2, and returns the full text.
    """
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

    os.unlink(tmp_path)  # clean up temp file
    return "\n\n".join(text_chunks)


# =============================================================================
# HELPER: Build and run the CrewAI crew
# =============================================================================

def run_crew(pdf_text: str, company_description: str) -> str:
    """
    Creates a two-agent CrewAI crew.
    Returns the final output string.
    """

    # ── Tool Setup ───────────────────────────────────────────────────────────
    search_tools = []
    if TAVILY_AVAILABLE and TavilySearchResults:
        try:
            # Explicitly pass the API key to bypass any environment issues
            tavily_tool = TavilySearchResults(
                api_key=os.getenv("TAVILY_API_KEY"),
                max_results=3
            )
            search_tools = [tavily_tool]
            print("✅ Tavily search tool added to agents")
        except Exception as e:
            st.warning(f"Could not initialize Tavily search: {e}")

    # ── Agent A: RFP Analyst ──────────────────────────────────────────────────
    rfp_analyst = Agent(
        role="Senior Grant RFP Analyst",
        goal=(
            "Carefully read the provided RFP document text and extract "
            "the core Problem Statement and all explicit Requirements "
            "the applicant must address."
        ),
        backstory=(
            "You are a meticulous grant analyst with 15 years of experience "
            "dissecting government and foundation RFPs. You excel at spotting "
            "the hidden requirements buried in dense bureaucratic language."
        ),
        tools=search_tools,      
        verbose=True,
        allow_delegation=False,
    )

    # ── Agent B: Proposal Writer ──────────────────────────────────────────────
    proposal_writer = Agent(
        role="Expert Grant Proposal Writer",
        goal=(
            "Write a compelling, precisely 300-word Executive Summary for a "
            "grant proposal that aligns the applicant company's strengths with "
            "the RFP's problem statement and requirements."
        ),
        backstory=(
            "You are an award-winning proposal writer who has helped nonprofits "
            "and startups secure over $50 million in grants. Your executive "
            "summaries are known for their clarity, persuasiveness, and perfect "
            "alignment with funder priorities."
        ),
        tools=search_tools,
        verbose=True,
        allow_delegation=False,
    )

    # ── Task 1: Extract RFP content ───────────────────────────────────────────
    extract_task = Task(
        description=(
            f"Below is the full text of a Grant RFP document:\n\n"
            f"---BEGIN RFP TEXT---\n{pdf_text}\n---END RFP TEXT---\n\n"
            "Please extract and clearly structure:\n"
            "1. **Problem Statement**: What problem or need does the funder want addressed?\n"
            "2. **Key Requirements**: Bullet-point list of all explicit requirements, "
            "eligibility criteria, and deliverables the applicant must meet."
        ),
        expected_output=(
            "A structured markdown document with two sections:\n"
            "## Problem Statement\n"
            "(paragraph summary)\n\n"
            "## Key Requirements\n"
            "(bullet-point list)"
        ),
        agent=rfp_analyst,
    )

    # ── Task 2: Write the Executive Summary ───────────────────────────────────
    write_task = Task(
        description=(
            "Using the extracted RFP Problem Statement and Requirements from the "
            "previous task, write a professional 300-word Executive Summary for a "
            "grant proposal.\n\n"
            f"**Applicant Company Description:**\n{company_description}\n\n"
            "Guidelines:\n"
            "- Open with a strong hook referencing the funder's stated problem.\n"
            "- Clearly introduce the applicant company and its relevant expertise.\n"
            "- Explain HOW the company's solution addresses the RFP requirements.\n"
            "- Close with the expected impact and a call-to-action.\n"
            "- Must be exactly ~300 words. Professional, persuasive tone."
        ),
        expected_output=(
            "A polished Executive Summary of approximately 300 words, "
            "ready to be pasted directly into a grant proposal document."
        ),
        agent=proposal_writer,
        context=[extract_task],   
    )

    # ── Assemble and run the Crew ─────────────────────────────────────────────
    crew = Crew(
        agents=[rfp_analyst, proposal_writer],
        tasks=[extract_task, write_task],
        process=Process.sequential,   
        verbose=True,
    )

    result = crew.kickoff()

    # CrewAI ≥ 0.80 returns a CrewOutput object; earlier versions return a str
    return str(result) if not isinstance(result, str) else result


# =============================================================================
# STREAMLIT UI
# =============================================================================

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grant Proposal Drafter",
    page_icon="📄",
    layout="centered",
)

# ── Custom CSS for a clean, professional look ─────────────────────────────────
st.markdown(
    """
    <style>
    /* Main font & background */
    html, body, [class*="css"] {
        font-family: 'Georgia', serif;
        background-color: #f7f5f0;
    }

    /* Header banner */
    .hero {
        background: linear-gradient(135deg, #1a3c5e 0%, #2e6da4 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .hero h1 { margin: 0 0 0.4rem 0; font-size: 2rem; }
    .hero p  { margin: 0; opacity: 0.85; font-size: 1rem; }

    /* Section cards */
    .card {
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }

    /* Result area */
    .result-box {
        background: #eef4fb;
        border-left: 4px solid #2e6da4;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        white-space: pre-wrap;
        font-family: 'Georgia', serif;
        font-size: 0.97rem;
        line-height: 1.7;
    }

    /* Warning banner */
    .api-warning {
        background: #fff8e1;
        border: 1px solid #f9a825;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
    }
    
    /* Success banner */
    .success-banner {
        background: #e8f5e9;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>📄 Grant Proposal Drafter</h1>
        <p>Upload a Grant RFP, describe your company, and let AI agents draft
        your Executive Summary in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── API-key guard ─────────────────────────────────────────────────────────────
openai_key = os.getenv("OPENAI_API_KEY", "")
if not openai_key:
    st.markdown(
        '<div class="api-warning">⚠️ <strong>OPENAI_API_KEY</strong> environment '
        "variable not set. The crew will fail without it. "
        "Set it in your terminal before running: "
        "<code>export OPENAI_API_KEY=sk-...</code></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
else:
    st.markdown(
        '<div class="success-banner">✅ OpenAI API key detected</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

# ── Tavily Status Display ─────────────────────────────────────────────────────
if TAVILY_AVAILABLE:
    st.markdown(
        '<div class="success-banner">🔍 Tavily search tool is ACTIVE - Agents can search the web for additional context!</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="api-warning">💡 Tavily search is disabled. Agents will rely solely on the PDF content. To enable web search, install tavily-python: pip install tavily-python</div>',
        unsafe_allow_html=True,
    )

st.markdown("")  # Add spacing

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Step 1 — Upload the Grant RFP")
uploaded_pdf = st.file_uploader(
    label="Upload a PDF",
    type=["pdf"],
    help="The RFP / Notice of Funding Opportunity you are responding to.",
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Step 2 — Describe Your Company")
company_desc = st.text_area(
    label="Company Description",
    placeholder=(
        "e.g. We are a 5-year-old social enterprise that provides affordable "
        "solar-powered water purification systems to rural communities in Sub-Saharan Africa. "
        "Our technology has reached 80,000 people across 3 countries..."
    ),
    height=160,
    help="Include your mission, key capabilities, past achievements, and target population.",
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Run button ────────────────────────────────────────────────────────────────
run_btn = st.button(
    "🚀 Generate Executive Summary",
    use_container_width=True,
    type="primary",
    disabled=(not uploaded_pdf or not company_desc.strip()),
)

# ── Execution & output ────────────────────────────────────────────────────────
if run_btn:
    if not openai_key:
        st.error(
            "Cannot run without OPENAI_API_KEY. "
            "Set the environment variable and restart the app."
        )
        st.stop()

    # ── Extract PDF text ──────────────────────────────────────────────────────
    with st.spinner("📖 Reading and extracting text from the PDF…"):
        try:
            pdf_text = extract_pdf_text(uploaded_pdf)
        except Exception as exc:
            st.error(f"Failed to read the PDF: {exc}")
            st.stop()

    if not pdf_text.strip():
        st.error(
            "Could not extract any text from the uploaded PDF. "
            "Please ensure it is not a scanned image-only document."
        )
        st.stop()

    st.markdown(
        f"✅ Extracted **{len(pdf_text):,} characters** from the PDF. "
        "Handing off to the AI crew…"
    )

    # ── Run CrewAI ────────────────────────────────────────────────────────────
    progress_placeholder = st.empty()
    with st.spinner(
        "🤖 Agents are working… (Agent A: analysing RFP → Agent B: drafting summary)"
    ):
        try:
            final_output = run_crew(pdf_text, company_desc)
        except Exception as exc:
            st.error(f"CrewAI encountered an error: {exc}")
            st.stop()

    progress_placeholder.empty()

    # ── Display result ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Generated Executive Summary")
    st.markdown(
        f'<div class="result-box">{final_output}</div>',
        unsafe_allow_html=True,
    )

    # Copy-friendly text area
    st.text_area(
        label="Copy the text below",
        value=final_output,
        height=300,
        help="Select all (Ctrl+A / Cmd+A) and copy.",
    )

    st.success("✅ Done! Review the Executive Summary above and paste it into your proposal.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Powered by [CrewAI](https://www.crewai.com/) · "
    "[LangChain](https://www.langchain.com/) · "
    "[Streamlit](https://streamlit.io/) · "
    "OpenAI GPT-4o"
)