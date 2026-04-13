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

# =============================================================================
# IMPORTS
# =============================================================================

try:
    import PyPDF2
except ImportError:
    st.error("Missing PyPDF2")
    st.stop()

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    st.error("Missing google-generativeai")
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

def generate_proposal(pdf_text: str, company_description: str) -> str:
    """Generate proposal using Google Gemini directly (no CrewAI complexity)"""
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert grant proposal writer. Based on the following RFP and company description, write a professional Executive Summary of approximately 300 words.

    === RFP TEXT ===
    {pdf_text[:8000]}  # Limit to avoid token limits

    === COMPANY DESCRIPTION ===
    {company_description}

    === INSTRUCTIONS ===
    Write a compelling Executive Summary that:
    1. Opens with a strong hook referencing the problem
    2. Introduces the organization and its expertise
    3. Explains how the organization's solution addresses the RFP requirements
    4. Includes specific, measurable expected impact
    5. Ends with a confident call to action

    Use professional, persuasive language. Make every sentence count.
    """
    
    response = model.generate_content(prompt)
    return response.text

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Grant Proposal Drafter", page_icon="🎯", layout="centered")

st.title("🎯 Grant Proposal Drafter")
st.markdown("*Free with Google Gemini AI - Generate executive summaries in seconds*")

if GEMINI_API_KEY:
    st.success("✅ API Ready - Generate proposals for FREE!")

st.info("💡 Get your free Gemini API key at [Google AI Studio](https://aistudio.google.com/)")

uploaded_file = st.file_uploader("Upload Grant RFP (PDF or TXT)", type=["pdf", "txt"])
company_desc = st.text_area("Describe Your Organization", height=150, 
    placeholder="Organization name, mission, key achievements, budget, team size, past grants...")

if st.button("🚀 Generate Executive Summary", disabled=(not uploaded_file or not company_desc)):
    with st.spinner("📖 Reading file..."):
        text = extract_text_from_file(uploaded_file)
    
    st.info(f"✅ Extracted {len(text):,} characters. AI is working...")
    
    with st.spinner("🎯 Generating executive summary (10-20 seconds)..."):
        try:
            result = generate_proposal(text, company_desc)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    
    st.markdown("---")
    st.subheader("📝 Executive Summary")
    st.write(result)
    
    # Download button
    st.download_button(
        label="📥 Download as Text",
        data=result,
        file_name="grant_executive_summary.txt",
        mime="text/plain"
    )
    
    st.success("Done!")

st.markdown("---")
st.caption("Powered by Google Gemini AI · Streamlit")
