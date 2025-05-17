import streamlit as st
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

st.title("MindMatch AI 🧠💼")
st.markdown("Upload your resume & paste a JD to check fit and get a rewritten summary")

uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")
job_description = st.text_area("Paste Job Description")

if uploaded_file and job_description:
    # --- resume extraction ---
    import fitz
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    resume_text = "".join([page.get_text() for page in doc])

    # --- embeddings ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(job_description, convert_to_tensor=True)
    score = util.pytorch_cos_sim(resume_emb, jd_emb).item()

    st.markdown(f"### 🧠 Similarity Score: `{score:.2f}`")
    
    # --- feedback ---
    if score > 0.8:
        st.success("Excellent match! Your resume aligns well.")
    elif score > 0.6:
        st.warning("Good match, but improvements are possible.")
    else:
        st.error("Low match. Try editing the summary or adding keywords.")

    # --- Gemini rewrite ---
    genai.configure(api_key="__APIKEY__")
    gemini = genai.GenerativeModel('suitable__gemini')
    prompt = f"Rewrite this resume summary:\n{resume_text[:800]}\n\nTo match this job:\n{job_description}"
    rewritten = gemini.generate_content(prompt).text
    st.markdown("### ✍️ Rewritten Summary:")
    st.write(rewritten)
