import streamlit as st
import requests
import json
import time
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from google import genai

# ==========================================
# 1. PAGE SETUP & CACHING
# ==========================================
st.set_page_config(page_title="AI Recruitment Portal", layout="wide")

@st.cache_resource
def load_models():
    # Initializes the local embedding engine and vector database once
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="candidate_profiles")
    return model, collection

model, collection = load_models()

# !!! REPLACE WITH YOUR ACTUAL GEMINI API KEY !!!
client = genai.Client(api_key="AQ.Ab8RN6KomN5tv32CAqqEpQG-Lxsi6oj6TLkQj6FPThYiw07ORQ") 

# ==========================================
# 2. CORE PIPELINE FUNCTIONS
# ==========================================

class FinalCandidateResult(BaseModel):
    candidate_id: str
    final_rank: int
    hire_justification: str
    adversarial_critique: str

def add_and_search_candidates(json_data, search_query, top_k=50):
    documents, metadatas, ids = [], [], []
    for candidate in json_data:
        cand_id = candidate.get("candidate_id")
        profile = candidate.get("profile", {})
        text = f"Headline: {profile.get('headline', '')}. Summary: {profile.get('summary', '')}. " \
               f"Skills: {', '.join([s['name'] for s in candidate.get('skills', [])])}."
        documents.append(text)
        ids.append(cand_id)
        metadatas.append({
            "name": profile.get("anonymized_name", "Unknown"),
            "years_of_experience": profile.get("years_of_experience", 0.0)
        })
    
    embeddings = model.encode(documents).tolist()
    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    query_embed = model.encode([search_query]).tolist()
    return collection.query(query_embeddings=query_embed, n_results=top_k)

def calculate_composite_scores(results, candidate_full_data):
    scored_candidates = []
    ids, distances, metadatas = results['ids'][0], results['distances'][0], results['metadatas'][0]
    max_dist = max(distances) if max(distances) > 0 else 1
    vector_scores = [1 - (d / max_dist) for d in distances]
    
    for i in range(len(ids)):
        cand_id = ids[i]
        full_cand = next((c for c in candidate_full_data if c['candidate_id'] == cand_id), None)
        v_score = vector_scores[i]
        exp_match = min(metadatas[i].get('years_of_experience', 0) / 20.0, 1.0)
        github_score = max(0, full_cand['redrob_signals'].get('github_activity_score', 0)) / 10.0 
        final_score = (0.5 * v_score) + (0.3 * exp_match) + (0.2 * github_score)
        scored_candidates.append({
            "candidate_id": cand_id,
            "name": metadatas[i]['name'],
            "final_score": round(final_score, 4)
        })
    return sorted(scored_candidates, key=lambda x: x['final_score'], reverse=True)

def get_critique(prompt):
    time.sleep(8) # Protects free tier rate limits
    return client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)

def rerank_and_critique_candidates(top_candidates, candidate_full_data, job_description):
    final_results = []
    for rank, cand in enumerate(top_candidates, 1):
        full_cand = next((c for c in candidate_full_data if c['candidate_id'] == cand['candidate_id']), None)
        
        rec_prompt = f"""
        Role: Lead Talent Acquisition Specialist. 
        Task: Evaluate this candidate's fit for the following role: {job_description}.
        Candidate Profile: {full_cand}

        Please provide a structured, objective evaluation:
        1. Key Achievements: Highlight 2-3 major accomplishments, strong technical skills, or successful projects from their profile.
        2. Fit Score (1-10): Rate their overall alignment with the job description.
        3. Overall Reason: Provide a concise summary of exactly why this candidate is a strong fit for the company and what unique value they bring.
        """
        
        hat_prompt = f"""
        Role: Fair and Objective Hiring Manager. 
        Task: Conduct a constructive risk assessment for this candidate: {full_cand} against the job description: {job_description}.

        Please provide a balanced review:
        1. Growth Areas: Identify 1-2 missing skills, experience gaps, or timeline inconsistencies. Be objective and professional.
        2. Mitigating Factors: Point out any existing strengths or transferable skills the candidate has that could make up for these gaps.
        3. Interview Focus: Suggest 1-2 specific questions we should ask in an interview to validate their actual experience level.
        """
        
        st.write(f" Analyzing candidate data with Gemini: **{cand.get('name', 'Candidate')}**...")
        res1 = get_critique(rec_prompt)
        res2 = get_critique(hat_prompt)
        
        final_results.append(FinalCandidateResult(
            candidate_id=cand['candidate_id'],
            final_rank=rank,
            hire_justification=res1.text,
            adversarial_critique=res2.text
        ))
    return final_results

def format_api_response_for_pipeline(candidate_id, candidate_name, extraction_result):
    formatted_data = {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": candidate_name,
            "years_of_experience": extraction_result.experience_years,
            "headline": "Extracted via API", 
            "summary": "Candidate processed from uploaded PDF."
        },
        "skills": [{"name": skill} for skill in extraction_result.hard_skills],
        "redrob_signals": {
            "github_activity_score": 7.5
        }
    }
    return formatted_data

# ==========================================
# 3. USER INTERFACE & FLOW CONTROLLER
# ==========================================
st.title(" AI-Powered Recruitment Dashboard")

with st.sidebar:
    st.header("Pipeline Settings")
    job_desc = st.text_area("Target Job Description", "Looking for a backend data engineer experienced with Spark, Airflow, and Python.", height=150)
    st.divider()
    st.markdown("💡 *Ensure your backend terminal (`uvicorn main:app --reload`) is active before submitting files.*")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Applicant Resume (PDF)", type=["pdf"])
with col2:
    st.write("") 
    st.write("") 
    run_db_only = st.button("Run Analysis on Existing Database 🚀")

if uploaded_file is not None or run_db_only:
    with st.status("Processing Pipeline...", expanded=True) as status:
        
        st.write("📂 Loading database array from `sample_candidates.json`...")
        with open('sample_candidates.json', 'r') as file:
            candidate_data = json.load(file)
            
        if uploaded_file is not None:
            st.write("🧠 Forwarding PDF to FastAPI server for structural parsing...")
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post("http://localhost:8000/extract-resume/", files=files)
                if response.status_code == 200:
                    api_data = response.json()
                    
                    class ExtractedResult:
                        hard_skills = api_data['hard_skills']
                        soft_skills = api_data['soft_skills']
                        experience_years = api_data['experience_years']
                        
                    new_candidate = format_api_response_for_pipeline(
                        candidate_id="cand_new_upload",
                        candidate_name="New Applicant (From PDF)",
                        extraction_result=ExtractedResult()
                    )
                    candidate_data.append(new_candidate)
                    st.write("✅ Parsed PDF data successfully merged into evaluation pool.")
                else:
                    st.error(f"FastAPI Backend Error: {response.text}")
            except Exception as e:
                st.error(f"Unable to connect to FastAPI server. Error details: {e}")
        else:
            st.write("⏩ Skipping parsing window. Proceeding directly with database metrics.")

        st.write("🔍 Computing vector representations via ChromaDB...")
        raw_results = add_and_search_candidates(candidate_data, job_desc, top_k=50)
        final_list = calculate_composite_scores(raw_results, candidate_data)

        st.write("📈 Initiating agent review passes (includes built-in delay buffers)...")
        reranked_results = rerank_and_critique_candidates(final_list[:3], candidate_data, job_desc)
        
        status.update(label="Pipeline Processing Complete!", state="complete", expanded=False)

    # 4. GRAPHICAL DASHBOARD PRESENTATION
    st.subheader("🏆 Highly Matched Talent Recommendations")
    
    for res in reranked_results:
        with st.expander(f"🏅 RANK #{res.final_rank}: Candidate Profile ID [{res.candidate_id}]", expanded=(res.final_rank == 1)):
            view_col1, view_col2 = st.columns(2)
            
            with view_col1:
                st.markdown("#### 📈 Fit Analysis & Structural Advantages")
                st.info(res.hire_justification.strip())
                
            with view_col2:
                st.markdown("#### 🔍 Constructive Risk Profile & Gaps")
                st.warning(res.adversarial_critique.strip())