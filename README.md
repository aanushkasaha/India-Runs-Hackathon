# AI-Powered Recruitment Pipeline: Architecture Overview

This document outlines the end-to-end architecture of the AI-Powered Recruitment Dashboard. The system acts as an intelligent Applicant Tracking System (ATS) that parses resumes, calculates semantic similarity to a job description, and utilizes multi-agent LLM evaluations to rank and critique candidates.

---

## System Infrastructure
The project is built on a decoupled, two-tier architecture:
* **Frontend (Streamlit):** Handles the user interface, file batch uploads, dynamic database switching, and visualization of the final AI evaluation reports.
* **Backend (FastAPI):** A headless background API server that specifically handles the extraction and structuring of raw PDF data into structured JSON profiles using Google's Gemini LLM.

---

## Pipeline Execution Phases

### Phase 1: Data Ingestion & Resume Parsing
The system accommodates two distinct data pathways, seamlessly normalizing both into a standardized evaluation pool.

* **Batch PDF Processing:** Users can upload multiple PDF resumes simultaneously via the Streamlit UI. The frontend sends these files to the FastAPI server (`/extract-resume/` endpoint).
* **AI Extraction:** The backend uses the Gemini API to parse the raw PDF text, extracting hard skills, soft skills, and total years of experience, returning a structured JSON response.
* **Database Parsing:** Alternatively (or concurrently), the system can load a baseline static array of applicants from a `sample_candidates.json` dataset.

### Phase 2: Vector Retrieval & Semantic Search
Once the candidate pool is normalized, the system identifies the most relevant applicants based on the target Job Description.

* **Embedding Engine:** Uses `SentenceTransformer('all-MiniLM-L6-v2')` to convert candidate text (headlines, summaries, and combined skills) into dense vector embeddings.
* **ChromaDB Integration:** Candidate vectors and metadata (experience, name) are loaded into a local ChromaDB collection.
* **Query Execution:** The Job Description is embedded and searched against the ChromaDB collection, retrieving the Top 50 nearest neighbor candidates based on raw semantic similarity.

### Phase 3: Composite Signal Scoring
Relying solely on vector similarity can cause edge-case inaccuracies. Phase 3 normalizes the ChromaDB distance scores and introduces weighted deterministic metrics to calculate a final score.

* **Vector Normalization:** Raw distance metrics are converted into a $0$ to $1$ scale.
* **Experience Matching:** Normalizes the candidate's years of experience against a baseline (e.g., $20$ years).
* **Custom Signals:** Integrates mock third-party data, such as a GitHub Activity Score.
* **Final Calculation:** Candidates are sorted based on the following composite weight distribution:
  
  $$Final\_Score = (0.5 \times Vector\_Score) + (0.3 \times Experience\_Match) + (0.2 \times Signal\_Score)$$

### Phase 4: Gemini Reranking & Multi-Agent Evaluation
The top 3 candidates emerging from Phase 3 undergo a deep-dive qualitative analysis. The system orchestrates a dual-agent LLM review using `gemini-3.1-flash-lite` to ensure unbiased, balanced assessments.

* **Agent 1 (The Advocate):** Prompted to act as a "Lead Talent Acquisition Specialist." It generates a *Hire Justification*, highlighting key achievements, technical overlap, and providing a structural fit score out of 10.
* **Agent 2 (The Skeptic):** Prompted to act as an "Objective Hiring Manager." It generates an *Adversarial Critique*, focusing on growth areas, timeline gaps, and mitigating factors, while suggesting specific technical interview questions to validate the candidate's actual depth.
* **Rate Limiting:** Incorporates strict execution delays (`time.sleep(8)`) to maintain compliance with API free-tier quotas.
* **Strict Schema:** Validates the dual-agent outputs through a Pydantic `FinalCandidateResult` model to ensure the UI renders the data cleanly without formatting breaks.

---

## Execution Flow Summary
1. User defines Job Description & uploads `n` PDF resumes.
2. `FastAPI` extracts data $\rightarrow$ bridged to main array.
3. `ChromaDB` embeds array $\rightarrow$ retrieves Top 50 semantic matches.
4. `Python` math logic calculates composite scores $\rightarrow$ filters to Top 3.
5. `Gemini` runs dual-agent analysis on Top 3 $\rightarrow$ passes justified evaluations to Streamlit UI.
