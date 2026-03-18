from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pdfplumber
import endee
import os
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

# Initialize Models
embed_model = None
tokenizer = None
model = None

try:
    print("Loading Embedding Model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading FLAN-T5-Small Model...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # type: ignore
    print("Models loaded successfully!")
except Exception as e:
    print(f"CRITICAL: Model loading failed: {e}")
    print("The app will start, but AI features might be disabled.")

# Endee client
db = endee.Endee()

# --- UTILS ---

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_skills(text):
    # Expanded skill list for better coverage
    skills_db = [
        "python", "java", "javascript", "react", "sql", "machine learning", 
        "html", "css", "node.js", "express", "mongodb", "aws", "docker", 
        "kubernetes", "typescript", "c++", "c#", "php", "swift", "go", 
        "ruby", "git", "rest api", "flask", "django", "tableau", "power bi"
    ]
    found = [skill for skill in skills_db if skill.lower() in text.lower()]
    return list(set(found)) # Use set to remove duplicates if any

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    request_files = request.files
    if "file" not in request_files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    # Fast PDF extraction
    resume_text = extract_text_from_pdf(path)

    # Save to data/resume.txt as requested
    with open("data/resume.txt", "w", encoding="utf-8") as f:
        f.write(resume_text)

    # Optional: Also try to update Endee if it's running
    try:
        if embed_model is not None and db is not None:
            index = db.get_index("resumes")
            if index is not None:
                all_chunks = [c.strip() for c in resume_text.split("\n") if c.strip()]
                for i, chunk in enumerate(all_chunks[:20]): # type: ignore
                    v = embed_model.encode(chunk).tolist()
                    index.upsert([{"id": f"{file.filename}_{i}", "vector": v, "meta": {"text": chunk}}])
    except Exception as e:
        print(f"Endee sync skipped: {e}")

    return jsonify({"message": "Resume uploaded and text extracted!", "skills": extract_skills(resume_text)})

@app.route("/skills")
def get_skills():
    try:
        with open("data/resume.txt", "r", encoding="utf-8") as f:
            text = f.read()
        return jsonify({"skills": extract_skills(text)})
    except FileNotFoundError:
        return jsonify({"error": "No resume found. Please upload one first."}), 404

def ats_score(resume_text, job_text):

    skills = [
        "python",
        "java",
        "javascript",
        "react",
        "sql",
        "machine learning",
        "docker",
        "aws",
        "html",
        "css"
    ]

    resume_skills = []
    job_skills = []

    for skill in skills:

        if skill in resume_text.lower():
            resume_skills.append(skill)

        if skill in job_text.lower():
            job_skills.append(skill)

    matched = list(set(resume_skills) & set(job_skills))
    missing = list(set(job_skills) - set(resume_skills))

    score = 0
    if len(job_skills) > 0:
        score = int((len(matched) / len(job_skills)) * 100)

    return score, matched, missing

@app.route("/ats")
def ats():
    try:
        with open("data/resume.txt") as f:
            resume = f.read()
        with open("data/job.txt") as f:
            job = f.read()
        score, matched, missing = ats_score(resume, job)
        return jsonify({
            "ATS Score": score,
            "Matching Skills": matched,
            "Missing Skills": missing
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/remove_resume", methods=["POST"])
def remove_resume():
    try:
        # Clear uploads folder
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Clear extracted text
        if os.path.exists("data/resume.txt"):
            os.remove("data/resume.txt")
            
        # Optional: Clear Endee index
        try:
            if db:
                db.delete_index("resumes")
                db.create_index("resumes", dimension=384, space_type="cosine")
        except:
            pass
            
        return jsonify({"message": "Resume and all associated files removed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_candidate_name():
    try:
        if os.path.exists("data/resume.txt"):
            with open("data/resume.txt", "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                # Clean up punctuation/icons if present
                for char in ["📞", "|", "✉", "🔗"]:
                    first_line = first_line.split(char)[0].strip()
                return first_line if first_line else "the Candidate"
    except:
        pass
    return "the Candidate"
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = str(data.get("question", ""))
    if not question:
        return jsonify({"error": "No question provided"}), 400

    if embed_model is None or tokenizer is None or model is None:
        return jsonify({"answer": "AI models are still loading. Please wait."})

    # 1. Quick Responses
    q_low = question.lower().strip().replace("?", "")
    candidate_name = get_candidate_name()

    if q_low in ["hi", "hii", "hello", "hey"]:
        return jsonify({"answer": "Hello! I am an AI Resume Analyzer. I can help analyze resumes and answer questions about the candidate."})
    if "how are you" in q_low:
        return jsonify({"answer": "I'm doing well! I'm ready to help analyze the resume and answer your questions."})
    if "who are you" in q_low:
        return jsonify({"answer": "I am an AI-powered Resume Analyzer designed to help understand candidate profiles."})
    if "technical skills" in q_low or "list skills" in q_low:
        try:
            with open("data/resume.txt", "r", encoding="utf-8") as f:
                skills = extract_skills(f.read())
            if skills:
                return jsonify({"answer": f"The candidate's technical skills are: {', '.join(skills)}."})
            return jsonify({"answer": "I couldn't find a clear list of technical skills in the resume."})
        except:
            return jsonify({"answer": "No resume data available to extract skills."})

    # 2. Precision Context Retrieval
    context = ""
    is_ats_query = any(word in q_low for word in ["ats", "match", "suitable", "fit", "score"])
    
    try:
        if embed_model is not None and db is not None:
            q_vector = embed_model.encode(question).tolist()
            index = db.get_index("resumes")
            if index is not None:
                results = index.query(vector=q_vector, top_k=3) # Reduced top_k for precision
                context = "\n".join([str(r["meta"]["text"]) for r in results if isinstance(r, dict) and "meta" in r])
    except Exception:
        # Smart Fallback: Keyword search in file
        try:
            with open("data/resume.txt", "r", encoding="utf-8") as f:
                resume_text = f.read()
                # Find relevant paragraphs based on keywords
                keywords = [w for w in q_low.split() if len(w) > 3]
                relevant_lines = []
                lines = resume_text.split("\n")
                for i, line in enumerate(lines):
                    if any(k in line.lower() for k in keywords):
                        # Capture surrounding context
                        s_idx = max(0, i-2)
                        e_idx = min(len(lines), i+3)
                        context_chunk = "\n".join(lines[s_idx:e_idx]) # type: ignore
                        relevant_lines.append(str(context_chunk))
                        if len("\n".join(relevant_lines)) > 1500: break
                
                context = "\n---\n".join(relevant_lines) if relevant_lines else resume_text[:1500] # type: ignore
        except:
            context = "No resume data available."

    # Add Job Context if needed
    if is_ats_query:
        try:
            with open("data/job.txt", "r", encoding="utf-8") as f:
                context += f"\n\nTARGET JOB:\n{f.read()}"
        except: pass

    # 3. Precision Prompt for FLAN-T5-Small
    # Simplified to "Question" and "Context" which small models handle best
    prompt = f"Question: {question} Context: {context}"
    
    try:
        # Check for off-topic questions (very basic filter)
        off_topic_keywords = ["weather", "joke", "news", "movie", "song", "food"]
        if any(word in q_low for word in off_topic_keywords):
            return jsonify({"answer": "I apologize, please ask about resume only."})

        inputs = tokenizer(prompt, return_tensors="pt") # type: ignore
        outputs = model.generate(inputs["input_ids"], max_length=150, do_sample=False) # type: ignore
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip() # type: ignore
        
        # Suggestions Bank
        suggestions = [
            "What are the candidate's technical skills?",
            "What projects has the candidate worked on?",
            "How many years of experience does the candidate have?",
            "Which university did the candidate graduate from?"
        ]

        # Professional fallback with specific refusal message if answer is garbage or too short
        if not answer or len(answer) < 5 or "Context:" in answer or "Question:" in answer:
            # If it's a social greeting we already handled it, if it's random text:
            top_suggestions = suggestions[:3] # type: ignore
            return jsonify({
                "answer": "I'm sorry, I couldn't find that in the resume. Please ask about resume only, or try one of these:",
                "suggestions": top_suggestions
            })
            
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/score_resume", methods=["POST"])
def score():
    data = request.get_json() or {}
    job_desc = str(data.get("job_description", ""))
    try:
        with open("data/resume.txt", "r", encoding="utf-8") as f:
            resume_text = f.read()
    except:
        return jsonify({"error": "No resume found."}), 404

    prompt = f"Job:\n{job_desc}\n\nResume:\n{resume_text[:1000]}\n\nCompare and provide a Match Score (0-100%):" # type: ignore
    
    if tokenizer is None or model is None:
        return jsonify({"error": "AI models are not loaded."}), 500

    try:
        inputs = tokenizer(prompt, return_tensors="pt") # type: ignore
        outputs = model.generate(**inputs, max_length=200) # type: ignore
        analysis_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip() # type: ignore
        return jsonify({"analysis": "Match Score: " + analysis_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
