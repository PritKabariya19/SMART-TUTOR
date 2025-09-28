from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import os, json, traceback, random, uuid
from langchain_core.prompts import ChatPromptTemplate
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

# ----------------- Flask App -----------------
app = Flask(__name__)
CORS(app)

# ----------------- Load Environment Variables -----------------
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("❌ Missing API keys! Please set them in .env")

# ----------------- Gemini Setup -----------------
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- Embeddings + Pinecone -----------------
embeddings = download_hugging_face_embeddings()
INDEX_NAME = "std8science"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------- AI JSON Rules -----------------
AI_JSON_RULES = """
Output must be strict JSON like:

{{
  "question": "Question here",
  "options": ["Option1", "Option2", "Option3", "Option4"],
  "answer": 0
}}

Rules:
- Only output JSON.
- Exactly 4 options.
- Answer index must match correct option.
"""

SUBJECT_PROMPTS = {
    "math": "Generate one fun and simple math MCQ from the following content:",
    "english": "Generate one English grammar/vocab/comprehension MCQ from the following content:",
    "science": "Generate one science MCQ (physics/chemistry/biology) from the following content:"
}

# ----------------- Leaderboard -----------------
leaderboard_data = {"You": 0}

# ----------------- Helper Functions -----------------
def generate_quiz(subject: str):
    try:
        # Step 1: Retrieve relevant content from Pinecone
        docs = retriever.get_relevant_documents(subject)
        if not docs:
            return {"error": f"No content found in database for subject '{subject}'"}

        # Combine retrieved content for context
        context_text = "\n".join([d.metadata.get("text", "") for d in docs])

        # Step 2: Generate quiz from Gemini using the content
        prompt = f"{SUBJECT_PROMPTS[subject]}\n{context_text}\n{AI_JSON_RULES}"
        response = gemini_model.generate_content(prompt)
        quiz_text = response.text.strip()

        # Cleanup if Gemini returns code fences
        if quiz_text.startswith("```"):
            quiz_text = "\n".join(quiz_text.splitlines()[1:-1]).strip()

        quiz_json = json.loads(quiz_text)

        # Shuffle options while keeping correct answer
        options = quiz_json["options"]
        correct_answer = options[quiz_json["answer"]]
        random.shuffle(options)
        quiz_json["options"] = options
        quiz_json["answer"] = options.index(correct_answer)

        # Step 3: Optionally store the quiz in Pinecone for future retrieval
        quiz_id = str(uuid.uuid4())
        text_to_embed = f"{subject} | {quiz_json['question']} | {quiz_json['options']}"
        vector = embeddings.embed_query(text_to_embed)
        docsearch.index.upsert([(quiz_id, vector, {"subject": subject, **quiz_json})])

        # Add ID to response
        quiz_json["id"] = quiz_id
        return quiz_json

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def home():
    return "Quiz API is running! Use /quiz/math, /quiz/english, or /quiz/science"

@app.route("/quiz/<subject>", methods=["GET"])
def get_quiz(subject):
    if subject not in SUBJECT_PROMPTS:
        return jsonify({"error": f"Subject '{subject}' not supported"}), 400
    return jsonify(generate_quiz(subject))

@app.route("/answer", methods=["POST"])
def answer_quiz():
    data = request.get_json()
    quiz_id = data.get("quizId")
    choice = data.get("choice")
    user = data.get("user", "You")

    # Fetch quiz from Pinecone
    result = docsearch.index.fetch([quiz_id])
    if quiz_id not in result["vectors"]:
        return jsonify({"error": "Quiz not found"}), 404
    quiz_meta = result["vectors"][quiz_id]["metadata"]

    correct = choice == quiz_meta["answer"]
    leaderboard_data[user] = leaderboard_data.get(user, 0) + (5 if correct else 0)
    return jsonify({"correct": correct, "stars": leaderboard_data[user]})

@app.route("/leaderboard", methods=["GET"])
def leaderboard():
    lb = [{"name": k, "stars": v} for k, v in leaderboard_data.items()]
    lb = sorted(lb, key=lambda x: x["stars"], reverse=True)
    return jsonify(lb)

# ----------------- Main -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
