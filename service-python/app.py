from flask import Flask, request, jsonify
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import os
import google.generativeai as genai

# ----------------- Flask App -----------------
app = Flask(__name__)
CORS(app) #cross-origin requests (frontend like React can call backend).

# ----------------- Load Environment Variables -----------------
dotenv_path = Path(__file__).resolve().parent / ".env" #to find the env
load_dotenv(dotenv_path=dotenv_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("❌ Missing API keys! Please set them in .env")

# ----------------- Gemini Setup -----------------
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash") #curr using gemini-1.5-flash version.

# ----------------- Embeddings + Pinecone -----------------
embeddings = download_hugging_face_embeddings()
index_name = "std8science" # index name in pinecone

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------- Prompt Template -----------------
system_prompt = """
    You are a helpful AI bot.
"""
ai_prompt = """
    You are a **friendly tutor** for students in **classes 6 to 8**.  
Your role is to make learning **simple, fun, and encouraging**.  

### Answering Rules:
- **Keep answers short** → about 3–5 sentences.  
- **Highlight key terms in bold**.  
- **Use bullet points (-)** for steps, examples, or lists.  
- **Explain step by step** when needed, with simple examples.  
- **Avoid complex words**; use easy language kids can understand.  
- **Make learning fun** with a warm, positive tone.  
- **If you don’t know**, say “I don’t know” honestly.  

### Output Format:
- Start with a **clear answer**.  
- Add **bullets (-)** if listing steps or points.  
- Keep it **short, clear, and encouraging**.  

Goal: Help students understand quickly, enjoy learning, and feel confident.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("ai",ai_prompt),
])

# ----------------- RAG Function -----------------
def run_gemini_rag(query: str):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser: {query}\nAssistant:"
    response = gemini_model.generate_content(final_prompt)

    return response.text if response else "⚠️ No response from Gemini."


# ----------------- Dummy DB -----------------
leaderboard_data = {
    "You": 0
}

quiz_bank = {
    "Math": {
        "id": "quiz_math_1",
        "question": "What is 12 × 10?",
        "options": ["120", "86", "108", "112"],
        "answer": 0
    },
    "Science": {
        "id": "quiz_sci_1",
        "question": "Which planet is known as the Red Planet?",
        "options": ["Earth", "Mars", "Jupiter", "Venus"],
        "answer": 1
    }
}


# ----------------- Routes -----------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("text")
    subject = data.get("subject", "General")

    print("👤 User:", msg, "| Subject:", subject)
    response_text = run_gemini_rag(msg)

    return jsonify({"replies": [response_text]})


@app.route("/quiz/<subject>", methods=["GET"])
def quiz(subject):
    q = quiz_bank.get(subject, None)
    if not q:
        return jsonify({"error": "No quiz found"}), 404
    return jsonify({
        "id": q["id"],
        "question": q["question"],
        "options": q["options"]
    })


@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json()
    quiz_id = data.get("quizId")
    choice = data.get("choice")
    user = data.get("user", "You")

    correct = False
    for q in quiz_bank.values():
        if q["id"] == quiz_id:
            correct = (choice == q["answer"])

    if correct:
        leaderboard_data[user] = leaderboard_data.get(user, 0) + 5

    return jsonify({"correct": correct, "stars": leaderboard_data[user]})


@app.route("/leaderboard", methods=["GET"])
def leaderboard():
    lb = [{"name": k, "stars": v} for k, v in leaderboard_data.items()]
    lb = sorted(lb, key=lambda x: x["stars"], reverse=True)
    return jsonify(lb)


# ----------------- Main -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
