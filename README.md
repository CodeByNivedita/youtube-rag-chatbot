
# 🎥 YouTube RAG Chatbot & Video Summarizer

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to interact with YouTube videos using natural language. Simply paste a YouTube video URL, and the application extracts the transcript, builds a vector database, and enables intelligent question-answering directly from the video content.

## 🚀 Features

* Extracts transcripts from YouTube videos
* Supports multiple transcript languages (English, Hindi, Spanish)
* Creates embeddings using OpenAI Embeddings
* Stores embeddings in a FAISS vector database
* Retrieval-Augmented Generation (RAG) pipeline using LangChain
* Interactive chat interface built with Streamlit
* Answers questions based only on the video content
* Handles missing or disabled transcripts gracefully

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* OpenAI GPT-4o-mini
* OpenAI Embeddings
* FAISS
* YouTube Transcript API

---

## 📂 Project Structure

```text
youtube-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
|─ b.ipynb
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser automatically.

---

## 📖 How It Works

1. User pastes a YouTube URL.
2. Transcript is fetched using YouTube Transcript API.
3. Transcript is split into chunks.
4. Chunks are converted into embeddings.
5. Embeddings are stored in FAISS.
6. Relevant chunks are retrieved based on the user's query.
7. GPT-4o-mini generates answers using the retrieved context.

---

## 💬 Example Questions

* What is this video about?
* Summarize the video.
* What are the key points discussed?
* Explain the concept of RAG mentioned in the video.
* What tools are used in this tutorial?

---

## 📸 Demo

Add screenshots of your Streamlit application here.

---

## 🔮 Future Improvements

* Support YouTube Shorts
* Automatic language detection
* Whisper-based transcription for videos without captions
* Conversation memory
* PDF export of summaries
* Multi-video chat support

---

## 👩‍💻 Author

Nivedita Kasaudhan

Built as a learning project to explore:

* Generative AI
* Retrieval-Augmented Generation (RAG)
* LangChain
* Vector Databases
* LLM-powered Applications
