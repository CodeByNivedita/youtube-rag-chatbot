import os

os.environ["OPENAI_API_KEY"] = "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"
import streamlit as st
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# ---------------- STREAMLIT CONFIG ----------------

st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube Chatbot & Video Summarizer")


# ---------------- CREATE RAG CHAIN ----------------

def create_chain(video_id):

    ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(
        video_id,
        languages=["en", "hi", "es"]
    )

    text = " ".join(
        snippet.text
        for snippet in transcript
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient,
just say you don't know.

{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = (
        parallel_chain
        | prompt
        | llm
        | parser
    )

    return main_chain


# ---------------- SESSION STATE ----------------

if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- VIDEO INPUT ----------------

video_url = st.text_input(
    "Paste YouTube URL"
)

if st.button("Process Video"):

    try:

        video_id = parse_qs(
            urlparse(video_url).query
        )["v"][0]

        with st.spinner(
            "Processing Video..."
        ):

            st.session_state.chain = create_chain(
                video_id
            )

        st.success(
            "Video processed successfully!"
        )

    except TranscriptsDisabled:

        st.error(
            "Transcripts are disabled for this video."
        )

    except NoTranscriptFound:

        st.error(
            "No transcript available in supported languages."
        )

    except Exception as e:

        st.error(str(e))


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Ask anything about the video..."
)

if (
    question
    and
    st.session_state.chain is not None
):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    answer = st.session_state.chain.invoke(
        question
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()

