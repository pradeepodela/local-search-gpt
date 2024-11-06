import streamlit as st
import torch
import clip
from PIL import Image
import glob
import os
import numpy as np
import torch.nn.functional as F
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.converters import PyPDFToDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_store" not in st.session_state:
    st.session_state.document_store = InMemoryDocumentStore()
    st.session_state.pipeline_initialized = False

# CLIP Model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "./new_data"

@st.cache_resource
def load_clip_model():
    return clip.load("ViT-L/14", device=device)

model, preprocess = load_clip_model()

@st.cache_data
def load_images():
    images = []
    if os.path.exists(IMAGE_DIR):
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('png', 'jpg', 'jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(IMAGE_DIR, image_file)
            image = Image.open(image_path).convert("RGB")
            images.append((image_file, image))
    return images

@st.cache_data
def encode_images(images):
    image_features = []
    for image_file, image in images:
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = F.normalize(image_feature, dim=-1)
        image_features.append((image_file, image_feature))
    return image_features

def search_images_by_text(text_query, top_k=5):
    text_inputs = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1)

    similarities = []
    for image_file, image_feature in image_features:
        similarity = torch.cosine_similarity(text_features, image_feature).item()
        similarities.append((image_file, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def search_images_by_image(query_image, top_k=5):
    query_image = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_image_feature = model.encode_image(query_image)
        query_image_feature = F.normalize(query_image_feature, dim=-1)

    similarities = []
    for image_file, image_feature in image_features:
        similarity = torch.cosine_similarity(query_image_feature, image_feature).item()
        similarities.append((image_file, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Custom CSS
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            color: #FF4B4B;
            font-weight: bold;
            text-align: center;
        }
        .subtitle {
            font-size: 24px;
            color: #FF914D;
            font-weight: bold;
            margin-top: 30px;
        }
        .result-container {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 10px;
        }
        .score-badge {
            color: white;
            background-color: #007BFF;
            padding: 5px;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Main App
st.markdown('<h1 class="title">Multi-Model Search & QA System</h1>', unsafe_allow_html=True)

# Sidebar for app selection and setup
with st.sidebar:
    st.header("Application Settings")
    app_mode = st.radio("Select Application Mode:", ["Document Q&A", "Image Search"])
    
    if app_mode == "Document Q&A":
        st.header("Document Setup")
        uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
        
        if uploaded_file and not st.session_state.pipeline_initialized:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize components
            document_embedder = SentenceTransformersDocumentEmbedder(model="BAAI/bge-small-en-v1.5")
            
            # Create indexing pipeline
            indexing_pipeline = Pipeline()
            indexing_pipeline.add_component("converter", PyPDFToDocument())
            indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
            indexing_pipeline.add_component("embedder", document_embedder)
            indexing_pipeline.add_component("writer", DocumentWriter(st.session_state.document_store))
            
            indexing_pipeline.connect("converter", "splitter")
            indexing_pipeline.connect("splitter", "embedder")
            indexing_pipeline.connect("embedder", "writer")

            text_embedder2 = SentenceTransformersTextEmbedder(model="BAAI/bge-small-en-v1.5")
            embedding_retriever2 = InMemoryEmbeddingRetriever(st.session_state.document_store)
            bm25_retriever2 = InMemoryBM25Retriever(st.session_state.document_store)
            document_joiner2 = DocumentJoiner()
            ranker2 = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")
            
            with st.spinner("Processing document..."):
                try:
                    indexing_pipeline.run({"converter": {"sources": ["temp.pdf"]}})
                    st.success(f"Processed {st.session_state.document_store.count_documents()} document chunks")
                    st.session_state.pipeline_initialized = True
                    
                    # Initialize retrieval components
                    text_embedder = SentenceTransformersTextEmbedder(model="BAAI/bge-small-en-v1.5")
                    embedding_retriever = InMemoryEmbeddingRetriever(st.session_state.document_store)
                    bm25_retriever = InMemoryBM25Retriever(st.session_state.document_store)
                    document_joiner = DocumentJoiner()
                    ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")
                    
                    template = """
                    act as a senior customer care executive and help users sorting out their queries. Be polite and friendly. Answer the user's questions based on the below context only dont try to make up any answer make sure that create a good version of all the documents that u recived and make the answer complining to the question make user the you sound exactly same as the documents delow.:
                    CONTEXT:
                    {% for document in documents %}
                        {{ document.content }}
                    {% endfor %}
                    Make sure to provide all the details. If the answer is not in the provided context just say, 'answer is not available in the context'. Don't provide the wrong answer.
                    If the person asks any external recommendation just say 'sorry i can't help you with that'.

                    Question: {{question}}

                    explain in detail
                    """
                    
                    prompt_builder = PromptBuilder(template=template)
                    
                    if "GOOGLE_API_KEY" not in os.environ:
                        os.environ["GOOGLE_API_KEY"] = 'AIzaSyDNIiOX5-Z1YFxZcaHFIEQr0DcXNvRelqI'
                    generator = GoogleAIGeminiGenerator(model="gemini-pro")
                    
                    # Create retrieval pipeline
                    st.session_state.retrieval_pipeline = Pipeline()
                    st.session_state.retrieval_pipeline.add_component("text_embedder", text_embedder)
                    st.session_state.retrieval_pipeline.add_component("embedding_retriever", embedding_retriever)
                    st.session_state.retrieval_pipeline.add_component("bm25_retriever", bm25_retriever)
                    st.session_state.retrieval_pipeline.add_component("document_joiner", document_joiner)
                    st.session_state.retrieval_pipeline.add_component("ranker", ranker)
                    st.session_state.retrieval_pipeline.add_component("prompt_builder", prompt_builder)
                    st.session_state.retrieval_pipeline.add_component("llm", generator)
                    
                    # Connect pipeline components
                    st.session_state.retrieval_pipeline.connect("text_embedder", "embedding_retriever")
                    st.session_state.retrieval_pipeline.connect("bm25_retriever", "document_joiner")
                    st.session_state.retrieval_pipeline.connect("embedding_retriever", "document_joiner")
                    st.session_state.retrieval_pipeline.connect("document_joiner", "ranker")
                    st.session_state.retrieval_pipeline.connect("ranker", "prompt_builder.documents")
                    st.session_state.retrieval_pipeline.connect("prompt_builder", "llm")

                    # Ranker pipeline
                    st.session_state.hybrid_retrieval2 = Pipeline()
                    st.session_state.hybrid_retrieval2.add_component("text_embedder", text_embedder2)
                    st.session_state.hybrid_retrieval2.add_component("embedding_retriever", embedding_retriever2)
                    st.session_state.hybrid_retrieval2.add_component("bm25_retriever", bm25_retriever2)
                    st.session_state.hybrid_retrieval2.add_component("document_joiner", document_joiner2)
                    st.session_state.hybrid_retrieval2.add_component("ranker", ranker2)

                    st.session_state.hybrid_retrieval2.connect("text_embedder", "embedding_retriever")
                    st.session_state.hybrid_retrieval2.connect("bm25_retriever", "document_joiner")
                    st.session_state.hybrid_retrieval2.connect("embedding_retriever", "document_joiner")
                    st.session_state.hybrid_retrieval2.connect("document_joiner", "ranker")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")

# Main content area
if app_mode == "Document Q&A":
    st.markdown('<h2 class="subtitle">Document Q&A System</h2>', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if st.session_state.pipeline_initialized:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.retrieval_pipeline.run(
                            {
                                "text_embedder": {"text": prompt},
                                "bm25_retriever": {"query": prompt},
                                "ranker": {"query": prompt},
                                "prompt_builder": {"question": prompt}
                            }
                        )
                        result2 = st.session_state.hybrid_retrieval2.run(
                            {
                                "text_embedder": {"text": prompt},
                                "bm25_retriever": {"query": prompt},
                                "ranker": {"query": prompt}
                            }
                        )
                        l = []
                        for i in result2['ranker']['documents']:
                            if i.meta['file_path'] in l:
                                pass
                            else:
                                l.append(i.meta['file_path'])
                            l.append(i.meta['page_number'])
                        
                        response = result['llm']['replies'][0]
                        response = f"{response} \n\nsource: {l} "
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            with st.chat_message("assistant"):
                message = "Please upload a document first to start the conversation."
                st.warning(message)
                st.session_state.messages.append({"role": "assistant", "content": message})

else:  # Image Search mode
    st.markdown('<h2 class="subtitle">Image Search System</h2>', unsafe_allow_html=True)
    
    # Load and encode images
    images = load_images()
    image_features = encode_images(images)
    
    search_type = st.radio("Select Search Type:", ["Text-to-Image", "Image-to-Image"])

    if search_type == "Text-to-Image":
        query = st.text_input("Enter a text description to find similar images:")
        
        if query:
            results = search_images_by_text(query)
            st.write(f"Top results for query: **{query}**")
            
            cols = st.columns(3)
            for idx, (image_file, score) in enumerate(results):
                with cols[idx % 3]:
                    st.markdown(f'<div class="result-container">', unsafe_allow_html=True)
                    image_path = os.path.join(IMAGE_DIR, image_file)
                    image = Image.open(image_path)
                    st.image(image, caption=image_file)
                    st.markdown(f'<span class="score-badge">Score: {score:.4f}</span>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    else:  # Image-to-Image search
        uploaded_image = st.file_uploader("Upload an image to find similar images:", type=["png", "jpg", "jpeg"])
        
        if uploaded_image is not None:
            query_image = Image.open(uploaded_image).convert("RGB")
            st.image(query_image, caption="Query Image", use_column_width=True)

            # Search and display results
            results = search_images_by_image(query_image)
            st.write("Top results for the uploaded image:")
            
            cols = st.columns(3)
            for idx, (image_file, score) in enumerate(results):
                with cols[idx % 3]:
                    st.markdown(f'<div class="result-container">', unsafe_allow_html=True)
                    image_path = os.path.join(IMAGE_DIR, image_file)
                    image = Image.open(image_path)
                    st.image(image, caption=image_file)
                    st.markdown(f'<span class="score-badge">Score: {score:.4f}</span>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    # Create the image directory if it doesn't exist
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
