from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
from dotenv import load_dotenv
import sys
import openai
import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Backend.config.config import OPENAI_API_KEY
from Backend.data.images.description import description as image_descriptions

client = openai.OpenAI(api_key=OPENAI_API_KEY)


# 1. Load raw text from a file with error handling
def load_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"❌ Error: File not found -> {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error while reading file: {e}")
        sys.exit(1)


# 2. Split the text into chunks using "***" and auto-tag them using KeyBERT
def split_and_auto_tag_chunks(text, source_filename):
    raw_chunks = text.split("***")
    raw_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    tagged_chunks = []

    for chunk in raw_chunks:

        tagged_chunks.append(
            {
                "text": chunk,
                "source": source_filename,
                "type": "text",
            }
        )

    return tagged_chunks


# # 3. Embed chunks using sentence-transformers
# def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
#     try:
#         model = SentenceTransformer(model_name)
#         embeddings = model.encode([c["text"] for c in chunks]).tolist()
#         return embeddings, model
#     except Exception as e:
#         print(f"❌ Error during embedding: {e}")
#         sys.exit(1)


def embed_chunks(chunks, model_name="all-MiniLM-L6-v2", backend="onnx", batch_size=16):
    """
    Embed a list of {"text": str} chunks using an ONNX-optimized SentenceTransformer.

    Args:
        chunks (List[Dict]): each dict must have a "text" key.
        model_name (str): huggingface model ID (e.g. "all-MiniLM-L6-v2").
        backend (str): "onnx" (or "onnx-gpu") to enable ONNXRuntime.
        batch_size (int): how many sentences to encode per forward pass.

    Returns:
        embeddings (List[List[float]]): one vector per chunk.
        model (SentenceTransformer): the loaded model instance.
    """
    try:
        # 1) Load (and auto-export) the ONNX model
        model = SentenceTransformer(
            model_name, backend=backend, model_kwargs={"file_name": "model.onnx"}
        )

        # 2) Extract texts
        texts = [c["text"] for c in chunks]

        # 3) Encode in one shot with built-in batching + progress bar
        embs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        # 4) Convert to native Python lists
        return embs.tolist(), model

    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        sys.exit(1)


# 4. Store chunks and embeddings in ChromaDB
def store_in_chroma(
    tagged_chunks,
    embeddings,
    persist_directory,
    collection_name,
):
    try:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        collection = chroma_client.get_or_create_collection(name=collection_name)
    except Exception as e:
        print(f"❌ Error initializing ChromaDB: {e}")
        sys.exit(1)

    metadatas = []
    documents = []
    ids = []

    for i, chunk in enumerate(tagged_chunks):

        # Base metadata
        metadata = {
            "source": chunk["source"],
            "type": chunk["type"],
        }

        # Include image filename if it's an image chunk
        if chunk["type"] == "image":
            metadata["image_filename"] = chunk.get("image_filename", "")

        metadatas.append(metadata)
        documents.append(chunk["text"])
        ids.append(str(i))

    try:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        print(
            f"✅ Stored {len(tagged_chunks)} chunks in ChromaDB collection '{collection_name}' with metadata."
        )
    except Exception as e:
        print(f"❌ Error while storing data in ChromaDB: {e}")
        sys.exit(1)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# def get_image_description(filepath, filename):
#     # Encode the image and use vision model to get a description.
#     base64_image = encode_image(filepath)
#     filetype = filename.split(".")[1]
#     url = f"data:image/{filetype};base64,{base64_image}"

#     # Direct chat completion implementation
#     user_prompt = [
#         {
#             "type": "text",
#             "text": (
#                 "This image is a technical diagram from an industrial controller manual. "
#                 "Please analyze the image and describe it in full detail. "
#                 "If it shows a circuit, block diagram, wiring layout, or flow chart, explain what components are involved, "
#                 "how they are connected, and what process or function the diagram is illustrating. "
#                 "Also extract structured metadata like: "
#                 "1. Diagram type (e.g., block diagram, wiring diagram, circuit). "
#                 "2. Main components or devices shown. "
#                 "3. Primary function or system being described. "
#                 "4. Any visible labels, titles, or annotations. "
#                 "Give me as a proper DESCRIPTION"
#             ),
#         },
#         {"type": "image_url", "image_url": {"url": url}},
#     ]

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant. Answer the user's questions based on the given image(s). Just state the answer; it need not be a complete sentence necessarily. If the question can't be answered from the given image(s), simply respond with 'NO ANSWER' and no other words. DO NOT make up answers.",
#             },
#             {"role": "user", "content": user_prompt},
#         ],
#         temperature=0,
#         top_p=0,
#         response_format={"type": "text"},
#     )

#     image_description = response.choices[0].message.content
#     return image_description


def process_images_for_source(source_filename):
    image_folder = os.path.join(
        os.path.dirname(__file__), "..", "data", "images", source_filename
    )
    visual_chunks = []

    if not os.path.isdir(image_folder):
        return visual_chunks

    SUPPORTED_IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
    for img_file in sorted(os.listdir(image_folder)):
        if any(img_file.lower().endswith(ext) for ext in SUPPORTED_IMG_EXTS):
            img_path = os.path.join(image_folder, img_file)
            print(f"🖼️  Processing image: {img_path}")
            try:
                # For openai desc
                # desc = get_image_description(img_path, img_file)
                desc = get_image_description(img_path, img_file, source_filename)
                if not desc:
                    continue
               

                visual_chunks.append(
                    {
                        "text": desc.strip(),
                        "source": source_filename,
                        "type": "image",
                        "image_filename": img_path,
                    }
                )
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

    return visual_chunks


def get_image_description(filepath, filename, source_filename):
    try:
        return image_descriptions[source_filename][filename]
    except KeyError:
        print(f"⚠️ No description found for {filename} in {source_filename}. Skipping.")
        return None


# 5. Main: process multiple text files
if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), "..", "data", "text_files")
    persist_directory = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    collection_name = "manual_docs"

    all_tagged_chunks = []

    print("📄 Scanning folder for text files...")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            print(f"📁 Processing: {filename}")

            full_text = load_text_file(filepath)
            source_filename = Path(filepath).stem

            # 🧾 Text chunks
            tagged_chunks = split_and_auto_tag_chunks(full_text, source_filename)

            # 🖼️ Image chunks
            image_chunks = process_images_for_source(source_filename)
            # print(f"Image Chunks : {image_chunks}")
            all_tagged_chunks.extend(tagged_chunks)
            all_tagged_chunks.extend(image_chunks)

    if not all_tagged_chunks:
        print("⚠️ No valid text or image chunks found.")
        exit(0)

    print("🧠 Generating embeddings...")
    embeddings, model = embed_chunks(all_tagged_chunks)

    print("💾 Storing in ChromaDB with metadata...")
    store_in_chroma(all_tagged_chunks, embeddings, persist_directory, collection_name)
    print("🎉 Done processing all text and image data.")
