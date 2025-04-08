from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import os
import PyPDF2

# Load models and tokenizer
print("Loading models...")
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline('text2text-generation', model='google/flan-t5-base')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

# Read and split PDF into paragraphs
doc_folder = "../docs"
docs = []
print(f"Reading PDFs from {doc_folder}...")
for file in os.listdir(doc_folder):
    file_path = os.path.join(doc_folder, file)
    if file.endswith(".pdf"):
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text() or ""
                paragraphs = text.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        docs.append(para)

if not docs:
    print("Error: No documents found in the docs folder!")
    exit()

print(f"Loaded {len(docs)} paragraphs from PDFs.")

# Encode documents
print("Encoding documents...")
doc_embeddings = retrieval_model.encode(docs)

# Function to get an answer
def get_answer(question):
    question_embedding = retrieval_model.encode(question)
    k = 5
    similarities = [retrieval_model.similarity(question_embedding, doc_emb).item() for doc_emb in doc_embeddings]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
    top_paragraphs = [docs[idx] for idx in top_indices]
    context = " ".join(top_paragraphs)
    
    # Truncate context to fit within token limit (leaving room for question and prompt)
    max_context_tokens = 400  # Reserve 112 tokens for question + prompt
    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
    if len(tokenized_context) > max_context_tokens:
        context = tokenizer.decode(tokenized_context[:max_context_tokens], skip_special_tokens=True)
    
    prompt = f"Question: {question}\nContext: {context}\nAnswer in a clear, detailed sentence:"
    answer = generator(prompt, max_length=150, num_return_sequences=1)
    return answer[0]['generated_text']

# Interactive loop
if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = get_answer(question)
        print("Answer:", answer)