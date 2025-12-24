import os
import glob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Load documents
os.makedirs("md-data", exist_ok=True)
uploaded_files = glob.glob("md-data/*.txt")

all_docs = []
for file in uploaded_files:
    loader = TextLoader(file, encoding="utf-8")
    docs = loader.load()
    domain_type = "frontend" if "front" in file.lower() else "backend"
    for doc in docs:
        all_docs.append(Document(page_content=doc.page_content, metadata={"domain": domain_type, "source": file}))

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(all_docs)

# Embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings,
                                 collection_name="projects_collection",
                                 persist_directory="./chroma_projects")
vectordb.persist()
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# System prompt and template
system_prompt = """You are an AI Project Generator.
Generate a full educational project for mentors. Follow the structure and tone of example projects.
Never mention RAG or retrieved documents. Produce a new project following the same style.
Make sure the first task is "Set Up Proper Folder Structure". Do not add Extra Challenges in tasks.
Output must be valid JSON only."""

project_template = """{
    "projects": [
        {
            "project_name": "string",
            "description": "string",
            "difficulty": "beginner/intermediate/advanced",
            "skills": ["string", "string"],
            "tools": ["string", "string"],
            "learning_outcomes": ["string", "string"],
            "tasks": [
                {
                    "task_title": "string",
                    "task_details": {
                        "objective": "string",
                        "why_this_is_important": "string",
                        "instructions": ["string", "string"],
                        "file_path": "string",
                        "expected_output": "string",
                        "skills_practiced": ["string", "string"]
                    }
                }
            ]
        }
    ]
}"""

prompt_template = PromptTemplate(
    template="""{system_prompt}

## Domain: {domain}
## Level: {level}
## Mentor Request: {description}

## Example Projects (from retrieval):
{examples}

## Required JSON Format:
{json_format}

## Format Instructions:
{format_instructions}

Generate a completely new project following the structure above. Output ONLY JSON, no other text.""",
    input_variables=["system_prompt", "domain", "level", "description", "examples", "json_format", "format_instructions"]
)

# ===== Helper functions =====
def get_domain_specific_documents(domain: str, query: str, k: int = 3):
    retrieved_docs = retriever.get_relevant_documents(query)
    domain_docs = [doc for doc in retrieved_docs if doc.metadata.get("domain", "") == domain]
    if len(domain_docs) >= k:
        return domain_docs[:k]
    general_docs = [doc for doc in retrieved_docs if doc not in domain_docs]
    return domain_docs + general_docs[:k - len(domain_docs)]
def generate_project(domain: str, level: str, description: str):
    retrieved_docs = get_domain_specific_documents(domain=domain.lower(), query=f"{domain} {level} {description}", k=3)
    retrieved_texts_str = "\n\n--- Retrieved Example Project ---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()
    
    full_prompt = prompt_template.format(
        system_prompt=system_prompt,
        domain=domain,
        level=level,
        description=description,
        examples=retrieved_texts_str,
        json_format=project_template,
        format_instructions=format_instructions
    )
    
    chain = llm | parser
    response = chain.invoke(full_prompt)
    return response