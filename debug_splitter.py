
import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
from llama_index.core.embeddings import BaseEmbedding
from typing import List

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str = "models/gemini-embedding-001"):
        super().__init__()
        self._model = model

    def _get_text_embedding(self, text: str) -> List[float]:
        print(f"DEBUG: Embedding text (len={len(text)})")
        # result = genai.embed_content(model=self._model, content=text)
        # return result["embedding"]
        return [0.1] * 768 # Dummy for speed

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

Settings.embed_model = GeminiEmbedding()

async def debug():
    print("🚀 Debugging Splitter...")
    
    # Test document (10KB)
    text = "Esta es una oración. " * 100
    doc = Document(text=text)
    
    print(f"📄 Doc length: {len(text)} chars")
    
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model,
        show_progress=True
    )
    
    print("✂️ Splitting into sentences...")
    # This is what happens inside get_nodes_from_documents
    from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
    sentences = split_by_sentence_tokenizer()(text)
    print(f"✅ Sentences count: {len(sentences)}")
    
    print("🧠 Starting get_nodes_from_documents...")
    nodes = splitter.get_nodes_from_documents([doc])
    print(f"✅ Success! Generated {len(nodes)} nodes.")

if __name__ == "__main__":
    asyncio.run(debug())
