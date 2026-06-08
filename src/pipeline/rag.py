"""RAG pipeline — chunk, embed, index, retrieve, generate.

Configuração: GROQ para LLM + OpenAI para embeddings.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _make_client() -> tuple[OpenAI, OpenAI, str | None]:
    """
    Retorna (llm_client, embed_client, embed_api_base).
    
    - GROQ para LLM (llama-3.3-70b-versatile)
    - OpenAI para embeddings (text-embedding-3-small)
    """
    # Cliente GROQ para LLM
    if "GROQ_API_KEY" in os.environ:
        llm_client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        print("🚀 GROQ configurado para LLM")
    else:
        raise RuntimeError("GROQ_API_KEY não configurada. Adicione no .env ou Streamlit Secrets.")
    
    # Cliente OpenAI para embeddings
    if "OPENAI_API_KEY" in os.environ:
        embed_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        embed_api_base = None
        print("🤖 OpenAI configurado para embeddings")
    elif "GEMINI_API_KEY" in os.environ:
        embed_client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        print("💎 Gemini configurado para embeddings (fallback)")
    else:
        raise RuntimeError("OPENAI_API_KEY ou GEMINI_API_KEY necessária para embeddings.")
    
    return llm_client, embed_client, embed_api_base


def _clean_text(text: str) -> str:
    """Remove caracteres inválidos, null bytes e normaliza espaços."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.llm_client, self.embed_client, embed_api_base = _make_client()
        
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")

        # Configura função de embedding (OpenAI ou Gemini)
        embed_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"),
            "model_name": self.embed_model,
        }
        if embed_api_base:
            embed_kwargs["api_base"] = embed_api_base
        
        self.embed_fn = OpenAIEmbeddingFunction(**embed_kwargs)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def _embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """Embedda textos em lotes seguros para evitar BadRequestError."""
        embeddings: list[list[float]] = []
        batch_size = 50  # Seguro para OpenAI/Gemini
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.embed_client.embeddings.create(
                model=self.embed_model,
                input=batch,
            )
            embeddings.extend([d.embedding for d in response.data])
            if i + batch_size < len(texts):
                time.sleep(1.0)  # Respeita rate limit
        return embeddings

    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking e indexa em Chroma."""
        # TODO 1.A — Ingestão de PDFs
        docs: list[dict] = []
        for pdf_path in sorted(self.corpus_dir.glob("*.pdf")):
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                cleaned = _clean_text(text) if text else ""
                if cleaned and len(cleaned) > 50:
                    docs.append({
                        "text": cleaned,
                        "source": pdf_path.name,
                        "page": page_num,
                    })
        print(f"📚 Ingeridos {len(docs)} documentos válidos de {len(list(self.corpus_dir.glob('*.pdf')))} PDFs.")

        # TODO 1.B — Chunking recursivo
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[dict] = []
        for doc in docs:
            splits = splitter.split_text(doc["text"])
            for chunk_idx, chunk_text in enumerate(splits):
                unique_id = f"{doc['source']}_p{doc['page']}_c{chunk_idx}"
                chunks.append({
                    "id": unique_id,
                    "text": _clean_text(chunk_text),
                    "source": doc["source"],
                    "page": doc["page"],
                })
        print(f"✂️  Gerados {len(chunks)} chunks (size=800, overlap=100).")

        # TODO 1.C — Embedding manual + indexação no Chroma
        if chunks:
            ids = [c["id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
            
            print("🔢 Gerando embeddings em lotes seguros...")
            embeddings = self._embed_texts_batch(documents)
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            print(f"💾 Total indexados: {len(chunks)} chunks na collection '{self.collection_name}'.")

        return self.collection.count()

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        
        hits: list[dict] = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results.get("distances") else None
                hits.append({
                    "text": doc_text,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", 0),
                    "distance": distance,
                })
        
        return hits

    def answer(self, question: str, k: int = 5) -> dict:
        """Pipeline completo: retrieve + augment + generate."""
        hits = self.retrieve(question, k=k)

        context_parts = []
        sources = []
        for hit in hits:
            header = f"[{hit['source']}:{hit['page']}]"
            context_parts.append(f"{header}\n{hit['text']}")
            sources.append((hit["source"], hit["page"]))
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "Nenhum contexto encontrado."

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # Usa GROQ para geração
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "Você é um assistente técnico que responde APENAS com base no contexto fornecido."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content if response.choices else "Erro ao gerar resposta."
        return {"answer": answer, "sources": sources}


PROMPT_TEMPLATE = """Voce e um assistente tecnico. Responda APENAS com base no contexto abaixo.
Se a informacao nao estiver no contexto, diga "Nao encontrado no corpus".
Sempre cite a fonte usando o formato [arquivo:pagina].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline