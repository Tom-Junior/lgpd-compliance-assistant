"""RAG pipeline — chunk, embed, index, retrieve, generate.

Configuração: GROQ para LLM + sentence-transformers para embeddings (100% gratuito).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ============================================================================
# Embedding Function customizada usando sentence-transformers (100% local)
# ============================================================================
class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function local usando sentence-transformers.
    
    Usa modelo multilíngue que funciona bem para português.
    Roda 100% no servidor, sem API externa, sem custo.
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self) -> SentenceTransformer:
        """Lazy load do modelo (só carrega quando necessário)."""
        if self._model is None:
            print(f"📥 Carregando modelo de embeddings: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print(f"✅ Modelo carregado com sucesso!")
        return self._model
    
    def __call__(self, input: Documents) -> Embeddings:
        """Gera embeddings para uma lista de textos."""
        model = self._load_model()
        embeddings = model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normaliza para cosine similarity
            show_progress_bar=False,
        )
        return embeddings.tolist()


# ============================================================================
# Cliente GROQ para LLM
# ============================================================================
def _make_llm_client() -> OpenAI:
    """Inicializa cliente GROQ para LLM."""
    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError(
            "GROQ_API_KEY não configurada. "
            "Obtenha em https://console.groq.com/keys e adicione no Streamlit Secrets."
        )
    
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    print("🚀 GROQ configurado para LLM (llama-3.3-70b-versatile)")
    return client


def _clean_text(text: str) -> str:
    """Remove caracteres inválidos, null bytes e normaliza espaços."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# Pipeline RAG
# ============================================================================
class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local + embeddings locais."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        # Cliente GROQ para geração
        self.llm_client = _make_llm_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
        
        # Embedding model local
        self.embed_model_name = embed_model or os.environ.get(
            "EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.embed_fn = LocalEmbeddingFunction(model_name=self.embed_model_name)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )

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

        # TODO 1.C — Indexação no Chroma (embeddings gerados localmente)
        if chunks:
            ids = [c["id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
            
            print("🔢 Gerando embeddings localmente (pode demorar 1-2 min na primeira vez)...")
            start_time = time.time()
            
            # Chroma chama automaticamente self.embed_fn internamente
            BATCH_SIZE = 100
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_ids = ids[i:i + BATCH_SIZE]
                batch_docs = documents[i:i + BATCH_SIZE]
                batch_metas = metadatas[i:i + BATCH_SIZE]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                )
                print(f"  📦 Batch {(i // BATCH_SIZE) + 1}: {len(batch_ids)} chunks indexados")
            
            elapsed = time.time() - start_time
            print(f"💾 Total indexados: {len(chunks)} chunks em {elapsed:.1f}s")

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

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente técnico especialista em LGPD. Responda APENAS com base no contexto fornecido. Cite os artigos da lei quando relevante.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content if response.choices else "Erro ao gerar resposta."
        return {"answer": answer, "sources": sources}


PROMPT_TEMPLATE = """Voce e um assistente tecnico especialista em LGPD. Responda APENAS com base no contexto abaixo.
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