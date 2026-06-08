"""RAG pipeline — chunk, embed, index, retrieve, generate.

Reaproveita as funcoes do notebook 02. Voce vai preencher 3 TODOs aqui.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _make_client() -> tuple[OpenAI, str]:
    """Inicializa cliente OpenAI-compatible conforme provider escolhido no .env."""
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
        embed_api_base = None
    else:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
    return client, embed_api_base


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
        self.client, embed_api_base = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")

        embed_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
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

    # Pipeline RAG: chunking + embedding + retrieval para corpus LGPD - TODO 1
    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking e indexa em Chroma.

        Retorna numero de chunks indexados.

        Ja deixei a estrutura do ciclo. Voce completa as 3 partes marcadas.
        """
        # SEU CODIGO AQUI — TODO 1.A
        # Iterar por todos os PDFs em self.corpus_dir.
        # Para cada PDF, ler todas as paginas com PdfReader e extrair texto.
        # Acumular numa lista `docs` com dicts: {"text": str, "source": str, "page": int}
        # Dica: reaproveite o snippet do notebook 02 (Etapa 1 — Ingestao de PDFs).
        docs: list[dict] = []
        for pdf_path in sorted(self.corpus_dir.glob("*.pdf")):
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    docs.append({
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_num,
                })
        print(f"📚 Ingeridos {len(docs)} documentos de {len(list(self.corpus_dir.glob('*.pdf')))} PDFs.")


        # SEU CODIGO AQUI — TODO 1.B
        # Aplicar RecursiveCharacterTextSplitter com chunk_size=800, overlap=100
        # Quebrar cada doc em chunks e construir lista `chunks` com:
        # {"id": unique_id, "text": str, "source": str, "page": int}
        # Dica: reaproveite o notebook 02 (Etapa 2 — Chunking Recursivo).
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
                    "text": chunk_text,
                    "source": doc["source"],
                    "page": doc["page"],
                })
        print(f"✂️  Gerados {len(chunks)} chunks (size=800, overlap=100).")

        # SEU CODIGO AQUI — TODO 1.C
        # Adicionar chunks no Chroma via self.collection.add(ids=, documents=, metadatas=)
        # Lembre de filtrar metadatas para conter apenas {source, page} (Chroma rejeita listas).

        # TODO 1.C — Adicionar chunks no Chroma EM LOTES (limite Gemini: 100/batch)
        if chunks:
            BATCH_SIZE = 80  # margem de segurança abaixo do limite de 100 do Gemini
            ids = [c["id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    
            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_num = (i // BATCH_SIZE) + 1
                batch_ids = ids[i:i + BATCH_SIZE]
                batch_docs = documents[i:i + BATCH_SIZE]
                batch_metas = metadatas[i:i + BATCH_SIZE]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                )
                print(f"  📦 Batch {batch_num}/{total_batches}: {len(batch_ids)} chunks indexados")
    
            print(f"💾 Total indexados: {len(chunks)} chunks na collection '{self.collection_name}'.")

        return self.collection.count()

    # ------------------------------------------------------------------ TODO 2
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""
        # SEU CODIGO AQUI — TODO 2
        # Usar self.collection.query(query_texts=[query], n_results=k)
        # Retornar lista de dicts: {"text", "source", "page", "distance"}
        # Dica: notebook 02, Etapa 4 — Retrieval.
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
        
        raise NotImplementedError("TODO 2: implementar retrieve()")

    # ------------------------------------------------------------------ TODO 3
    def answer(self, question: str, k: int = 5) -> dict:
        """Pipeline completo: retrieve + augment + generate. Retorna {answer, sources}."""
        hits = self.retrieve(question, k=k)

        # SEU CODIGO AQUI — TODO 3
        # 1. Montar contexto concatenando os textos dos hits com cabecalho [source:page]
        context_parts = []
        sources = []
        for hit in hits:
            header = f"[{hit['source']}:{hit['page']}]"
            context_parts.append(f"{header}\n{hit['text']}")
            sources.append((hit["source"], hit["page"]))
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "Nenhum contexto encontrado."
        
        # 2. Construir prompt com PROMPT_TEMPLATE (definido abaixo)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        
        # 3. Chamar self.client.chat.completions.create(model=self.llm_model, ...)
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente técnico que responde APENAS com base no contexto fornecido. Seja preciso e cite as fontes.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        
        # 4. Retornar {"answer": resposta, "sources": [(s, p) for h in hits]}
        answer = response.choices[0].message.content if response.choices else "Erro ao gerar resposta."
        return {"answer": answer, "sources": sources}
        
        # Dica: notebook 02, Etapa 5 — Augment + Generate.
        raise NotImplementedError("TODO 3: implementar answer()")


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
