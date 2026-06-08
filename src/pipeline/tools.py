"""Function-calling / tool-use — registro de tools usadas pelo agente.

Reaproveita o LAB-001. Voce vai preencher 1 TODO aqui (sua tool especifica).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader


# ============================================================================
# TODO 4 — Sua tool especifica do dominio
# ============================================================================
# Cada projeto precisa de UMA tool customizada que faca sentido para o problema.
# Exemplos por dominio:
#   - Livro tecnico:    lookup_chapter(chapter: int) -> str
#   - Changelog:        check_compat(lib: str, version: str) -> dict
#   - Podcast:          get_timestamp(quote: str) -> str
#   - Codigo:           run_snippet(code: str) -> str  (sandboxed)
#   - Documentos legais: cite_article(law: str, article: int) -> str
#
# 1. Implemente a funcao Python real abaixo (substitua o exemplo)
# 2. Adicione o schema JSON em TOOLS abaixo
# 3. Registre em TOOL_REGISTRY
# ============================================================================


# ----------------------------------------------------------------------------
# Cache em memória dos artigos da LGPD (parseado uma única vez)
# ----------------------------------------------------------------------------
_ARTICLES_CACHE: dict[int, str] = {}
_CORPUS_DIR = Path("data/corpus")


def _extract_text_from_pdfs(corpus_dir: Path = _CORPUS_DIR) -> str:
    """Extrai texto completo de todos os PDFs no diretório corpus.

    Lê todos os arquivos .pdf em corpus_dir, extraindo o texto de cada página
    e concatenando em uma única string.

    Returns:
        String com o texto integral de todos os PDFs concatenados.
    """
    full_text = []
    pdf_files = sorted(corpus_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  Nenhum arquivo PDF encontrado em {corpus_dir}")
        return ""

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    full_text.append(text)
            print(f"📄 Extraído texto de: {pdf_path.name} ({len(reader.pages)} páginas)")
        except Exception as e:
            print(f"⚠️  Erro ao ler {pdf_path.name}: {e}")

    return "\n\n".join(full_text)


def _parse_lgpd_articles(corpus_dir: Path = _CORPUS_DIR) -> dict[int, str]:
    """Parseia o texto da LGPD extraindo cada artigo pelo número.

    Estratégia:
      1. Lê todos os PDFs do diretório corpus e extrai texto completo.
      2. Usa regex para capturar blocos que começam em "Art. N" até o próximo
         "Art." ou fim do texto.
      3. Armazena em cache global para evitar re-parsear em cada chamada.

    Returns:
        Dicionário {numero_artigo: texto_integral_do_artigo}.
    """
    global _ARTICLES_CACHE
    if _ARTICLES_CACHE:
        return _ARTICLES_CACHE

    text = _extract_text_from_pdfs(corpus_dir)

    if not text:
        print("⚠️  Nenhum texto extraído dos PDFs do corpus.")
        return {}

    # Regex captura "Art. N" até o próximo "Art." ou fim do texto (DOTALL)
    pattern = r"(Art\.\s*\d+[º°a-zA-Z]?\s*[-–—]?.*?)(?=\n\s*Art\.\s*\d+|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        art_num_match = re.search(r"Art\.\s*(\d+)", match)
        if art_num_match:
            num = int(art_num_match.group(1))
            _ARTICLES_CACHE[num] = match.strip()

    print(f"📜 Parseados {len(_ARTICLES_CACHE)} artigos da LGPD em memória.")
    return _ARTICLES_CACHE


def cite_article(article_number: int) -> str:
    """Retorna o texto integral do artigo N da LGPD (Lei 13.709/2018).

    Esta tool é usada para citar artigos específicos da lei e evitar
    alucinações do LLM (LLMs tendem a inventar números de artigos).

    Args:
        article_number: Número do artigo da LGPD (1 a 65).

    Returns:
        String com o texto integral do artigo, ou mensagem de erro se
        o artigo não for encontrado no corpus.
    """
    articles = _parse_lgpd_articles()

    if not articles:
        return (
            "ERRO: Corpus da LGPD não carregado. "
            "Verifique se existem arquivos PDF em data/corpus/."
        )

    if not isinstance(article_number, int) or article_number < 1:
        return f"ERRO: número de artigo inválido: {article_number}. Deve ser inteiro positivo."

    if article_number not in articles:
        max_art = max(articles.keys())
        return (
            f"ERRO: Artigo {article_number} não encontrado na LGPD. "
            f"Artigos disponíveis no corpus: 1 a {max_art}."
        )

    return (
        f"📜 Lei 13.709/2018 — LGPD — Art. {article_number}:\n\n"
        f"{articles[article_number]}"
    )


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cite_article",
            "description": (
                "Retorna o texto INTEGRAL do artigo N da LGPD (Lei 13.709/2018). "
                "Use SEMPRE que precisar citar um artigo específico da lei para "
                "responder perguntas sobre conformidade, bases legais, direitos do "
                "titular, sanções, segurança de dados, etc. Esta tool evita "
                "alucinações — o LLM não deve inventar o conteúdo de artigos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "integer",
                        "description": (
                            "Número do artigo da LGPD (1 a 65). Exemplos: "
                            "5 (princípios), 7 (bases legais), 11 (dados sensíveis), "
                            "18 (direitos do titular), 46 (segurança), 48 (incidentes), "
                            "52 (sanções)."
                        ),
                    },
                },
                "required": ["article_number"],
            },
        },
    },
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "cite_article": cite_article,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
