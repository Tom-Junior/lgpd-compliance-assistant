"""Model routing cheap-first com fallback.

Reaproveita o notebook 05. Voce vai preencher 1 TODO aqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


# ------------------------------------------------------------------ TODO 6
def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium).

    Estrategia heuristica simples. Em producao, evoluiria para classifier treinado.
    """
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    # Normaliza a query para comparação
    query_lower = query.lower().strip()
    
    # Palavras-chave que indicam complexidade
    complex_keywords = [
        "explique", "compare", "analise", "projete", "arquitetura",
        "implemente", "desenvolva", "crie", "construa", "designe",
        "como", "por que", "porque", "explique detalhadamente",
        "compare as", "analise os", "projete um", "implemente um",
        "quais são as implicações", "como você faria", "qual a melhor abordagem"
    ]
    
    # Regra 1: Query curta e termina em "?" → simple
    if len(query) < 60 and query.strip().endswith("?"):
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason=f"Query curta ({len(query)} chars) e termina em '?'"
        )
    
    # Regra 2: Contém palavras-chave de complexidade → complex
    for keyword in complex_keywords:
        if keyword in query_lower:
            return RouteDecision(
                model=premium_model,
                complexity="complex",
                reason=f"Contém palavra-chave de complexidade: '{keyword}'"
            )
    
    # Regra 3: Query longa (> 150 chars) → complex
    if len(query) > 150:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"Query longa ({len(query)} chars) sugere pergunta complexa"
        )
    
    # Regra 4: Múltiplas perguntas (mais de 1 "?") → complex
    if query.count("?") > 1:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"Múltiplas perguntas ({query.count('?')} '?') sugere análise complexa"
        )
    
    # Default → simple
    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="Nenhuma regra de complexidade aplicada, usando modelo barato"
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()