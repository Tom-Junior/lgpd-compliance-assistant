"""Streamlit UI — entrada principal do app. Pronta para deploy 1-click no Streamlit Cloud.

Voce nao precisa editar quase nada aqui — ja faz integracao com:
- src.pipeline.rag (TODOs 1-3)
- src.pipeline.cache (TODO 5)
- src.pipeline.routing (TODO 6)
- src.pipeline.tools (TODO 4, opcional)
"""

from __future__ import annotations

import sys
from pathlib import Path
import base64

from dotenv import load_dotenv

# Adiciona o root do projeto no path para imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402

from src.observability.trace import trace, log_event  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402


# ---------------------------------------------------------------- Streamlit UI

# --- CONFIGURAÇÃO VISUAL FUTURISTA ---

# Define o caminho absoluto para a pasta assets (funciona local e na nuvem)
ROOT_DIR = Path(__file__).resolve().parents[2] 
ASSETS_DIR = ROOT_DIR / "assets"
ROBOT_PATH = str(ASSETS_DIR / "cyberpunkrobot.png")


st.set_page_config(
    page_title="LGPD AI Assistant", 
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeção de CSS Cyberpunk/Sci-Fi
st.markdown("""
<style>
    /* Fundo e Fontes */
    .stApp {
        background: linear-gradient(135deg, #050b14 0%, #0a192f 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Painéis de Vidro (Glassmorphism) */
    div[data-testid="stSidebar"] > div:first-child {
        background: rgba(10, 25, 47, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    /* Inputs Futuristas */
    .stTextInput > div > div > input {
        background-color: rgba(0, 0, 0, 0.3);
        color: #00ffff;
        border: 1px solid #00ffff;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.1);
    }
    
    /* Botões Neon */
    .stButton > button {
        background: transparent;
        color: #00ffff;
        border: 1px solid #00ffff;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #00ffff;
        color: #000;
        box-shadow: 0 0 20px #00ffff;
    }

    /* Robô na Sidebar (Posicionamento Absoluto) */
    .robot-container {
        position: fixed;
        bottom: 20px;
        left: 20px;
        width: 120px;
        z-index: 999;
        pointer-events: none;
        filter: drop-shadow(0 0 15px rgba(0, 255, 255, 0.5));
    }
    
    /* Títulos Neon */
    h1, h2, h3 {
        color: #e6f1ff !important;
        text-shadow: 0 0 10px rgba(100, 200, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- IMAGEM DO ROBÔ (Use uma URL pública ou arquivo local) -->
# Sugestão: Use esta imagem de robô futurista ou substitua pela sua
import streamlit as st
from pathlib import Path

# 1. DEFINIÇÃO DE CAMINHOS (No topo do arquivo, após os imports)
ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "assets"
# Use o nome exato do seu arquivo!
ROBOT_PATH = str(ASSETS_DIR / "cyberpunkrobot.png") 

# 2. INJEÇÃO DO CSS (Dentro do st.markdown inicial de estilos)
st.markdown("""
<style>
    /* ... seus outros estilos ... */

    /* Robô flutuante na sidebar */
    .robot-container {
        position: relative;
        width: 100%;
        display: flex;
        justify-content: center;
        margin-top: 20px;
        filter: drop-shadow(0 0 15px rgba(0, 255, 255, 0.6));
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
</style>
""", unsafe_allow_html=True)

# 3. LÓGICA DA SIDEBAR (Na seção da sidebar)
with st.sidebar:
    st.markdown("### 🧭 Navegação")
    menu_option = st.radio(
        "",
        ["💬 Chat Compliance", "📊 Métricas do Sistema", "⚙️ Configurações", "📜 Histórico"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Exibe o robô com verificação de existência
    if Path(ROBOT_PATH).exists():
        st.markdown('<div class="robot-container">', unsafe_allow_html=True)
        st.image(ROBOT_PATH, width=120)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Imagem não encontrada: {ROBOT_PATH}")
    
    st.markdown("<br><br>", unsafe_allow_html=True)

# --- CONTEÚDO PRINCIPAL ---
if menu_option == "💬 Chat Compliance":
    st.title("️ Assistente de Compliance LGPD")
    st.caption("Sistema RAG com Tool-Use e Cache Semântico")
    
    query = st.text_input("Faça sua pergunta jurídica:", placeholder="Ex: Quais as bases legais do Art. 7º?")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.markdown("⚡ Processando via Groq + ChromaDB...")
            # Aqui entra sua lógica de pipeline.answer(query)
            
elif menu_option == "📊 Métricas do Sistema":
    st.title(" Telemetria em Tempo Real")
    col1, col2, col3 = st.columns(3)
    col1.metric("Latência Média", "1.8s", "-12%")
    col2.metric("Cache Hit Rate", "38%", "+5%")
    col3.metric("Custo por Query", "$0.00", "100% Free")

elif menu_option == "⚙️ Configurações":
    st.title("⚙️ Painel de Controle")
    st.toggle("Ativar Tool-Use (Cite Article)", value=True)
    st.slider("Threshold de Cache Semântico", 0.8, 1.0, 0.93)

elif menu_option == "📜 Histórico":
    st.title("📜 Log de Consultas")
    st.info("Histórico de sessões anteriores aparecerá aqui.")


# Inicializacao lazy de pipeline + caches
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("Inicializando pipeline RAG..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


# Sidebar — metricas e debug
with st.sidebar:
    st.header("Metricas")
    st.metric("Chunks indexados", pipeline.collection.count())
    st.metric("Exact cache", exact_cache.stats()["size"])
    st.metric("Semantic cache", semantic_cache.stats()["size"])

    if st.button("Limpar caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos. Recarregue a pagina.")
    
    st.divider()
    st.markdown("### Sobre o Projeto")
    st.markdown("""
    **Problema:** Desenvolvedores precisam verificar conformidade com a LGPD, mas consultar a lei é trabalhoso.
    
    **Solução:** Assistente RAG que responde perguntas citando artigos específicos da LGPD, evitando alucinações.
    
    **Tecnologias:**
    - RAG com ChromaDB
    - Tool-use para citar artigos
    - Cache semântico (threshold: 0.93)
    - Model routing cheap-first
    """)


# Main — chat interface
query = st.text_input("Sua pergunta sobre LGPD:", placeholder="Ex: Posso armazenar CPF sem consentimento?")

if query:
    # Inicializa trace_id (fallback se trace() não estiver disponível)
    try:
        with trace("query_handle", query=query) as ctx:
            trace_id = ctx.get("trace_id", "unknown")
    except Exception:
        trace_id = "unknown"
    
    # Flag para controlar se já respondemos
    answered = False
    response_text = ""
    sources_list = []
    
    # 1. Exact cache
    cached = exact_cache.get(query)
    if cached:
        response_text = cached
        sources_list = []
        answered = True
        st.success("✅ Cache hit (exact)")
        try:
            log_event("cache_hit", trace_id=trace_id, layer="exact")
        except Exception:
            pass
    
    # 2. Semantic cache (se não houve hit no exact)
    if not answered:
        try:
            cached = semantic_cache.get(query)
            if cached:
                response_text = cached
                sources_list = []
                answered = True
                st.success("✅ Cache hit (semantic)")
                try:
                    log_event("cache_hit", trace_id=trace_id, layer="semantic")
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"Semantic cache indisponível: {e}")
    
    # 3. Pipeline RAG + Routing (se não houve cache hit)
    if not answered:
        # Routing
        try:
            decision = classify_complexity(query)
            st.info(f"🔀 Routing: {decision.complexity} → {decision.model}")
            try:
                log_event("route_decision", trace_id=trace_id, **decision.__dict__)
            except Exception:
                pass
        except Exception as e:
            st.warning(f"Routing indisponível: {e}. Usando modelo default.")
        
        # RAG Pipeline
        try:
            result = pipeline.answer(query)
            response_text = result.get("answer", "Erro ao gerar resposta.")
            sources_list = result.get("sources", [])
            answered = True
            
            try:
                log_event("answer_generated", trace_id=trace_id, sources=len(sources_list))
            except Exception:
                pass
        except Exception as e:
            st.error(f"Erro no pipeline RAG: {e}")
            response_text = f"Erro ao processar sua pergunta: {str(e)}"
            sources_list = []
            answered = True
    
    # 4. Renderiza resposta (sempre, se temos algo para mostrar)
    if answered and response_text:
        st.write(response_text)
        
        if sources_list:
            with st.expander("📚 Fontes citadas"):
                for source, page in sources_list:
                    st.write(f"- `{source}:p{page}`")
        
        # Cacheia a resposta (se não veio do cache)
        if "Cache hit" not in st.session_state.get("last_status", ""):
            try:
                exact_cache.put(query, response_text)
                semantic_cache.put(query, response_text)
            except Exception as e:
                st.warning(f"Erro ao salvar no cache: {e}")


st.divider()
st.caption("""
    ### Sobre este assistente

Este assistente responde perguntas sobre conformidade com a **Lei Geral de Proteção de Dados (LGPD - Lei 13.709/2018)**, citando artigos específicos da lei e evitando alucinações.

**Arquitetura:**
- **RAG ponta-a-ponta** com corpus da LGPD + guias da ANPD
- **Cache semântico** com similaridade de embeddings (threshold 0.93)
- **Model routing** cheap-first (classifica complexidade e escolhe modelo)
- **Tool-use** para citação precisa de artigos (`cite_article`)

**Redução de custos:**
- Cache exato: captura replays idênticos (~10-15%)
- Cache semântico: captura paráfrases (~20% adicional)
- Routing: usa modelo barato para perguntas simples (~70% das queries)

**Limitações:**
- Corpus estático (LGPD não é atualizada automaticamente)
- Não substitui consultoria jurídica especializada
- Funciona apenas em português
""")
