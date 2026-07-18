import string
from collections import Counter
import streamlit as st
import spacy
import nltk
from nltk.corpus import stopwords

# Baixa os recursos necessários do NLTK para a Atividade 4
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Carrega o modelo em português do spaCy
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("pt_core_news_sm")
        return spacy.load("pt_core_news_sm")

nlp = load_spacy()

# Configuração da página do Streamlit
st.set_page_config(page_title="NLP Multi-Task App", page_icon="🤖", layout="wide")
st.title("🤖 Plataforma Unificada de Soluções em NLP")
st.markdown("Selecione uma das 10 atividades no menu lateral para testar a funcionalidade.")

# Menu lateral para navegação entre as atividades
atividade = st.sidebar.selectbox(
    "Escolha a Atividade:",
    [
        "Atividade 1: Análise de Sentimentos Simples",
        "Atividade 2: Tokenização de Avaliações",
        "Atividade 3: Classificador de Solicitações Bancárias",
        "Atividade 4: Remoção de Stopwords (NLTK)",
        "Atividade 5: Detecção Automática de Reclamações",
        "Atividade 6: Reconhecimento de Entidades (NER)",
        "Atividade 7: Frequência de Palavras",
        "Atividade 8: Identificador de Intenções para Chatbot",
        "Atividade 9: Normalização de Textos",
        "Atividade 10: Classificador de Feedback Completo"
    ]
)

st.divider()

# -----------------------------------------------------------------------------
# ATIVIDADE 1: Análise de Sentimentos Simples
# -----------------------------------------------------------------------------
if atividade.startswith("Atividade 1:"):
    st.header("📋 Atividade 1 - Análise de Sentimentos por Palavras-Chave")
    text_input = st.text_area("Digite o comentário do cliente:", "O produto é maravilhoso e a entrega foi super rápida!")
    
    if st.button("Analisar Sentimento"):
        doc = nlp(text_input.lower())
        pos_words = {"bom", "ótimo", "maravilhoso", "excelente", "rápido", "gostei", "perfeito", "recomendo"}
        neg_words = {"ruim", "péssimo", "lento", "defeito", "quebrado", "odiei", "atrasou", "pior"}
        
        tokens = [token.text for token in doc]
        pos_score = sum(1 for t in tokens if t in pos_words)
        neg_score = sum(1 for t in tokens if t in neg_words)
        
        if pos_score > neg_score:
            st.success(f"🟢 Sentimento: **Positivo** (Palavras positivas: {pos_score} | Negativas: {neg_score})")
        elif neg_score > pos_score:
            st.error(f"🔴 Sentimento: **Negativo** (Palavras positivas: {pos_score} | Negativas: {neg_score})")
        else:
            st.warning(f"🟡 Sentimento: **Neutro** ou Inconclusivo")

# -----------------------------------------------------------------------------
# ATIVIDADE 2: Tokenização de Avaliações
# -----------------------------------------------------------------------------
elif atividade.startswith("Atividade 2:"):
    st.header("📋 Atividade 2 - Tokenização de Avaliações de E-commerce")
    text_input = st.text_area("Insira a avaliação completa:", "O tênis é confortável, mas a caixa veio amassada.")
    
    if st.button("Tokenizar Texto"):
        doc = nlp(text_input)
        tokens = [token.text for token in doc]
        st.write("### Lista de Tokens Gerados:")
        st.write(tokens)
        st.info(f"Total de tokens extraídos: **{len(tokens)}**")

# -----------------------------------------------------------------------------
# ATIVIDADE 3: Classificador de Solicitações Bancárias
# -----------------------------------------------------------------------------
elif atividade.startswith("Atividade 3:"):
    st.header("📋 Atividade 3 - Classificador de Solicitações Bancárias")
    text_input = st.text_area("Digite a solicitação do cliente:", "Quero fazer o bloqueio do meu cartão agora.")
    
    if st.button("Classificar Solicitação"):
        doc = nlp(text_input.lower())
        tokens = [token.text for token in doc]
        
        if any(w in tokens for w in ["bloquear", "bloqueio", "perdi"]):
            st.info("🎯 Categoria Detectada: **Bloqueio de Cartão**")
        elif any(w in tokens for w in ["segunda", "via", "boleto", "fatura"]):
            st.info("🎯 Categoria Detectada: **Segunda Via de Boleto**")
        else:
            st.warning("🎯 Categoria Detectada: **Outros Assuntos / Não Identificado**")

# -----------------------------------------------------------------------------
# ATIVIDADE 4: Remoção de Stopwords (NLTK)
# -----------------------------------------------------------------------------
elif atividade.startswith("Atividade 4:"):
    st.header("📋 Atividade 4 - Remoção de Stopwords com NLTK")
    text_input = st.text_area("Texto original:", "Esta é uma demonstração de como remover as palavras irrelevantes de um texto.")
    
    if st.button("Remover Stopwords"):
        stop_words = set(stopwords.words("portuguese"))
        words = text_input.split()
        filtered_words = [w for w in words if w.lower().translate(str.maketrans('', '', string.punctuation)) not in stop_words]
        
        st.write("### Texto Filtrado:")
        st.write(" ".join(filtered_words))
        
        col1, col2 = st.columns(2)
        col1.metric("Palavras Originais", len(words))
        col2.metric("Palavras Após Filtro", len(filtered_words))

# -----------------------------------------------------------------------------
# ATIVIDADE 5: Detecção Automática de Reclamações
# -----------------------------------------------------------------------------
elif atividade.startswith("Atividade 5:"):
    st.header("📋 Atividade 5 - Detector Automático de Reclamações")
    text_input = st.text_area("Mensagem do Suporte:", "O sistema apresentou um erro terrível e o serviço está péssimo.")
    
    if st.button("Verificar Alertas"):
        doc = nlp(text_input.lower())
        gatilhos_negativos = {"ruim", "erro", "péssimo", "falha", "problema", "quebrou", "horrível", "bug"}
        
        encontradas = [token.text for token in doc if token.text in gatilhos_negativos]
        
        if encontradas:
            st.error(f"🚨 **Reclamação Detectada!** Palavras críticas encontradas: {list(set(encontradas))}")
        else:
            st.success("✅ Nenhuma palavra crítica de reclamação foi encontrada na mensagem.")

# -----------------------------------------------------------------------------
# ATIVIDADE 6: Reconhecimento de Entidades (NER)
# -----------------------------------------------------------------------------
elif atividade.startswith("Atividade 6:"):
    st.header("📋 Atividade 6 - Extração de Entidades Nomeadas (NER)")
    text_input = st.text_area("Documento de entrada:", "A Microsoft fechou uma parceria com a prefeitura de São Paulo através do CEO Carlos da Silva.")
    
    if st.button("Extrair Entidades"):
        doc = nlp(text_input)
        
        if doc.ents:
            st.write("### Entidades Encontradas:")
            for ent in doc.ents:
                tipo = ent.label_
                if tipo == "ORG": tipo = "🏢 Organização / Empresa"
                elif tipo == "LOC": tipo = "📍 Localização / Cidade"
                elif tipo == "PER": tipo = "👤 Pessoa"
                elif tipo == "MISC": tipo = "✨ Miscelânea"
                
                st.write(f"- **{ent.text}** -> {tipo}")
        else:
            st.warning("Nenhuma entidade nomeada (Pessoas, Empresas, Locais) foi identificada no texto.")

# -----------------------------------------------------------------------------
# ATIVIDADE 7: Frequência de Palavras
# -----------------------------------------------------------------------------
elif atividade.startswith("Atividade 7:"):
    st.header("📋 Atividade 7 - Contador de Frequência de Palavras")
    text_input = st.text_area("Texto do post viral:", "Incrível este evento! O evento trouxe inovação e muita tecnologia para o setor de tecnologia.")
    
    if st.button("Calcular Frequência"):
        doc = nlp(text_input.lower())
        palavras = [token.text for token in doc if not token.is_punct and not token.is_space]
        
        contagem = Counter(palavras)
        
        st.write("### Ranking de Palavras Mais Comuns:")
        for palavra, freq in contagem.most_common(10):
            st.write(f"🔹 **{palavra}**: apareceu {freq} vez(es)")

# -----------------------------------------------------------------------------
# ATIVIDADE 8: Identificador de Intenções para Chatbot (CORRIGIDO)
# -----------------------------------------------------------------------------
elif atividade.startswith("Atividade 8:"):
    st.header("📋 Atividade 8 - Mecanismo de Intenções para Chatbot")
    text_input = st.text_area("Mensagem enviada ao Chatbot:", "Quero cancelar minha assinatura anual por favor.")
    
    if st.button("Identificar Intenção"):
        doc = nlp(text_input.lower())
        tokens = [token.text for token in doc]
        
        intencoes = {
            "🛒 Comprar": ["comprar", "adquirir", "preço", "valor", "assinar", "compras"],
            "❌ Cancelar": ["cancelar", "cancelamento", "excluir", "encerrar", "desistir"],
            "🛠️ Suporte": ["suporte", "ajuda", "ajudar", "problema", "manual", "atendimento"]
        }
        
        intencao_descoberta = None
        for intencao, palavras_chave in intencoes.items():
            if any(p in tokens for p in palavras_chave):
                intencao_descoberta = intencao
                break
        
        if intencao_descoberta:
            st.info(f"Intenção do usuário mapeada: **{intencao_descoberta}**")
        else:
            st.warning("Intenção não compreendida. Direcionando para triagem geral.")

# -----------------------------------------------------------------------------
# ATIVIDADE 9: Normalização de Textos
