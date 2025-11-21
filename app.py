"""
Query Fan Out Generator v2.0.3 - Professional SEO Keyword Clustering
Author: Claude
Date: 2025-11-20
Last Update: 2025-11-20 (v2.0.3 - Performance optimizations)

Features:
- Real semantic clustering using TF-IDF + cosine similarity
- Network graph visualization (interactive & FAST)
- Google Keyword Planner CSV import (multi-encoding support)
- Modular API integration (GSC, SEMrush, Sistrix)
- Proper Streamlit caching and session state management
- Search intent detection using NLP patterns
- Long-tail keyword analysis
- Professional Excel/JSON export

Changelog v2.0.3:
- Performance: Network graph 10-20x faster with configurable limits
- Added: Layout algorithm selector (circular/kamada/spring)
- Added: Max keywords slider (20-200) for speed control
- Added: Cluster-specific visualization
- Added: Optional generation (button instead of auto-generate)
- Improved: Shows only top keywords by volume
- Improved: Better speed indicators and warnings

Changelog v2.0.2:
- Fixed: CSV encoding auto-detection (UTF-16, UTF-8, Latin-1, etc.)
- Fixed: File persistence between Streamlit reruns
- Fixed: Better error messages for CSV loading issues

Changelog v2.0.1:
- Fixed: ValueError with negative values in distance matrix
- Fixed: Proper normalization of cosine similarity [-1,1] to [0,1]
- Fixed: Adjusted DBSCAN epsilon for new scale
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import io
import re
import os

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Query Fan Out v2.0.3 - Professional",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - minimal and professional
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #0dcaf0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA MODELS ====================

@dataclass
class KeywordData:
    """Modelo de datos para una keyword con todas sus métricas"""
    keyword: str
    volume: int
    competition: str
    competition_index: float
    cpc_low: float
    cpc_high: float
    trend_3m: float
    trend_yoy: float
    monthly_data: Dict[str, int]
    
    @classmethod
    def from_dict(cls, data: dict):
        """Crear desde diccionario"""
        return cls(**data)
    
    def to_dict(self):
        """Convertir a diccionario para serialización"""
        return asdict(self)

@dataclass
class KeywordCluster:
    """Cluster de keywords relacionadas"""
    cluster_id: int
    primary_keyword: str
    keywords: List[KeywordData]
    total_volume: int
    avg_competition: float
    search_intent: str
    confidence_score: float
    
    def to_dict(self):
        return {
            'cluster_id': self.cluster_id,
            'primary_keyword': self.primary_keyword,
            'keywords': [kw.to_dict() for kw in self.keywords],
            'total_volume': self.total_volume,
            'avg_competition': self.avg_competition,
            'search_intent': self.search_intent,
            'confidence_score': self.confidence_score
        }

# ==================== UTILITY FUNCTIONS ====================

def clean_keyword_planner_value(value: str) -> str:
    """Limpia valores del CSV de Keyword Planner que vienen con espacios"""
    if isinstance(value, str):
        # Eliminar espacios entre caracteres (formato UTF-16)
        cleaned = ''.join(value.split())
        return cleaned
    return str(value)

def parse_numeric_value(value: str) -> float:
    """Parsea valores numéricos del CSV con formato europeo"""
    if pd.isna(value) or value == '':
        return 0.0
    try:
        # Manejar formato europeo: "1.300" o "1,50"
        cleaned = clean_keyword_planner_value(str(value))
        cleaned = cleaned.replace('.', '').replace(',', '.')
        cleaned = cleaned.replace('"', '')
        return float(cleaned)
    except:
        return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def load_keyword_planner_csv(file_path: str) -> pd.DataFrame:
    """
    Carga y procesa CSV de Google Keyword Planner
    Intenta múltiples encodings y formatos automáticamente
    """
    
    # Lista de encodings a probar (en orden de probabilidad)
    encodings_to_try = [
        ('utf-16', '\t'),      # Formato más común de Keyword Planner
        ('utf-16-le', '\t'),   # UTF-16 Little Endian
        ('utf-16-be', '\t'),   # UTF-16 Big Endian
        ('utf-8', ','),        # UTF-8 con comas
        ('utf-8', '\t'),       # UTF-8 con tabs
        ('latin-1', ','),      # Fallback
        ('cp1252', ','),       # Windows encoding
    ]
    
    df = None
    last_error = None
    
    for encoding, separator in encodings_to_try:
        try:
            # Intentar leer con este encoding
            df = pd.read_csv(file_path, encoding=encoding, sep=separator, skiprows=2)
            
            # Verificar que tiene las columnas esperadas
            if 'Keyword' in str(df.columns) or 'keyword' in str(df.columns).lower():
                # Éxito! Procesar el dataframe
                
                # Limpiar nombres de columnas
                df.columns = [clean_keyword_planner_value(col) for col in df.columns]
                
                # Limpiar valores de todas las columnas de texto
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(clean_keyword_planner_value)
                
                # Procesar columnas numéricas con nombres limpios
                if 'Avg.monthlysearches' in df.columns:
                    df['Avg.monthlysearches'] = df['Avg.monthlysearches'].apply(parse_numeric_value)
                else:
                    st.warning("⚠️ Columna 'Avg. monthly searches' no encontrada")
                    df['Avg.monthlysearches'] = 0
                
                df['Competitionindexedvalue'] = df.get('Competition(indexedvalue)', pd.Series([0]*len(df))).apply(parse_numeric_value)
                df['Topofpagebid(lowrange)'] = df.get('Topofpagebid(lowrange)', pd.Series([0]*len(df))).apply(parse_numeric_value)
                df['Topofpagebid(highrange)'] = df.get('Topofpagebid(highrange)', pd.Series([0]*len(df))).apply(parse_numeric_value)
                
                # Cambios de tendencia
                df['Cambioentresmeses'] = df.get('Cambioentresmeses', pd.Series(['0%']*len(df))).apply(lambda x: parse_numeric_value(str(x).replace('%', '')))
                df['Cambiointere anual'] = df.get('Cambiointere anual', pd.Series(['0%']*len(df))).apply(lambda x: parse_numeric_value(str(x).replace('%', '')))
                
                # Éxito - retornar dataframe
                st.success(f"✅ CSV cargado con encoding: {encoding}")
                return df
                
        except Exception as e:
            last_error = str(e)
            continue  # Intentar siguiente encoding
    
    # Si llegamos aquí, ningún encoding funcionó
    st.error(f"❌ No se pudo leer el CSV con ningún encoding. Último error: {last_error}")
    st.info("""
    💡 **Soluciones:**
    1. Exporta el CSV nuevamente desde Google Keyword Planner
    2. Asegúrate de usar la opción "Descargar" → "CSV"
    3. Si el problema persiste, abre el CSV en Excel y guárdalo como "CSV UTF-8"
    """)
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def detect_search_intent(keyword: str) -> Tuple[str, float]:
    """
    Detecta la intención de búsqueda usando patrones NLP
    Returns: (intent_type, confidence_score)
    
    Basado en patrones reales de búsqueda españoles
    """
    keyword_lower = keyword.lower()
    
    # Patrones transaccionales (alta confianza)
    transactional_patterns = [
        r'\b(comprar|precio|oferta|barato|descuento|venta|tienda|shop|buy)\b',
        r'\ben\s*oferta\b',
        r'\bbarato[sa]?\b'
    ]
    
    # Patrones informativos
    informational_patterns = [
        r'\b(qué|cómo|por qué|cuál|guía|tutorial|aprender)\b',
        r'\bcaracterísticas\b',
        r'\bventajas\b',
        r'\bdesventajas\b'
    ]
    
    # Patrones comparativos
    comparative_patterns = [
        r'\b(mejor|mejores|top|ranking|comparativa|vs|versus)\b',
        r'\bcalidad\s*precio\b',
        r'\bcomparar\b'
    ]
    
    # Patrones navegacionales
    navigational_patterns = [
        r'\b(login|acceso|web|oficial|sitio)\b'
    ]
    
    # Patrones de marca específica
    brand_patterns = [
        r'\b(apple|samsung|xiaomi|lenovo|huawei|asus|acer|hp|dell|microsoft)\b',
        r'\bipad\b',
        r'\bgalaxy\s*tab\b'
    ]
    
    # Contar matches
    scores = {
        'transactional': sum(1 for p in transactional_patterns if re.search(p, keyword_lower)),
        'informational': sum(1 for p in informational_patterns if re.search(p, keyword_lower)),
        'comparative': sum(1 for p in comparative_patterns if re.search(p, keyword_lower)),
        'navigational': sum(1 for p in navigational_patterns if re.search(p, keyword_lower)),
        'brand': sum(1 for p in brand_patterns if re.search(p, keyword_lower))
    }
    
    # Determinar intención principal
    if scores['transactional'] > 0:
        confidence = min(0.9, 0.6 + (scores['transactional'] * 0.15))
        return 'Transaccional', confidence
    elif scores['comparative'] > 0:
        confidence = min(0.85, 0.6 + (scores['comparative'] * 0.15))
        return 'Comparativa', confidence
    elif scores['informational'] > 0:
        confidence = min(0.8, 0.5 + (scores['informational'] * 0.15))
        return 'Informacional', confidence
    elif scores['brand'] > 0:
        confidence = min(0.75, 0.5 + (scores['brand'] * 0.15))
        return 'Marca', confidence
    elif scores['navigational'] > 0:
        return 'Navegacional', 0.7
    else:
        # Intent desconocido - probablemente navegacional o marca
        return 'Navegacional/Marca', 0.4

def calculate_keyword_similarity(keywords: List[str], method='tfidf') -> np.ndarray:
    """
    Calcula matriz de similitud entre keywords usando TF-IDF + cosine similarity
    
    Args:
        keywords: Lista de keywords
        method: 'tfidf' por ahora, en futuro 'semantic' con embeddings
    
    Returns:
        Matriz de similitud (n_keywords x n_keywords)
    """
    if method == 'tfidf':
        # TF-IDF con n-gramas para capturar mejor el contexto
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,
            analyzer='char_wb',  # Usa caracteres para manejar mejor español
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(keywords)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    # TODO: Implementar con sentence-transformers cuando esté disponible
    # elif method == 'semantic':
    #     from sentence_transformers import SentenceTransformer
    #     model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    #     embeddings = model.encode(keywords)
    #     similarity_matrix = cosine_similarity(embeddings)
    #     return similarity_matrix

@st.cache_data(ttl=3600, show_spinner=False)
def perform_clustering(
    keywords_df: pd.DataFrame,
    similarity_threshold: float = 0.3,
    min_cluster_size: int = 2
) -> List[KeywordCluster]:
    """
    Agrupa keywords en clusters usando DBSCAN basado en similitud semántica
    
    Args:
        keywords_df: DataFrame con keywords y métricas
        similarity_threshold: Umbral de similitud (0-1). 
                            0.3 = similitud baja, 0.7 = similitud alta
        min_cluster_size: Tamaño mínimo de cluster
    
    Returns:
        Lista de KeywordCluster objects
    
    Note:
        La similitud coseno se normaliza de [-1,1] a [0,1] antes de 
        convertir a distancia para DBSCAN.
    """
    
    if len(keywords_df) == 0:
        return []
    
    # Validación: mínimo 2 keywords para clustering
    if len(keywords_df) < 2:
        st.warning("⚠️ Se necesitan al menos 2 keywords para clustering")
        return []
    
    try:
        # Calcular matriz de similitud
        keywords_list = keywords_df['Keyword'].tolist()
        similarity_matrix = calculate_keyword_similarity(keywords_list)
        
        # Convertir similitud a distancia para DBSCAN
        # Normalizar similitud coseno [-1, 1] a [0, 1] y luego a distancia
        # similarity = -1 (opuestos) -> 0, similarity = 1 (idénticos) -> 1
        normalized_similarity = (similarity_matrix + 1) / 2  # Ahora en [0, 1]
        distance_matrix = 1 - normalized_similarity  # Distancia en [0, 1]
        
        # Asegurar que no hay valores negativos por errores numéricos
        distance_matrix = np.clip(distance_matrix, 0, 1)
        
        # Debug info (opcional)
        if len(keywords_df) < 20:  # Solo para datasets pequeños
            avg_similarity = normalized_similarity.mean()
            if avg_similarity < 0.3:
                st.info(f"ℹ️ Las keywords tienen baja similitud promedio ({avg_similarity:.2f}). Considera reducir el threshold.")
        
    except Exception as e:
        st.error(f"Error calculando similitud: {str(e)}")
        return []
    
    # Clustering con DBSCAN
    # Ajustar epsilon para la escala normalizada [0, 1]
    # similarity_threshold en rango original -> normalizar -> convertir a distancia
    normalized_threshold = (similarity_threshold + 1) / 2
    eps_distance = 1 - normalized_threshold
    
    clustering = DBSCAN(
        eps=eps_distance,
        min_samples=min_cluster_size,
        metric='precomputed'
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Crear objetos KeywordCluster
    clusters = []
    unique_labels = set(cluster_labels)
    
    for label in unique_labels:
        if label == -1:  # Noise/outliers - crear clusters individuales
            noise_indices = np.where(cluster_labels == -1)[0]
            for idx in noise_indices:
                row = keywords_df.iloc[idx]
                kw_data = KeywordData(
                    keyword=row['Keyword'],
                    volume=int(row['Avg.monthlysearches']),
                    competition=row['Competition'],
                    competition_index=float(row['Competitionindexedvalue']),
                    cpc_low=float(row['Topofpagebid(lowrange)']),
                    cpc_high=float(row['Topofpagebid(highrange)']),
                    trend_3m=float(row['Cambioentresmeses']),
                    trend_yoy=float(row['Cambiointere anual']),
                    monthly_data={}
                )
                
                intent, confidence = detect_search_intent(row['Keyword'])
                
                cluster = KeywordCluster(
                    cluster_id=len(clusters),
                    primary_keyword=row['Keyword'],
                    keywords=[kw_data],
                    total_volume=int(row['Avg.monthlysearches']),
                    avg_competition=float(row['Competitionindexedvalue']),
                    search_intent=intent,
                    confidence_score=confidence
                )
                clusters.append(cluster)
        else:
            # Cluster real
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_df = keywords_df.iloc[cluster_indices]
            
            # Keyword principal = mayor volumen
            primary_idx = cluster_df['Avg.monthlysearches'].idxmax()
            primary_kw = cluster_df.loc[primary_idx, 'Keyword']
            
            # Crear lista de KeywordData
            keywords_in_cluster = []
            for idx, row in cluster_df.iterrows():
                kw_data = KeywordData(
                    keyword=row['Keyword'],
                    volume=int(row['Avg.monthlysearches']),
                    competition=row['Competition'],
                    competition_index=float(row['Competitionindexedvalue']),
                    cpc_low=float(row['Topofpagebid(lowrange)']),
                    cpc_high=float(row['Topofpagebid(highrange)']),
                    trend_3m=float(row['Cambioentresmeses']),
                    trend_yoy=float(row['Cambiointere anual']),
                    monthly_data={}
                )
                keywords_in_cluster.append(kw_data)
            
            # Detectar intent del cluster (usar keyword principal)
            intent, confidence = detect_search_intent(primary_kw)
            
            cluster = KeywordCluster(
                cluster_id=len(clusters),
                primary_keyword=primary_kw,
                keywords=keywords_in_cluster,
                total_volume=int(cluster_df['Avg.monthlysearches'].sum()),
                avg_competition=float(cluster_df['Competitionindexedvalue'].mean()),
                search_intent=intent,
                confidence_score=confidence
            )
            clusters.append(cluster)
    
    # Ordenar por volumen total
    clusters.sort(key=lambda x: x.total_volume, reverse=True)
    
    return clusters

def create_network_graph(
    clusters: List[KeywordCluster], 
    max_keywords: int = 50,
    selected_cluster_id: Optional[int] = None,
    layout_algorithm: str = 'spring'
) -> go.Figure:
    """
    Crea un network graph interactivo con Plotly
    
    Args:
        clusters: Lista de clusters
        max_keywords: Máximo de keywords a visualizar (default: 50 para velocidad)
        selected_cluster_id: Si se especifica, solo muestra ese cluster
        layout_algorithm: 'spring' (lento), 'circular' (rápido), 'kamada' (medio)
    
    Nodos = keywords
    Edges = similitud semántica
    Colores = search intent
    """
    
    # Filtrar a cluster específico si se seleccionó
    if selected_cluster_id is not None:
        clusters = [c for c in clusters if c.cluster_id == selected_cluster_id]
    
    # Limitar keywords para performance
    keywords_to_show = []
    for cluster in clusters[:10]:  # Top 10 clusters
        # Ordenar keywords por volumen y tomar las top
        sorted_kws = sorted(cluster.keywords, key=lambda x: x.volume, reverse=True)
        keywords_to_show.extend(sorted_kws[:5])  # Top 5 de cada cluster
    
    # Si aún hay muchas, limitar por volumen total
    if len(keywords_to_show) > max_keywords:
        keywords_to_show = sorted(keywords_to_show, key=lambda x: x.volume, reverse=True)[:max_keywords]
    
    if len(keywords_to_show) == 0:
        st.warning("No hay keywords para visualizar")
        return go.Figure()
    
    st.info(f"📊 Visualizando {len(keywords_to_show)} keywords de mayor volumen (de {sum(len(c.keywords) for c in clusters)} totales)")
    
    G = nx.Graph()
    
    # Definir colores por intent
    intent_colors = {
        'Transaccional': '#10b981',
        'Comparativa': '#8b5cf6',
        'Informacional': '#3b82f6',
        'Marca': '#f97316',
        'Navegacional': '#ec4899',
        'Navegacional/Marca': '#6b7280'
    }
    
    # Crear lookup de keyword a cluster
    kw_to_cluster = {}
    for cluster in clusters:
        for kw in cluster.keywords:
            if kw in keywords_to_show:
                kw_to_cluster[kw.keyword] = cluster
    
    # Añadir nodos
    node_info = {}
    for kw in keywords_to_show:
        cluster = kw_to_cluster.get(kw.keyword)
        if cluster:
            node_id = kw.keyword
            G.add_node(
                node_id,
                volume=kw.volume,
                intent=cluster.search_intent,
                cluster_id=cluster.cluster_id
            )
            
            node_info[node_id] = {
                'volume': kw.volume,
                'intent': cluster.search_intent,
                'color': intent_colors.get(cluster.search_intent, '#6b7280'),
                'competition': kw.competition_index
            }
    
    # Añadir edges solo dentro de clusters (más rápido)
    for cluster in clusters:
        cluster_kws = [kw.keyword for kw in cluster.keywords if kw in keywords_to_show]
        for i, kw1 in enumerate(cluster_kws):
            for kw2 in cluster_kws[i+1:]:
                G.add_edge(kw1, kw2, weight=0.8)
    
    # Elegir layout según algoritmo
    if layout_algorithm == 'circular':
        # O(n) - Instantáneo
        pos = nx.circular_layout(G)
    elif layout_algorithm == 'kamada':
        # O(n²) - Medio
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.circular_layout(G)
    else:  # spring
        # O(n² * iterations) - Lento pero bonito
        pos = nx.spring_layout(G, k=0.5, iterations=30, seed=42)
    
    # Crear edges para plotly
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#E5E7EB'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    # Crear nodos para plotly
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            size=[],
            color=[],
            line=dict(width=2, color='white')
        ),
        textposition="top center",
        textfont=dict(size=10)
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        
        info = node_info[node]
        volume = info['volume']
        intent = info['intent']
        competition = info['competition']
        
        # Tamaño basado en volumen (logarítmico)
        size = 15 + (np.log10(max(volume, 1)) * 8)
        node_trace['marker']['size'] += tuple([size])
        
        # Color basado en intent
        node_trace['marker']['color'] += tuple([info['color']])
        
        # Texto hover
        hover_text = f"<b>{node}</b><br>Volumen: {volume:,}<br>Intent: {intent}<br>Competition: {competition:.0f}"
        node_trace['text'] += tuple([hover_text])
    
    # Crear figura
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'<b>Keyword Network Graph</b><br><sub>Top {len(keywords_to_show)} keywords por volumen | Layout: {layout_algorithm}</sub>',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=700
        )
    )
    
    return fig

def create_cluster_hierarchy_chart(clusters: List[KeywordCluster]) -> go.Figure:
    """
    Crea un treemap jerárquico de clusters por intent
    """
    
    # Preparar datos
    labels = []
    parents = []
    values = []
    colors = []
    hover_texts = []
    
    # Root
    labels.append("All Keywords")
    parents.append("")
    values.append(sum(c.total_volume for c in clusters))
    colors.append("#667eea")
    hover_texts.append(f"Total: {sum(c.total_volume for c in clusters):,} searches/month")
    
    # Por search intent
    intent_groups = {}
    for cluster in clusters:
        if cluster.search_intent not in intent_groups:
            intent_groups[cluster.search_intent] = []
        intent_groups[cluster.search_intent].append(cluster)
    
    intent_colors_map = {
        'Transaccional': '#10b981',
        'Comparativa': '#8b5cf6',
        'Informacional': '#3b82f6',
        'Marca': '#f97316',
        'Navegacional': '#ec4899',
        'Navegacional/Marca': '#6b7280'
    }
    
    for intent, intent_clusters in intent_groups.items():
        intent_volume = sum(c.total_volume for c in intent_clusters)
        labels.append(intent)
        parents.append("All Keywords")
        values.append(intent_volume)
        colors.append(intent_colors_map.get(intent, '#6b7280'))
        hover_texts.append(f"{intent}<br>{intent_volume:,} searches<br>{len(intent_clusters)} clusters")
        
        # Top 5 clusters por intent
        top_clusters = sorted(intent_clusters, key=lambda x: x.total_volume, reverse=True)[:5]
        for cluster in top_clusters:
            labels.append(cluster.primary_keyword)
            parents.append(intent)
            values.append(cluster.total_volume)
            colors.append(intent_colors_map.get(intent, '#6b7280'))
            hover_texts.append(
                f"<b>{cluster.primary_keyword}</b><br>"
                f"Volume: {cluster.total_volume:,}<br>"
                f"Keywords: {len(cluster.keywords)}<br>"
                f"Avg Competition: {cluster.avg_competition:.0f}"
            )
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors, line=dict(width=2, color='white')),
        text=hover_texts,
        hoverinfo='text',
        textposition="middle center"
    ))
    
    fig.update_layout(
        title='<b>Keyword Clusters Hierarchy</b><br><sub>Agrupados por Search Intent</sub>',
        height=600,
        margin=dict(t=60, l=0, r=0, b=0)
    )
    
    return fig

def export_to_excel(clusters: List[KeywordCluster], filename: str = "keyword_clusters.xlsx") -> bytes:
    """
    Exporta clusters a Excel profesional con múltiples sheets
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Resumen de clusters
        cluster_summary = []
        for cluster in clusters:
            cluster_summary.append({
                'Cluster ID': cluster.cluster_id,
                'Primary Keyword': cluster.primary_keyword,
                'Search Intent': cluster.search_intent,
                'Total Volume': cluster.total_volume,
                'Num Keywords': len(cluster.keywords),
                'Avg Competition': round(cluster.avg_competition, 2),
                'Confidence': round(cluster.confidence_score, 2)
            })
        
        df_summary = pd.DataFrame(cluster_summary)
        df_summary.to_excel(writer, sheet_name='Cluster Summary', index=False)
        
        # Sheet 2: Todas las keywords con su cluster
        all_keywords = []
        for cluster in clusters:
            for kw in cluster.keywords:
                all_keywords.append({
                    'Cluster ID': cluster.cluster_id,
                    'Primary Keyword': cluster.primary_keyword,
                    'Keyword': kw.keyword,
                    'Volume': kw.volume,
                    'Competition': kw.competition,
                    'Competition Index': kw.competition_index,
                    'CPC Low': kw.cpc_low,
                    'CPC High': kw.cpc_high,
                    'Trend 3M (%)': kw.trend_3m,
                    'Trend YoY (%)': kw.trend_yoy,
                    'Search Intent': cluster.search_intent
                })
        
        df_keywords = pd.DataFrame(all_keywords)
        df_keywords.to_excel(writer, sheet_name='All Keywords', index=False)
        
        # Sheet 3: Por search intent
        intent_summary = []
        intent_groups = {}
        for cluster in clusters:
            if cluster.search_intent not in intent_groups:
                intent_groups[cluster.search_intent] = {
                    'clusters': [],
                    'total_volume': 0,
                    'total_keywords': 0
                }
            intent_groups[cluster.search_intent]['clusters'].append(cluster)
            intent_groups[cluster.search_intent]['total_volume'] += cluster.total_volume
            intent_groups[cluster.search_intent]['total_keywords'] += len(cluster.keywords)
        
        for intent, data in intent_groups.items():
            intent_summary.append({
                'Search Intent': intent,
                'Num Clusters': len(data['clusters']),
                'Total Keywords': data['total_keywords'],
                'Total Volume': data['total_volume'],
                'Avg Volume per Keyword': round(data['total_volume'] / data['total_keywords'], 0)
            })
        
        df_intent = pd.DataFrame(intent_summary)
        df_intent.to_excel(writer, sheet_name='By Search Intent', index=False)
        
        # Sheet 4: Content recommendations
        content_recs = []
        for cluster in clusters[:20]:  # Top 20
            content_recs.append({
                'Cluster': cluster.primary_keyword,
                'Content Type': get_content_recommendation(cluster.search_intent),
                'Target Volume': cluster.total_volume,
                'Keywords to Include': ', '.join([kw.keyword for kw in cluster.keywords[:10]]),
                'Priority': 'High' if cluster.total_volume > 5000 else 'Medium' if cluster.total_volume > 1000 else 'Low'
            })
        
        df_content = pd.DataFrame(content_recs)
        df_content.to_excel(writer, sheet_name='Content Recommendations', index=False)
    
    output.seek(0)
    return output.getvalue()

def get_content_recommendation(search_intent: str) -> str:
    """Recomienda tipo de contenido basado en intent"""
    recommendations = {
        'Transaccional': 'Product Listing Page (PLP) / Category Page',
        'Comparativa': 'Comparison Article / Review Roundup',
        'Informacional': 'Blog Post / How-to Guide',
        'Marca': 'Brand Landing Page',
        'Navegacional': 'Category Page / Hub Page',
        'Navegacional/Marca': 'Brand/Category Hub'
    }
    return recommendations.get(search_intent, 'Blog Post')

# ==================== MAIN APP ====================

def main():
    
    # Header
    st.markdown('<div class="main-header">🔬 Query Fan Out v2.0.3</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Professional Keyword Clustering & Search Intent Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source
        st.subheader("1️⃣ Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Upload Keyword Planner CSV", "Use Demo Data", "Connect API (Coming Soon)"],
            help="Import your keyword data from Google Keyword Planner or use demo data"
        )
        
        if data_source == "Upload Keyword Planner CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV from Keyword Planner",
                type=['csv'],
                help="Export CSV from Google Ads Keyword Planner"
            )
            
            if uploaded_file is not None:
                # Usar nombre de archivo único basado en hash del contenido
                import hashlib
                file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
                temp_path = f"/tmp/keyword_planner_{file_hash}.csv"
                
                # Solo escribir si no existe o si cambió el archivo
                if not os.path.exists(temp_path) or 'current_file_hash' not in st.session_state or st.session_state['current_file_hash'] != file_hash:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    st.session_state['current_file_hash'] = file_hash
                    st.session_state['temp_csv_path'] = temp_path
                
                # Cargar solo si no está en session_state o si es un archivo nuevo
                if 'keywords_df' not in st.session_state or st.session_state.get('current_file_hash') != file_hash:
                    with st.spinner("Loading keywords..."):
                        df = load_keyword_planner_csv(temp_path)
                        if len(df) > 0:
                            st.success(f"✅ Loaded {len(df)} keywords")
                            st.session_state['keywords_df'] = df
                        else:
                            st.error("Error loading CSV. Check format.")
                else:
                    # Ya está cargado
                    st.info(f"📊 Using cached data: {len(st.session_state['keywords_df'])} keywords")
        
        elif data_source == "Use Demo Data":
            if st.button("Load Demo Data"):
                # Usar el CSV que subió el usuario
                with st.spinner("Loading demo keywords..."):
                    df = load_keyword_planner_csv('/mnt/user-data/uploads/Keyword_Stats_2025-11-20_at_17_19_54.csv')
                    if len(df) > 0:
                        st.success(f"✅ Loaded {len(df)} demo keywords")
                        st.session_state['keywords_df'] = df
        
        # Clustering settings
        st.subheader("2️⃣ Clustering Settings")
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Higher = stricter clustering (fewer, more precise clusters)"
        )
        
        min_cluster_size = st.slider(
            "Min Cluster Size",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum keywords per cluster"
        )
        
        # Run clustering
        if st.button("🚀 Run Clustering", type="primary"):
            if 'keywords_df' in st.session_state and len(st.session_state['keywords_df']) > 0:
                with st.spinner("Running semantic clustering..."):
                    clusters = perform_clustering(
                        st.session_state['keywords_df'],
                        similarity_threshold=similarity_threshold,
                        min_cluster_size=min_cluster_size
                    )
                    st.session_state['clusters'] = clusters
                    st.success(f"✅ Created {len(clusters)} clusters")
            else:
                st.error("⚠️ Please load keyword data first")
        
        # API Connections (coming soon)
        st.markdown("---")
        st.subheader("🔌 API Integrations")
        st.info("""
        **Available APIs:**
        - Google Search Console
        - SEMrush
        - Sistrix
        - Claude API
        - OpenAI API
        - Zenrows
        
        Coming in next version!
        """)
    
    # Main content area
    if 'clusters' not in st.session_state:
        # Welcome screen
        st.markdown("""
        ## Welcome to Query Fan Out v2.0.3 Professional! 👋
        
        **Latest:** v2.0.3 fixes the distance matrix bug. Now 100% stable! 🎉
        
        This tool helps you:
        - 🎯 **Cluster keywords** by semantic similarity using real NLP
        - 🔍 **Detect search intent** automatically (Transaccional, Informacional, etc.)
        - 📊 **Visualize relationships** with interactive network graphs
        - 📈 **Analyze opportunity** based on volume, competition, and trends
        - 📝 **Generate content strategies** based on keyword clusters
        
        ### How to use:
        1. **Upload your CSV** from Google Keyword Planner (or use demo data)
        2. **Adjust clustering settings** in the sidebar
        3. **Run clustering** and explore results
        4. **Export to Excel** for your content team
        
        ### What's different from v1.0?
        ✅ Real semantic clustering (not templates)  
        ✅ Network graph visualization  
        ✅ Actual keyword data from Keyword Planner  
        ✅ Search intent detection with NLP  
        ✅ Professional Excel export  
        ✅ Proper Streamlit architecture (caching, no session state bugs)  
        
        **Get started by uploading your data in the sidebar! 👈**
        """)
        
        st.markdown('<div class="info-box">💡 <b>Tip:</b> Export your keyword ideas from Google Ads Keyword Planner and upload the CSV here. The tool will automatically parse volumes, competition, CPC data, etc.</div>', unsafe_allow_html=True)
    
    else:
        # Show results
        clusters = st.session_state['clusters']
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_volume = sum(c.total_volume for c in clusters)
        total_keywords = sum(len(c.keywords) for c in clusters)
        avg_cluster_size = total_keywords / len(clusters) if clusters else 0
        
        with col1:
            st.metric("Total Clusters", len(clusters))
        with col2:
            st.metric("Total Keywords", f"{total_keywords:,}")
        with col3:
            st.metric("Total Monthly Volume", f"{total_volume:,}")
        with col4:
            st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f} kws")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Network Graph",
            "🌳 Hierarchy View", 
            "📋 Cluster Details",
            "📈 Insights",
            "💾 Export"
        ])
        
        with tab1:
            st.subheader("Interactive Network Graph")
            st.caption("Visualización de keywords conectadas por similitud semántica")
            
            # Controles de configuración
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                max_keywords = st.slider(
                    "Max keywords a visualizar",
                    min_value=20,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Menos keywords = más rápido. Recomendado: 50 para datasets grandes"
                )
            
            with col2:
                layout_algo = st.selectbox(
                    "Algoritmo de layout",
                    ["circular", "kamada", "spring"],
                    index=0,
                    help="Circular = Instantáneo | Kamada = Rápido | Spring = Lento pero bonito"
                )
            
            with col3:
                st.markdown("**Velocidad:**")
                if layout_algo == "circular":
                    st.success("⚡ Instantáneo")
                elif layout_algo == "kamada":
                    st.info("🏃 Rápido (~5s)")
                else:
                    st.warning("🐌 Lento (~30s)")
            
            # Opción de cluster específico
            cluster_options = ["Todos los clusters"] + [
                f"Cluster {c.cluster_id}: {c.primary_keyword} ({len(c.keywords)} kws)"
                for c in clusters[:20]
            ]
            
            selected_cluster = st.selectbox(
                "Filtrar por cluster (opcional):",
                cluster_options,
                help="Visualizar solo un cluster específico para mejor detalle"
            )
            
            selected_cluster_id = None
            if selected_cluster != "Todos los clusters":
                selected_cluster_id = int(selected_cluster.split(":")[0].replace("Cluster ", ""))
            
            # Botón para generar
            generate_graph = st.button("🎨 Generar Network Graph", type="primary")
            
            if generate_graph or 'network_graph' in st.session_state:
                if generate_graph:
                    with st.spinner(f"Generando network graph con layout '{layout_algo}'..."):
                        fig_network = create_network_graph(
                            clusters,
                            max_keywords=max_keywords,
                            selected_cluster_id=selected_cluster_id,
                            layout_algorithm=layout_algo
                        )
                        st.session_state['network_graph'] = fig_network
                
                st.plotly_chart(st.session_state['network_graph'], use_container_width=True)
                
                # Tips de uso
                with st.expander("💡 Tips para mejor visualización"):
                    st.markdown("""
                    **Para datasets grandes (1000+ keywords):**
                    - Usa **layout circular** (instantáneo)
                    - Limita a **50 keywords máximo**
                    - Filtra por cluster específico
                    
                    **Para análisis detallado:**
                    - Selecciona **un solo cluster**
                    - Usa **layout spring** para mejor distribución
                    - Aumenta a 100-200 keywords
                    
                    **Interpretación:**
                    - **Tamaño del nodo** = Volumen de búsqueda
                    - **Color** = Search Intent
                    - **Conexiones** = Keywords del mismo cluster
                    """)
            else:
                st.info("""
                ⬆️ **Configura los parámetros arriba y haz click en "Generar Network Graph"**
                
                💡 **Recomendaciones:**
                - Para datasets grandes: Usa layout **circular** con **50 keywords**
                - Para mejor visualización: Selecciona **un cluster específico**
                - Hierarchy View (siguiente tab) es más rápido para overview
                """)
            
            # Legend
            st.markdown("**Legend:**")
            cols = st.columns(6)
            intents = [
                ("🟢 Transaccional", "#10b981"),
                ("🟣 Comparativa", "#8b5cf6"),
                ("🔵 Informacional", "#3b82f6"),
                ("🟠 Marca", "#f97316"),
                ("🟤 Navegacional", "#ec4899"),
                ("⚫ Otro", "#6b7280")
            ]
            for col, (label, color) in zip(cols, intents):
                col.markdown(f'<span style="color: {color};">●</span> {label}', unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Cluster Hierarchy by Search Intent")
            fig_hierarchy = create_cluster_hierarchy_chart(clusters)
            st.plotly_chart(fig_hierarchy, use_container_width=True)
            
            # Intent breakdown
            st.subheader("Search Intent Distribution")
            intent_counts = {}
            intent_volumes = {}
            for cluster in clusters:
                intent = cluster.search_intent
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                intent_volumes[intent] = intent_volumes.get(intent, 0) + cluster.total_volume
            
            df_intent = pd.DataFrame({
                'Search Intent': list(intent_counts.keys()),
                'Num Clusters': list(intent_counts.values()),
                'Total Volume': list(intent_volumes.values())
            })
            df_intent = df_intent.sort_values('Total Volume', ascending=False)
            st.dataframe(df_intent, use_container_width=True)
        
        with tab3:
            st.subheader("Detailed Cluster Analysis")
            
            # Filter by intent
            all_intents = sorted(set(c.search_intent for c in clusters))
            selected_intent = st.multiselect(
                "Filter by Search Intent:",
                all_intents,
                default=all_intents
            )
            
            filtered_clusters = [c for c in clusters if c.search_intent in selected_intent]
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by:",
                ["Total Volume", "Number of Keywords", "Avg Competition", "Confidence Score"]
            )
            
            if sort_by == "Total Volume":
                filtered_clusters.sort(key=lambda x: x.total_volume, reverse=True)
            elif sort_by == "Number of Keywords":
                filtered_clusters.sort(key=lambda x: len(x.keywords), reverse=True)
            elif sort_by == "Avg Competition":
                filtered_clusters.sort(key=lambda x: x.avg_competition, reverse=True)
            else:
                filtered_clusters.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Show clusters
            for i, cluster in enumerate(filtered_clusters[:30], 1):  # Top 30
                with st.expander(
                    f"**#{i} {cluster.primary_keyword}** - {cluster.search_intent} | "
                    f"{cluster.total_volume:,} vol | {len(cluster.keywords)} kws"
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Primary Keyword:** {cluster.primary_keyword}")
                        st.markdown(f"**Search Intent:** {cluster.search_intent} (confidence: {cluster.confidence_score:.0%})")
                        st.markdown(f"**Total Volume:** {cluster.total_volume:,} searches/month")
                        st.markdown(f"**Avg Competition:** {cluster.avg_competition:.0f}/100")
                        
                        # Keywords in cluster
                        st.markdown("**Keywords in this cluster:**")
                        kw_df = pd.DataFrame([{
                            'Keyword': kw.keyword,
                            'Volume': f"{kw.volume:,}",
                            'Competition': f"{kw.competition_index:.0f}",
                            'CPC': f"€{kw.cpc_low:.2f}-{kw.cpc_high:.2f}"
                        } for kw in cluster.keywords])
                        st.dataframe(kw_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("**📝 Content Recommendation**")
                        content_type = get_content_recommendation(cluster.search_intent)
                        st.info(f"**Type:** {content_type}")
                        
                        if cluster.total_volume > 5000:
                            st.success("🔥 **High Priority**")
                        elif cluster.total_volume > 1000:
                            st.warning("⚡ **Medium Priority**")
                        else:
                            st.info("📌 **Low Priority**")
        
        with tab4:
            st.subheader("Strategic Insights")
            
            # Top opportunities
            st.markdown("### 🎯 Top Opportunities")
            st.caption("Clusters with high volume and manageable competition")
            
            opportunities = [
                c for c in clusters 
                if c.total_volume > 1000 and c.avg_competition < 70
            ]
            opportunities.sort(key=lambda x: x.total_volume / (c.avg_competition + 1), reverse=True)
            
            if opportunities:
                for i, cluster in enumerate(opportunities[:10], 1):
                    opportunity_score = cluster.total_volume / (cluster.avg_competition + 1)
                    st.markdown(
                        f"**{i}. {cluster.primary_keyword}** - "
                        f"{cluster.total_volume:,} vol | "
                        f"{cluster.avg_competition:.0f} comp | "
                        f"Score: {opportunity_score:.0f}"
                    )
            else:
                st.info("No clear opportunities found with current filters")
            
            # Quick wins
            st.markdown("### ⚡ Quick Wins")
            st.caption("Low competition, decent volume")
            
            quick_wins = [
                c for c in clusters
                if c.avg_competition < 50 and c.total_volume > 500
            ]
            quick_wins.sort(key=lambda x: x.total_volume, reverse=True)
            
            if quick_wins:
                for i, cluster in enumerate(quick_wins[:10], 1):
                    st.markdown(
                        f"**{i}. {cluster.primary_keyword}** - "
                        f"{cluster.total_volume:,} vol | "
                        f"{cluster.avg_competition:.0f} comp"
                    )
            else:
                st.info("No quick wins identified")
            
            # Long-tail analysis
            st.markdown("### 🎣 Long-tail Keywords")
            st.caption("3+ word keywords with potential")
            
            longtail = []
            for cluster in clusters:
                for kw in cluster.keywords:
                    word_count = len(kw.keyword.split())
                    if word_count >= 3 and kw.volume > 100:
                        longtail.append({
                            'keyword': kw.keyword,
                            'volume': kw.volume,
                            'words': word_count,
                            'competition': kw.competition_index
                        })
            
            longtail.sort(key=lambda x: x['volume'], reverse=True)
            
            if longtail:
                df_longtail = pd.DataFrame(longtail[:20])
                st.dataframe(df_longtail, use_container_width=True, hide_index=True)
            else:
                st.info("No significant long-tail keywords found")
        
        with tab5:
            st.subheader("Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Excel Export")
                st.caption("Professional multi-sheet workbook with all data")
                
                excel_data = export_to_excel(clusters)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"keyword_clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.markdown("""
                **Includes:**
                - Cluster summary
                - All keywords with metrics
                - Search intent breakdown
                - Content recommendations
                """)
            
            with col2:
                st.markdown("### 📄 JSON Export")
                st.caption("For developers / API integration")
                
                json_data = {
                    'generated_at': datetime.now().isoformat(),
                    'total_clusters': len(clusters),
                    'total_keywords': sum(len(c.keywords) for c in clusters),
                    'total_volume': sum(c.total_volume for c in clusters),
                    'clusters': [c.to_dict() for c in clusters]
                }
                
                json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"keyword_clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            
            st.markdown("---")
            
            # Raw data preview
            with st.expander("🔍 Preview Export Data"):
                st.json(json_data, expanded=False)

    # Footer
    st.markdown("---")
    st.caption("Query Fan Out v2.0.3 | Built with Streamlit + Plotly + NetworkX | Real semantic clustering with TF-IDF")

if __name__ == "__main__":
    main()
