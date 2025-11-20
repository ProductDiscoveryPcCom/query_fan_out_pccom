"""
API Integration Examples for Query Fan Out v2.0

Este archivo contiene ejemplos de cómo integrar las APIs disponibles:
- Google Search Console
- SEMrush
- Sistrix
- Claude API
- OpenAI API
- Zenrows

Copia estas funciones a app_v2.py cuando quieras activar las integraciones.
"""

import requests
import pandas as pd
from typing import List, Dict
import anthropic
import openai

# ==================== GOOGLE SEARCH CONSOLE ====================

def get_gsc_keywords(
    site_url: str,
    api_key: str,
    start_date: str = '2024-10-01',
    end_date: str = '2024-11-20',
    row_limit: int = 5000
) -> pd.DataFrame:
    """
    Obtiene keywords de Google Search Console
    
    Args:
        site_url: URL del sitio (ej: 'https://www.pccomponentes.com')
        api_key: API key de Google Cloud
        start_date: Fecha inicio (YYYY-MM-DD)
        end_date: Fecha fin (YYYY-MM-DD)
        row_limit: Máximo resultados
    
    Returns:
        DataFrame con columnas: keyword, clicks, impressions, ctr, position
    """
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    
    # Configurar credenciales
    # TODO: Implementar OAuth2 flow real
    creds = Credentials(token=api_key)
    service = build('searchconsole', 'v1', credentials=creds)
    
    # Request
    request = {
        'startDate': start_date,
        'endDate': end_date,
        'dimensions': ['query'],
        'rowLimit': row_limit,
        'startRow': 0
    }
    
    response = service.searchanalytics().query(
        siteUrl=site_url,
        body=request
    ).execute()
    
    # Procesar resultados
    if 'rows' in response:
        data = []
        for row in response['rows']:
            data.append({
                'keyword': row['keys'][0],
                'clicks': row['clicks'],
                'impressions': row['impressions'],
                'ctr': row['ctr'],
                'position': row['position']
            })
        
        return pd.DataFrame(data)
    
    return pd.DataFrame()


# ==================== SEMRUSH API ====================

def get_semrush_keywords(
    domain: str,
    api_key: str,
    database: str = 'es',
    limit: int = 5000
) -> pd.DataFrame:
    """
    Obtiene keywords de SEMrush Organic Research
    
    Args:
        domain: Dominio (ej: 'pccomponentes.com')
        api_key: API key de SEMrush
        database: Base de datos (es, us, uk, etc.)
        limit: Máximo resultados
    
    Returns:
        DataFrame con keywords, volume, CPC, competition, etc.
    """
    url = "https://api.semrush.com/"
    
    params = {
        'type': 'domain_organic',
        'key': api_key,
        'display_limit': limit,
        'export_columns': 'Ph,Po,Nq,Cp,Co,Kd,Tr,Tc,Nr',
        'domain': domain,
        'database': database
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text), sep=';')
        
        # Renombrar columnas
        df.columns = [
            'keyword',
            'position',
            'volume',
            'cpc',
            'competition',
            'difficulty',
            'traffic',
            'traffic_cost',
            'results'
        ]
        
        return df
    
    return pd.DataFrame()


def get_semrush_keyword_overview(
    keyword: str,
    api_key: str,
    database: str = 'es'
) -> Dict:
    """
    Obtiene overview de una keyword específica
    
    Returns:
        Dict con volume, CPC, competition, trend, etc.
    """
    url = "https://api.semrush.com/"
    
    params = {
        'type': 'phrase_all',
        'key': api_key,
        'phrase': keyword,
        'database': database,
        'export_columns': 'Ph,Nq,Cp,Co,Kd'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.text.split('\n')[1].split(';')  # Skip header
        
        return {
            'keyword': data[0],
            'volume': int(data[1]) if data[1] else 0,
            'cpc': float(data[2]) if data[2] else 0.0,
            'competition': float(data[3]) if data[3] else 0.0,
            'difficulty': int(data[4]) if data[4] else 0
        }
    
    return {}


# ==================== SISTRIX API ====================

def get_sistrix_keywords(
    domain: str,
    api_key: str,
    country: str = 'es',
    limit: int = 100
) -> pd.DataFrame:
    """
    Obtiene keywords de Sistrix
    
    Args:
        domain: Dominio
        api_key: API key de Sistrix
        country: País (es, us, uk, etc.)
        limit: Máximo resultados
    
    Returns:
        DataFrame con keywords y posiciones
    """
    url = f"https://api.sistrix.com/domain.keywords"
    
    params = {
        'api_key': api_key,
        'domain': domain,
        'country': country,
        'num': limit,
        'format': 'json'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        keywords = []
        for item in data.get('answer', []):
            keywords.append({
                'keyword': item.get('keyword'),
                'position': item.get('position'),
                'url': item.get('url'),
                'competition': item.get('competition')
            })
        
        return pd.DataFrame(keywords)
    
    return pd.DataFrame()


# ==================== ZENROWS (SERP SCRAPING) ====================

def scrape_serp_with_zenrows(
    keyword: str,
    api_key: str,
    country: str = 'es',
    language: str = 'es',
    num_results: int = 100
) -> List[Dict]:
    """
    Scraping de SERP usando Zenrows
    
    Returns:
        Lista de dicts con: title, url, description, position
    """
    url = "https://api.zenrows.com/v1/"
    
    # Build Google search URL
    search_url = f"https://www.google.com/search?q={keyword}&gl={country}&hl={language}&num={num_results}"
    
    params = {
        'apikey': api_key,
        'url': search_url,
        'js_render': 'true',
        'premium_proxy': 'true'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for i, result in enumerate(soup.select('div.g'), 1):
            title_elem = result.select_one('h3')
            link_elem = result.select_one('a')
            desc_elem = result.select_one('div[style*="-webkit-line-clamp"]')
            
            if title_elem and link_elem:
                results.append({
                    'position': i,
                    'title': title_elem.get_text(),
                    'url': link_elem.get('href'),
                    'description': desc_elem.get_text() if desc_elem else ''
                })
        
        return results
    
    return []


def analyze_serp_similarity(
    keywords: List[str],
    api_key: str
) -> Dict[str, List[str]]:
    """
    Analiza similitud SERP entre keywords (base de clustering SERP-based)
    
    Returns:
        Dict mapping keyword -> list of similar keywords (based on SERP overlap)
    """
    serp_data = {}
    
    # Scrape SERPs para cada keyword
    for keyword in keywords:
        results = scrape_serp_with_zenrows(keyword, api_key, num_results=10)
        # Extraer solo URLs
        serp_data[keyword] = [r['url'] for r in results]
    
    # Calcular overlap
    similarity_map = {}
    for kw1 in keywords:
        similar = []
        urls1 = set(serp_data[kw1])
        
        for kw2 in keywords:
            if kw1 != kw2:
                urls2 = set(serp_data[kw2])
                # Jaccard similarity
                overlap = len(urls1 & urls2) / len(urls1 | urls2)
                
                if overlap > 0.5:  # >50% overlap
                    similar.append(kw2)
        
        similarity_map[kw1] = similar
    
    return similarity_map


# ==================== CLAUDE API ====================

def generate_content_brief_with_claude(
    cluster: 'KeywordCluster',
    api_key: str
) -> Dict:
    """
    Genera content brief usando Claude API
    
    Args:
        cluster: KeywordCluster object
        api_key: Anthropic API key
    
    Returns:
        Dict con: title, outline, target_words, key_points, ctas
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # Preparar keywords
    keywords_str = ', '.join([kw.keyword for kw in cluster.keywords[:10]])
    
    prompt = f"""Eres un experto en SEO y content marketing para PcComponentes.

Cluster de keywords: {cluster.primary_keyword}
Search intent: {cluster.search_intent}
Volumen total: {cluster.total_volume} búsquedas/mes
Keywords relacionadas: {keywords_str}

Genera un content brief profesional con:

1. Título SEO optimizado (max 60 caracteres)
2. Meta description (max 155 caracteres)
3. Estructura del contenido (H2s y H3s)
4. Puntos clave a cubrir
5. Target word count
6. CTAs recomendados

Responde en formato JSON."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse response (asumiendo JSON)
    import json
    brief = json.loads(message.content[0].text)
    
    return brief


# ==================== OPENAI API ====================

def analyze_search_intent_with_openai(
    keywords: List[str],
    api_key: str
) -> Dict[str, str]:
    """
    Analiza search intent usando GPT-4
    
    Returns:
        Dict mapping keyword -> intent
    """
    openai.api_key = api_key
    
    keywords_str = '\n'.join([f"- {kw}" for kw in keywords])
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Eres un experto en SEO. Clasifica keywords por search intent: Transaccional, Informacional, Comparativa, Navegacional, o Marca."
            },
            {
                "role": "user",
                "content": f"Clasifica estas keywords:\n{keywords_str}\n\nResponde en formato: keyword | intent"
            }
        ],
        temperature=0.3
    )
    
    # Parse response
    intent_map = {}
    lines = response.choices[0].message.content.split('\n')
    
    for line in lines:
        if '|' in line:
            parts = line.split('|')
            if len(parts) == 2:
                keyword = parts[0].strip().replace('- ', '')
                intent = parts[1].strip()
                intent_map[keyword] = intent
    
    return intent_map


# ==================== INTEGRATION HELPERS ====================

def enrich_clusters_with_gsc(
    clusters: List['KeywordCluster'],
    gsc_data: pd.DataFrame
) -> List['KeywordCluster']:
    """
    Enriquece clusters con datos de GSC (clicks, impressions, CTR, position)
    """
    gsc_dict = gsc_data.set_index('keyword').to_dict('index')
    
    for cluster in clusters:
        for kw in cluster.keywords:
            if kw.keyword in gsc_dict:
                gsc_info = gsc_dict[kw.keyword]
                # Añadir datos GSC al keyword
                kw.clicks = gsc_info.get('clicks', 0)
                kw.impressions = gsc_info.get('impressions', 0)
                kw.ctr = gsc_info.get('ctr', 0.0)
                kw.position = gsc_info.get('position', 0)
    
    return clusters


def enrich_clusters_with_semrush(
    clusters: List['KeywordCluster'],
    semrush_data: pd.DataFrame
) -> List['KeywordCluster']:
    """
    Enriquece clusters con datos de SEMrush (difficulty, traffic, etc.)
    """
    semrush_dict = semrush_data.set_index('keyword').to_dict('index')
    
    for cluster in clusters:
        for kw in cluster.keywords:
            if kw.keyword in semrush_dict:
                semrush_info = semrush_dict[kw.keyword]
                kw.difficulty = semrush_info.get('difficulty', 0)
                kw.traffic = semrush_info.get('traffic', 0)
    
    return clusters


# ==================== EXAMPLE USAGE ====================

def example_full_pipeline():
    """
    Ejemplo de pipeline completo con todas las APIs
    """
    
    # 1. Cargar keywords de Keyword Planner
    # keywords_df = load_keyword_planner_csv('your_file.csv')
    
    # 2. Enriquecer con Google Search Console
    # gsc_data = get_gsc_keywords(
    #     site_url='https://www.pccomponentes.com',
    #     api_key='YOUR_GSC_API_KEY'
    # )
    
    # 3. Enriquecer con SEMrush
    # semrush_data = get_semrush_keywords(
    #     domain='pccomponentes.com',
    #     api_key='YOUR_SEMRUSH_API_KEY'
    # )
    
    # 4. Clustering semántico
    # clusters = perform_clustering(keywords_df)
    
    # 5. Enriquecer clusters
    # clusters = enrich_clusters_with_gsc(clusters, gsc_data)
    # clusters = enrich_clusters_with_semrush(clusters, semrush_data)
    
    # 6. Análisis SERP (para keywords prioritarias)
    # top_keywords = [c.primary_keyword for c in clusters[:20]]
    # serp_similarity = analyze_serp_similarity(
    #     keywords=top_keywords,
    #     api_key='YOUR_ZENROWS_API_KEY'
    # )
    
    # 7. Generar content briefs con Claude
    # for cluster in clusters[:10]:
    #     brief = generate_content_brief_with_claude(
    #         cluster=cluster,
    #         api_key='YOUR_ANTHROPIC_API_KEY'
    #     )
    #     print(f"Brief for {cluster.primary_keyword}:")
    #     print(brief)
    
    pass


if __name__ == "__main__":
    print("API Integration Examples")
    print("Copy these functions to app_v2.py when ready to use")
    print("\nAvailable integrations:")
    print("- Google Search Console")
    print("- SEMrush")
    print("- Sistrix")
    print("- Zenrows (SERP scraping)")
    print("- Claude API")
    print("- OpenAI API")
