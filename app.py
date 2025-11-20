import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict
import json

# Configuración de la página
st.set_page_config(
    page_title="Query Fan Out Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .category-card {
        border-left: 5px solid;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class Category:
    name: str
    color: str
    icon: str
    queries: List[str]
    volume: str
    content_type: str

class QueryFanOutGenerator:
    def __init__(self, main_keyword: str, volume: int):
        self.main_keyword = main_keyword
        self.volume = volume
        self.categories = self._generate_categories()
    
    def _generate_categories(self) -> List[Category]:
        """Genera categorías basadas en la keyword principal"""
        base_word = self.main_keyword.replace("mejor ", "").replace("mejores ", "")
        
        return [
            Category(
                name="Informacional",
                color="#3b82f6",
                icon="📚",
                queries=[
                    f"qué {base_word} comprar",
                    f"cómo elegir {base_word}",
                    f"guía de compra {base_word}",
                    f"ventajas {base_word}",
                    f"características {base_word}",
                ],
                volume="Media-Alta",
                content_type="Blog Post / Guía"
            ),
            Category(
                name="Comparativa",
                color="#8b5cf6",
                icon="📊",
                queries=[
                    f"{self.main_keyword} calidad precio",
                    f"comparativa {base_word}",
                    f"top {base_word} 2025",
                    f"{self.main_keyword} vs alternativas",
                    f"ranking {base_word}",
                ],
                volume="Alta",
                content_type="Artículo Comparativo"
            ),
            Category(
                name="Transaccional",
                color="#10b981",
                icon="💰",
                queries=[
                    f"comprar {base_word}",
                    f"{base_word} oferta",
                    f"{base_word} precio",
                    f"{base_word} barato",
                    f"{base_word} descuento",
                ],
                volume="Muy Alta",
                content_type="PLP / Categoría"
            ),
            Category(
                name="De Marca",
                color="#f97316",
                icon="🎯",
                queries=[
                    f"{self.main_keyword} [marca 1]",
                    f"{self.main_keyword} [marca 2]",
                    f"{self.main_keyword} [marca 3]",
                    f"qué marca de {base_word}",
                    f"comparar marcas {base_word}",
                ],
                volume="Media",
                content_type="Landing por Marca"
            ),
            Category(
                name="Long Tail",
                color="#ec4899",
                icon="👥",
                queries=[
                    f"{base_word} para [uso específico 1]",
                    f"{base_word} con [característica 1]",
                    f"{base_word} [adjetivo] y [adjetivo]",
                    f"{self.main_keyword} según [criterio]",
                    f"{base_word} recomendado para [situación]",
                ],
                volume="Baja-Media",
                content_type="Landing Específica"
            ),
        ]
    
    def generate_sunburst_chart(self):
        """Genera un gráfico sunburst interactivo"""
        labels = [self.main_keyword]
        parents = [""]
        values = [self.volume]
        colors = ["#ef4444"]
        
        for cat in self.categories:
            # Añadir categoría
            labels.append(cat.name)
            parents.append(self.main_keyword)
            values.append(len(cat.queries) * 100)
            colors.append(cat.color)
            
            # Añadir queries
            for query in cat.queries:
                labels.append(query)
                parents.append(cat.name)
                values.append(50)
                colors.append(cat.color)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Queries: %{value}<extra></extra>',
        ))
        
        fig.update_layout(
            height=700,
            margin=dict(t=0, l=0, r=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        return fig
    
    def generate_content_proposals(self) -> Dict[str, Dict]:
        """Genera propuestas de contenido para cada categoría"""
        proposals = {}
        
        proposals["informacional"] = {
            "titulo": f"Guía Completa 2025: Cómo elegir {self.main_keyword}",
            "tipo": "Blog Post",
            "objetivo": "Educar al usuario sobre criterios de selección y casos de uso",
            "estructura": [
                "Introducción: contexto y tendencias 2025",
                "10 criterios clave de selección",
                "Tecnologías y características principales",
                "Casos de uso por situación",
                "Errores comunes al comprar",
                "FAQ con dudas frecuentes"
            ],
            "cta": f"Explora nuestra selección de {self.main_keyword.replace('mejor ', '')} →"
        }
        
        proposals["comparativa"] = {
            "titulo": f"Los 10 {self.main_keyword.title()} de 2025: Análisis y Comparativa",
            "tipo": "Artículo Comparativo",
            "objetivo": "Presentar los mejores modelos con análisis objetivo",
            "estructura": [
                "Metodología de selección transparente",
                "Tabla comparativa interactiva",
                "Top 10 con análisis individual",
                "Pros y contras sin filtros",
                "Mejor por categoría de precio",
                "Veredicto final"
            ],
            "cta": "Ver precio y disponibilidad de cada modelo →"
        }
        
        proposals["transaccional"] = {
            "titulo": f"{self.main_keyword.title()} - Comprar Online",
            "tipo": "Página de Categoría (PLP)",
            "objetivo": "Optimizar para conversión directa",
            "estructura": [
                "Filtros inteligentes (precio, marca, características)",
                "Señales de confianza (stock, envío, garantía)",
                "Promociones destacadas",
                "Prefooter SEO optimizado",
                "Schema markup completo"
            ],
            "cta": "Añadir al carrito"
        }
        
        proposals["marca"] = {
            "titulo": f"{self.main_keyword.title()} [Marca X] - Comparativa",
            "tipo": "Landing por Marca",
            "objetivo": "Ayudar a elegir entre modelos de la misma marca",
            "estructura": [
                "Gama completa de la marca",
                "Comparativa entre modelos",
                "Mejor modelo según necesidad",
                "Comparación vs competencia",
                "Grid de productos disponibles"
            ],
            "cta": "Ver todos los modelos de [Marca] →"
        }
        
        proposals["longtail"] = {
            "titulo": f"Los 7 {self.main_keyword.title()} [característica específica]",
            "tipo": "Landing Específica",
            "objetivo": "Captar búsquedas ultra-específicas",
            "estructura": [
                "Qué es la característica específica",
                "Ventajas de esta característica",
                "Top 7 con esta característica",
                "Comparativa técnica",
                "Vale la pena la inversión?"
            ],
            "cta": "Ver oferta especial →"
        }
        
        return proposals

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Query Fan Out Generator</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem;'>Genera estrategias de contenido SEO basadas en intención de búsqueda</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        main_keyword = st.text_input(
            "Palabra clave principal",
            value="mejor robot aspirador",
            help="Ingresa tu keyword principal (ej: mejor smartphone, mejores auriculares)"
        )
        
        volume = st.number_input(
            "Volumen de búsqueda mensual",
            min_value=100,
            max_value=1000000,
            value=18100,
            step=100,
            help="Volumen estimado de búsquedas mensuales"
        )
        
        st.divider()
        
        if st.button("🚀 Generar Fan Out", type="primary"):
            st.session_state.generated = True
            st.session_state.generator = QueryFanOutGenerator(main_keyword, volume)
        
        st.divider()
        
        st.markdown("### 📚 Recursos")
        st.markdown("""
        - [Documentación](https://github.com)
        - [Reportar bug](https://github.com)
        - [Sugerir mejora](https://github.com)
        """)
    
    # Main content
    if 'generated' not in st.session_state:
        st.info("👈 Configura tu keyword en el panel lateral y haz click en 'Generar Fan Out'")
        
        # Información inicial
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 ¿Qué es Query Fan Out?")
            st.write("Técnica SEO que expande una keyword principal en múltiples variaciones según intención de búsqueda.")
        
        with col2:
            st.markdown("### 🎯 ¿Para qué sirve?")
            st.write("Evita canibalización de keywords y crea contenido diferenciado para cada etapa del customer journey.")
        
        with col3:
            st.markdown("### 🚀 ¿Cómo funciona?")
            st.write("Genera automáticamente categorías, queries y propuestas de contenido listas para implementar.")
    
    else:
        generator = st.session_state.generator
        
        # Tabs principales
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualización", "🔍 Queries", "📝 Propuestas de Contenido", "💾 Exportar"])
        
        with tab1:
            st.subheader(f"Query Fan Out: {generator.main_keyword}")
            st.caption(f"Volumen mensual estimado: {generator.volume:,} búsquedas")
            
            # Gráfico sunburst
            fig = generator.generate_sunburst_chart()
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Haz click en los segmentos para explorar las queries de cada categoría")
        
        with tab2:
            st.subheader("🔍 Queries por Categoría")
            
            for cat in generator.categories:
                with st.expander(f"{cat.icon} {cat.name} - {cat.volume} volumen", expanded=False):
                    st.markdown(f"**Tipo de contenido:** {cat.content_type}")
                    st.markdown(f"**Color identificativo:** `{cat.color}`")
                    
                    st.markdown("**Queries identificadas:**")
                    for i, query in enumerate(cat.queries, 1):
                        st.markdown(f"{i}. {query}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📋 Copiar queries", key=f"copy_{cat.name}"):
                            st.code("\n".join(cat.queries))
                    with col2:
                        if st.button(f"➕ Agregar más queries", key=f"add_{cat.name}"):
                            st.info("Funcionalidad próximamente")
        
        with tab3:
            st.subheader("📝 Propuestas de Contenido Detalladas")
            
            proposals = generator.generate_content_proposals()
            
            # Informacional
            with st.container():
                st.markdown(f"""
                <div style='border-left: 5px solid #3b82f6; padding: 1.5rem; border-radius: 10px; background-color: rgba(59, 130, 246, 0.1); margin-bottom: 2rem;'>
                    <h3>📚 {proposals['informacional']['titulo']}</h3>
                    <p><strong>Tipo:</strong> {proposals['informacional']['tipo']}</p>
                    <p><strong>Objetivo:</strong> {proposals['informacional']['objetivo']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Estructura sugerida:**")
                    for item in proposals['informacional']['estructura']:
                        st.markdown(f"• {item}")
                
                with col2:
                    st.markdown("**🎯 Call-to-Action:**")
                    st.success(proposals['informacional']['cta'])
            
            # Comparativa
            with st.container():
                st.markdown(f"""
                <div style='border-left: 5px solid #8b5cf6; padding: 1.5rem; border-radius: 10px; background-color: rgba(139, 92, 246, 0.1); margin-bottom: 2rem;'>
                    <h3>📊 {proposals['comparativa']['titulo']}</h3>
                    <p><strong>Tipo:</strong> {proposals['comparativa']['tipo']}</p>
                    <p><strong>Objetivo:</strong> {proposals['comparativa']['objetivo']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Estructura sugerida:**")
                    for item in proposals['comparativa']['estructura']:
                        st.markdown(f"• {item}")
                
                with col2:
                    st.markdown("**🎯 Call-to-Action:**")
                    st.success(proposals['comparativa']['cta'])
            
            # Transaccional
            with st.container():
                st.markdown(f"""
                <div style='border-left: 5px solid #10b981; padding: 1.5rem; border-radius: 10px; background-color: rgba(16, 185, 129, 0.1); margin-bottom: 2rem;'>
                    <h3>💰 {proposals['transaccional']['titulo']}</h3>
                    <p><strong>Tipo:</strong> {proposals['transaccional']['tipo']}</p>
                    <p><strong>Objetivo:</strong> {proposals['transaccional']['objetivo']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Estructura sugerida:**")
                    for item in proposals['transaccional']['estructura']:
                        st.markdown(f"• {item}")
                
                with col2:
                    st.markdown("**🎯 Call-to-Action:**")
                    st.success(proposals['transaccional']['cta'])
            
            # Marca
            with st.container():
                st.markdown(f"""
                <div style='border-left: 5px solid #f97316; padding: 1.5rem; border-radius: 10px; background-color: rgba(249, 115, 22, 0.1); margin-bottom: 2rem;'>
                    <h3>🎯 {proposals['marca']['titulo']}</h3>
                    <p><strong>Tipo:</strong> {proposals['marca']['tipo']}</p>
                    <p><strong>Objetivo:</strong> {proposals['marca']['objetivo']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Estructura sugerida:**")
                    for item in proposals['marca']['estructura']:
                        st.markdown(f"• {item}")
                
                with col2:
                    st.markdown("**🎯 Call-to-Action:**")
                    st.success(proposals['marca']['cta'])
            
            # Long Tail
            with st.container():
                st.markdown(f"""
                <div style='border-left: 5px solid #ec4899; padding: 1.5rem; border-radius: 10px; background-color: rgba(236, 72, 153, 0.1); margin-bottom: 2rem;'>
                    <h3>👥 {proposals['longtail']['titulo']}</h3>
                    <p><strong>Tipo:</strong> {proposals['longtail']['tipo']}</p>
                    <p><strong>Objetivo:</strong> {proposals['longtail']['objetivo']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Estructura sugerida:**")
                    for item in proposals['longtail']['estructura']:
                        st.markdown(f"• {item}")
                
                with col2:
                    st.markdown("**🎯 Call-to-Action:**")
                    st.success(proposals['longtail']['cta'])
        
        with tab4:
            st.subheader("💾 Exportar Resultados")
            
            # Preparar datos para exportación
            export_data = {
                "keyword_principal": generator.main_keyword,
                "volumen_mensual": generator.volume,
                "categorias": []
            }
            
            for cat in generator.categories:
                export_data["categorias"].append({
                    "nombre": cat.name,
                    "tipo_contenido": cat.content_type,
                    "volumen": cat.volume,
                    "queries": cat.queries
                })
            
            export_data["propuestas_contenido"] = proposals
            
            # Formato JSON
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Descargar JSON",
                    data=json_str,
                    file_name=f"query_fanout_{generator.main_keyword.replace(' ', '_')}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV simplificado
                csv_data = "Categoría,Tipo Contenido,Volumen,Query\n"
                for cat in generator.categories:
                    for query in cat.queries:
                        csv_data += f"{cat.name},{cat.content_type},{cat.volume},{query}\n"
                
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv_data,
                    file_name=f"queries_{generator.main_keyword.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            
            st.markdown("---")
            st.markdown("### 📋 Vista previa JSON:")
            st.json(export_data)

if __name__ == "__main__":
    main()
