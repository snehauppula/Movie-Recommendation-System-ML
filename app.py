import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from recommender import load_data, build_vectorizer, search_movie, find_similar_movies, clean_title

# Page Config - No sidebar
st.set_page_config(
    page_title="🎬 CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling with IMPROVED TEXT VISIBILITY
st.markdown("""
    <style>
        /* Hide sidebar */
        [data-testid="collapsedControl"] {
            display: none;
        }
        
        /* Full width main content */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Title styling */
        .title-main {
            text-align: center;
            color: white;
            font-size: 4rem;
            font-weight: 900;
            margin-bottom: 10px;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
            letter-spacing: 2px;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.95);
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 30px;
            letter-spacing: 1px;
        }
        
        /* Movie card - LARGE BOLD TEXT */
        .movie-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            color: white;
            border: 2px solid rgba(255,255,255,0.2);
        }
        
        .movie-card h4 {
            font-size: 1.4rem !important;
            font-weight: 900 !important;
            margin: 0 0 15px 0 !important;
            color: #ffffff !important;
            text-shadow: 2px 2px 3px rgba(0,0,0,0.3) !important;
        }
        
        .movie-card p {
            font-size: 1.05rem !important;
            margin: 10px 0 !important;
            font-weight: 600 !important;
            color: rgba(255,255,255,0.95) !important;
        }
        
        /* Recommendation item - HIGHLY READABLE */
        .rec-item {
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(245,245,250,0.98));
            border-left: 7px solid #667eea;
            border-radius: 12px;
            padding: 22px;
            margin: 15px 0;
            box-shadow: 0 6px 14px rgba(0,0,0,0.15);
            border-top: 3px solid rgba(102, 126, 234, 0.3);
        }
        
        .rec-item strong {
            font-size: 1.25rem !important;
            color: #2d2d2d !important;
            display: block !important;
            margin-bottom: 10px !important;
            font-weight: 800 !important;
        }
        
        .rec-item small {
            font-size: 1.05rem !important;
            color: #555555 !important;
            font-weight: 600 !important;
        }
        
        /* Score badge - LARGER AND BOLDER */
        .score-badge {
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 10px 18px;
            border-radius: 25px;
            font-weight: 900 !important;
            font-size: 1.1rem !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            box-shadow: 0 4px 10px rgba(245, 87, 108, 0.3);
        }
        
        /* Search title */
        .search-title {
            font-size: 1.7rem !important;
            font-weight: 900 !important;
            color: white !important;
            margin-bottom: 25px !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        }
        
        /* Recommendations title */
        .recommendations-title {
            font-size: 1.8rem !important;
            font-weight: 900 !important;
            color: white !important;
            margin: 30px 0 20px 0 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        }
        
        /* Button styling - LARGE BOLD TEXT */
        .stButton>button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 14px 30px;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
            width: 100%;
            box-shadow: 0 4px 10px rgba(245, 87, 108, 0.3);
        }
        
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 14px rgba(245, 87, 108, 0.4);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: rgba(255,255,255,0.9);
            border-radius: 8px;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
        }
        
        /* Analytics title */
        .analytics-title {
            font-size: 1.5rem !important;
            font-weight: 900 !important;
            color: white !important;
            margin-bottom: 20px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data(ttl=3600)
def load_app_data():
    """Load and prepare data"""
    try:
        movies, ratings = load_data()
        movies = movies.copy()
        # Add cleaned titles
        movies["clean_title"] = movies["title"].apply(clean_title)
        return movies, ratings
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data(ttl=3600)
def vectorize_data(movies_df):
    """Build TF-IDF vectorizer"""
    try:
        if "clean_title" not in movies_df.columns:
            return None, None
        vectorizer, vectors = build_vectorizer(movies_df["clean_title"])
        return vectorizer, vectors
    except Exception as e:
        st.error(f"Error vectorizing data: {e}")
        return None, None

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="title-main">🎬 CineMatch</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover Movies You\'ll Love ✨</p>', unsafe_allow_html=True)

# Main content
movies, ratings = load_app_data()

if movies is None or ratings is None:
    st.error("Failed to load data. Please check your CSV files.")
    st.stop()

# Load vectorizer
vectorizer, vectors = vectorize_data(movies)
if vectorizer is None or vectors is None:
    st.error("Failed to vectorize data.")
    st.stop()

# Tabs for different modes
tab1, tab2 = st.tabs(["🎯 Search & Discover", "⭐ Browse Top Movies"])

# TAB 1: SEARCH MODE
with tab1:
    st.markdown("---")
    
    # Search controls in three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        query = st.text_input("🎬 Movie Title", placeholder="e.g., The Matrix, Inception...")
    
    with col2:
        min_score = st.slider("📊 Similarity Threshold", 0.0, 1.0, 0.2, 0.05)
    
    with col3:
        top_n = st.slider("🔢 Results to Show", 1, 15, 8)
    
    search_btn = st.button("🔍 Search", use_container_width=True, key="search_btn")
    
    # Search logic
    if search_btn:
        if not query or not query.strip():
            st.warning("⚠️ Please enter a movie title to search!")
        else:
            st.markdown("---")
            
            # Search for matches
            with st.spinner("🎬 Searching for movies..."):
                matches = search_movie(query, movies, vectorizer, vectors, top_n=top_n, min_score=min_score)
            
            if matches.empty:
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.error("😢 No matches found!")
                    st.info("💡 Try:\n• Different keywords\n• Lower similarity threshold\n• Check spelling")
            else:
                st.markdown(f'<div class="search-title">🎯 Found {len(matches)} Match(es) for "{query}"</div>', unsafe_allow_html=True)
                
                # Display matches in columns
                cols = st.columns(min(3, len(matches)))
                for idx, (i, row) in enumerate(matches.iterrows()):
                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <h4>🎬 {row['title']}</h4>
                            <p style="font-size: 1.05rem; margin: 12px 0;"><strong style="color: #fff; font-size: 1.1rem;">Genres:</strong> <span style="color: rgba(255,255,255,0.95); font-weight: 600;">{row['genres']}</span></p>
                            <p style="margin-top: 15px;"><span class="score-badge">Score: {row['_score']:.1%}</span></p>
                            <p style="font-size: 0.95rem; color: rgba(255,255,255,0.85); margin-top: 10px;">Movie ID: {int(row['movieId'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Select movie for recommendations
                st.markdown("---")
                st.markdown('<h3 style="font-size: 1.4rem; font-weight: 900; color: white; margin-bottom: 20px;">⭐ Select a Movie for Recommendations</h3>', unsafe_allow_html=True)
                
                options = [f"{row['title']} • {row['genres'].split('|')[0]}" for _, row in matches.iterrows()]
                selected_idx = st.selectbox("Pick a movie:", 
                                           range(len(options)), 
                                           format_func=lambda i: options[i],
                                           key="movie_select")
                
                if selected_idx is not None:
                    selected_movie_id = int(matches.iloc[selected_idx]['movieId'])
                    selected_title = matches.iloc[selected_idx]['title']
                    
                    # Get recommendations
                    with st.spinner(f"✨ Finding recommendations for '{selected_title}'..."):
                        recs = find_similar_movies(selected_movie_id, movies, ratings)
                    
                    if recs.empty:
                        st.info(f"ℹ️ No strong recommendations found for '{selected_title}'. Try another movie!")
                    else:
                        st.markdown(f'<div class="recommendations-title">✨ Top {len(recs)} Recommendations</div>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size: 1.2rem; color: rgba(255,255,255,0.95); margin-bottom: 30px; font-weight: 700;">Based on users who enjoyed <span style="color: #fff; font-weight: 900; font-size: 1.25rem;">{selected_title}</span></p>', unsafe_allow_html=True)
                        
                        # Table-style headings for recommendations
                        head_col1, head_col2, head_col3, head_col4 = st.columns([0.4, 3, 1.2, 1.2])
                        with head_col1:
                            st.markdown("<div style='font-size:1.1rem; font-weight:900; color:#f5576c;'>#</div>", unsafe_allow_html=True)
                        with head_col2:
                            st.markdown("<div style='font-size:1.1rem; font-weight:900; color:#fff;'>Movie Title</div>", unsafe_allow_html=True)
                        with head_col3:
                            st.markdown("<div style='font-size:1.1rem; font-weight:900; color:#fff;'>Score (x)</div>", unsafe_allow_html=True)
                        with head_col4:
                            st.markdown("<div style='font-size:1.1rem; font-weight:900; color:#fff;'>% Similar Users</div>", unsafe_allow_html=True)

                        # Display recommendations in beautiful layout
                        for idx, (_, rec) in enumerate(recs.head(10).iterrows(), 1):
                            col1, col2, col3, col4 = st.columns([0.4, 3, 1.2, 1.2])
                            with col1:
                                st.markdown(f"<h2 style='color: #f5576c; font-size: 2.2rem; font-weight: 900; margin: 0;'>{idx}</h2>", unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                <div class="rec-item">
                                    <strong>{rec['title']}</strong><br/>
                                    <small>{rec['genres']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                score_pct = rec['score']
                                st.markdown(f"<div style='font-size:1.25rem; font-weight:800; color:#fff;'>{score_pct:.2f}x</div>", unsafe_allow_html=True)
                            with col4:
                                st.markdown(f"<div style='font-size:1.25rem; font-weight:800; color:#fff;'>{rec['similar']:.1%}</div>", unsafe_allow_html=True)
                        
                        # Visualizations
                        st.markdown("---")
                        st.markdown('<h2 class="analytics-title">📊 Analytics & Insights</h2>', unsafe_allow_html=True)
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            st.markdown('<p style="font-size: 1.2rem; font-weight: 900; color: white; margin-bottom: 15px;">📈 Score Distribution</p>', unsafe_allow_html=True)
                            rec_display = recs.head(10).copy()
                            rec_display = rec_display.sort_values('score', ascending=True)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = plt.cm.viridis(np.linspace(0, 1, len(rec_display)))
                            ax.barh(rec_display['title'], rec_display['score'], color=colors)
                            ax.set_xlabel('Recommendation Score', fontsize=12, fontweight='bold')
                            ax.set_title('Score Comparison', fontsize=13, fontweight='bold')
                            ax.tick_params(labelsize=10)
                            ax.grid(axis='x', alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                        
                        with viz_col2:
                            st.markdown('<p style="font-size: 1.2rem; font-weight: 900; color: white; margin-bottom: 15px;">👥 User Preference Comparison</p>', unsafe_allow_html=True)
                            comp_data = recs.head(10)[['title', 'similar', 'all']].set_index('title')
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            x = np.arange(len(comp_data))
                            width = 0.35
                            
                            bars1 = ax.bar(x - width/2, comp_data['similar'], width, label='Similar Users', 
                                          color='#667eea', alpha=0.8)
                            bars2 = ax.bar(x + width/2, comp_data['all'], width, label='All Users', 
                                          color='#764ba2', alpha=0.8)
                            
                            ax.set_ylabel('Preference %', fontsize=12, fontweight='bold')
                            ax.set_title('Similar vs All Users', fontsize=13, fontweight='bold')
                            ax.set_xticks(x)
                            ax.set_xticklabels(comp_data.index, rotation=45, ha='right', fontsize=9)
                            ax.legend(fontsize=11)
                            ax.grid(axis='y', alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)

# TAB 2: BROWSE MODE
with tab2:
    st.markdown("---")
    st.markdown('<h2 style="font-size: 1.6rem; font-weight: 900; color: white; margin-bottom: 25px;">⭐ Popular Movies to Explore</h2>', unsafe_allow_html=True)
    
    # Show popular movies in grid
    top_movies = movies.head(12)
    cols = st.columns(4)
    
    for idx, (_, movie) in enumerate(top_movies.iterrows()):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="movie-card">
                <h4 style="font-size: 1.3rem; margin-bottom: 15px;">🎬 {movie['title']}</h4>
                <p style="font-size: 1.05rem; font-weight: 600;">{movie['genres']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"🎯 Get Recommendations", key=f"explore_{idx}", use_container_width=True):
                selected_movie_id = int(movie['movieId'])
                selected_title = movie['title']
                
                with st.spinner(f"✨ Finding recommendations for '{selected_title}'..."):
                    recs = find_similar_movies(selected_movie_id, movies, ratings)
                
                if recs.empty:
                    st.warning(f"No recommendations for {selected_title}")
                else:
                    st.success(f"✨ Found {len(recs)} recommendations for '{selected_title}'!")
                    
                    # Display recommendations
                    for rec_idx, (_, rec) in enumerate(recs.head(8).iterrows(), 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
                            <div class="rec-item">
                                <strong>{rec_idx}. {rec['title']}</strong><br/>
                                <small>{rec['genres']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.metric("Score", f"{rec['score']:.2f}x", label_visibility="collapsed")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); font-size: 1.05rem; font-weight: 600;'>✨ Powered by TF-IDF & Collaborative Filtering | Made with ❤️</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    import os
    import sys
    # This allow running with 'python app.py' directly by wrapping it in streamlit
    if not os.environ.get("STREAMLIT_RUNNING"):
        print("🚀 Starting Streamlit app...")
        os.environ["STREAMLIT_RUNNING"] = "true"
        # We use sys.executable to ensure we use the same python interpreter
        # We use -m streamlit run to run the current file
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
