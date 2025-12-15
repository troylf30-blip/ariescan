import streamlit as st
import pdfplumber
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =========================================================
# 1. SETUP & CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CareerLift AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (REVISED FOR CONTRAST & ALIGNMENT) ---
st.markdown("""
<style>
    /* 1. Background & Font Global */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* 2. Header Styles */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #3B82F6, #2DD4BF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }
    
    /* 3. Score Card (Perbaikan Alignment) */
    .score-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        padding: 20px; /* Dikurangi sedikit biar ga terlalu tinggi */
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .score-value {
        font-size: 4rem; /* Font size disesuaikan */
        font-weight: 900;
        margin: 5px 0;
        background: -webkit-linear-gradient(90deg, #60A5FA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-label {
        font-size: 1rem;
        color: #CBD5E1;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* 4. Keyword Chips (HIGH CONTRAST FIX) */
    .tags-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 10px 0;
    }
    
    /* Missing: Kontras Tinggi (Teks Putih di Background Merah Pekat) */
    .keyword-tag-missing {
        background-color: #7f1d1d; /* Merah Marun Gelap */
        color: #ffffff !important; /* Teks Putih Wajib */
        border: 1px solid #ef4444; /* Border Merah Terang */
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Matched: Kontras Tinggi (Teks Putih di Background Hijau Pekat) */
    .keyword-tag-match {
        background-color: #064e3b; /* Hijau Botol Gelap */
        color: #d1fae5 !important; /* Teks Hijau Pucat/Putih */
        border: 1px solid #10b981; /* Border Hijau Neon */
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* 5. Tombol Utama (HOVER FIX) */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563EB, #0F766E);
        color: white !important; /* Force text white */
        font-weight: bold;
        font-size: 1.1rem;
        border-radius: 12px;
        padding: 15px 0;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    /* Saat hover, background jadi sedikit lebih terang, teks tetap putih */
    .stButton>button:hover {
        background: linear-gradient(90deg, #3B82F6, #14B8A6);
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }
    
    /* 6. Input Fields Styling */
    .stTextArea textarea {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F1F5F9 !important;
        border-radius: 10px;
    }
    .stFileUploader {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border: 1px dashed #475569;
    }
    
    /* Helper untuk Alignment */
    .align-bottom {
        display: flex;
        align-items: flex-end;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. LOGIC BACKEND
# =========================================================

@st.cache_resource
def load_ai_model():
    return SentenceTransformer('bert-base-uncased')

HR_JARGON = {
    'analyze', 'assist', 'build', 'collaborate', 'communicate', 'coordinate', 'contribute',
    'create', 'demonstrate', 'deploy', 'design', 'develop', 'drive', 'ensure', 'establish',
    'execute', 'facilitate', 'generate', 'help', 'identify', 'implement', 'improve',
    'integrate', 'interact', 'lead', 'maintain', 'manage', 'monitor', 'operate', 'optimize',
    'participate', 'perform', 'plan', 'prepare', 'provide', 'recommend', 'resolve',
    'review', 'support', 'test', 'troubleshoot', 'understand', 'utilize', 'verify', 'work',
    'write', 'looking', 'seeking', 'working', 'using', 'learning', 'growing', 'wanted',
    'able', 'capable', 'comfortable', 'competent', 'dynamic', 'effective', 'efficient',
    'essential', 'excellent', 'experienced', 'familiar', 'fluent', 'good', 'great',
    'hands-on', 'high', 'ideal', 'innovative', 'knowledgeable', 'motivated', 'passionate',
    'preferred', 'proficient', 'qualified', 'relevant', 'reliable', 'responsible',
    'skilled', 'solid', 'strong', 'successful', 'superb', 'technical', 'thorough',
    'willing', 'required', 'desirable', 'plus', 'optional', 'proactive',
    'ability', 'applicant', 'application', 'benefit', 'bonus', 'candidate', 'career',
    'client', 'company', 'degree', 'department', 'description', 'duties', 'education',
    'email', 'employee', 'environment', 'experience', 'field', 'gender', 'goal',
    'industry', 'job', 'knowledge', 'level', 'location', 'mission', 'offer',
    'opportunity', 'organization', 'people', 'phone', 'position', 'practice', 'problem',
    'process', 'product', 'project', 'qualification', 'quality', 'reference',
    'requirement', 'responsibility', 'result', 'role', 'salary', 'service', 'skill',
    'solution', 'standard', 'strategy', 'success', 'summary', 'system', 'task', 'team',
    'technology', 'term', 'time', 'tool', 'understanding', 'user', 'vision', 'year',
    'both', 'into', 'tuning', 'audiences', 'specifications'
}

BASIC_STOPWORDS = {
    'and', 'the', 'to', 'in', 'of', 'for', 'with', 'a', 'an', 'is', 'are', 'on', 'at', 
    'be', 'as', 'by', 'that', 'this', 'it', 'from', 'or', 'we', 'you', 'can', 'will',
    'if', 'not', 'but', 'what', 'all', 'were', 'when', 'there', 'use', 'any', 'do',
    'has', 'have', 'which', 'their', 'other', 'more', 'about', 'out', 'up', 'so'
}

ALL_IGNORED_WORDS = HR_JARGON.union(BASIC_STOPWORDS)

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except:
        return None

def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_keywords_manual(text):
    words = re.findall(r'\b[a-z0-9+#]{2,}\b', text.lower())
    keywords = set()
    for w in words:
        if w not in ALL_IGNORED_WORDS:
            if not w.isdigit(): 
                keywords.add(w)
    return keywords

def calculate_hybrid_score(cv_text, jd_text):
    model = load_ai_model()
    clean_cv = clean_text(cv_text)
    clean_jd = clean_text(jd_text)
    
    # 1. AI Score
    embeddings = model.encode([clean_cv, clean_jd])
    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1), 
        embeddings[1].reshape(1, -1)
    )[0][0]
    ai_score = max(similarity, 0) * 100
    
    # 2. Keyword Score
    cv_keywords = get_keywords_manual(clean_cv)
    jd_keywords = get_keywords_manual(clean_jd)
    
    match_count = len(cv_keywords.intersection(jd_keywords))
    total_jd_needed = len(jd_keywords)
    
    if total_jd_needed == 0:
        keyword_score = 0
    else:
        keyword_score = (match_count / total_jd_needed) * 100
    
    final_score = (0.4 * ai_score) + (0.6 * keyword_score)
    return round(final_score, 2), round(ai_score, 2), round(keyword_score, 2), jd_keywords, cv_keywords

# =========================================================
# 3. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("### 🤖 AIRIE SCAN")
    st.info("""
    **Fitur Utama:**
    * **AI Matching:** Memahami konteks pengalaman.
    * **Keyword Scanner:** Mendeteksi skill teknis.
    * **Hybrid Score:** Kombinasi cerdas untuk akurasi tinggi.
    """)
    st.markdown("---")
    st.caption("v1.0.3 MVP - Dark Mode Edition")

# =========================================================
# 4. MAIN UI
# =========================================================

# Header
st.markdown("<h1 class='main-header'>AI Resume Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analisis CV cerdas untuk menembus sistem ATS & Rekruter.</p>", unsafe_allow_html=True)

# Layout Input (2 Kolom)
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 📂 1. Upload CV (PDF)")
    uploaded_file = st.file_uploader("Drop CV kamu disini", type=["pdf"])
    cv_text = ""
    
    # Placeholder container untuk pesan sukses agar sejajar
    status_container = st.empty()
    
    if uploaded_file:
        with st.spinner("Mengurai PDF..."):
            cv_text = extract_text_from_pdf(uploaded_file)
            if cv_text:
                status_container.success(f"✅ Berhasil membaca: {uploaded_file.name}")
            else:
                status_container.error("❌ Gagal membaca file.")

with col2:
    st.markdown("##### 📝 2. Job Description")
    # HEIGHT FIX: Saya naikkan tinggi text area agar sejajar visual dengan box upload + pesan sukses
    jd_text = st.text_area("Paste kualifikasi lowongan disini...", height=205, placeholder="Contoh: We are looking for Data Scientist with Python, SQL...")

# Tombol
st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🚀 Analisis Kecocokan Sekarang")

# =========================================================
# 5. RESULT DASHBOARD
# =========================================================

if analyze_btn:
    if not cv_text or not jd_text:
        st.warning("⚠️ Data belum lengkap. Mohon upload CV dan isi Job Desc.")
    else:
        with st.spinner("🧠 AI sedang berpikir..."):
            final_score, ai_score, kw_score, jd_kws, cv_kws = calculate_hybrid_score(cv_text, jd_text)
            
            st.markdown("---")
            
            # SCORE CARD SECTION
            col_score, col_stats = st.columns([1.5, 2.5])
            
            with col_score:
                # ALIGNMENT FIX: Menggunakan margin top kosong agar "sejajar" dengan Score Card
                st.markdown("<div style='margin-top: 1px;'></div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-label">Total Match Score</div>
                    <div class="score-value">{final_score:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback Text - Diatur agar tidak terlalu jauh jaraknya
                if final_score >= 75:
                    st.success("🌟 **Excellent Match!**")
                elif final_score >= 50:
                    st.warning("⚠️ **Potential Match**")
                else:
                    st.error("❌ **Low Match**")

            with col_stats:
                st.markdown("#### 📊 Performance Breakdown")
                
                # Breakdown Items
                st.write(f"**🤖 AI Semantic Match** ({ai_score:.2f}%)")
                st.progress(int(ai_score))
                st.caption("Kecocokan berdasarkan makna kalimat & konteks pengalaman.")
                
                st.write(f"**🔑 Keyword Coverage** ({kw_score:.2f}%)")
                st.progress(int(kw_score))
                st.caption("Persentase skill wajib yang terpenuhi.")

            # TABS DETAIL
            st.markdown("<br><br>", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["🔴 Skill Missing (Perlu Ditambah)", "🟢 Skill Matched (Sudah Ada)"])
            
            missing = list(jd_kws - cv_kws)
            matched = list(cv_kws.intersection(jd_kws))

            with tab1:
                if missing:
                    st.markdown("##### 💡 Saran Perbaikan:")
                    st.markdown("Skill ini **ditemukan di Lowongan** tapi **TIDAK ADA di CV kamu**:")
                    
                    html_tags = '<div class="tags-container">'
                    for w in missing[:30]: 
                        html_tags += f'<span class="keyword-tag-missing">{w}</span>'
                    html_tags += '</div>'
                    st.markdown(html_tags, unsafe_allow_html=True)
                else:
                    st.success("🎉 Sempurna! Tidak ada skill penting yang terlewat.")

            with tab2:
                if matched:
                    st.markdown("##### ✅ Kekuatan Profil Kamu:")
                    st.markdown("Skill berikut sudah sesuai dengan permintaan:")
                    
                    html_tags = '<div class="tags-container">'
                    for w in matched:
                        html_tags += f'<span class="keyword-tag-match">{w}</span>'
                    html_tags += '</div>'
                    st.markdown(html_tags, unsafe_allow_html=True)
                else:
                    st.info("Belum ada skill spesifik yang terdeteksi sama persis.")
