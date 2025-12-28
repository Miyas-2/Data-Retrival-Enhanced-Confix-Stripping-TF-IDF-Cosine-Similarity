import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter

# Library Pembaca File
import PyPDF2
from docx import Document

# Library NLP (Sastrawi)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. KONFIGURASI & FUNGSI UTILITY (BACKEND)
# ==========================================

st.set_page_config(page_title="Sistem Temu Balik Dokumen (VSM)", layout="wide")

# Cache agar loading Sastrawi tidak berat saat refresh
@st.cache_resource
def load_sastrawi_tools():
    # Inisialisasi Stemmer (Nazief & Adriani)
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    
    # Inisialisasi Stopword Remover
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    return stemmer, stopword_remover

stemmer, stopword_remover = load_sastrawi_tools()

# --- Fungsi Pembaca File ---
def read_txt(file):
    return file.getvalue().decode("utf-8")

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    doc = Document(file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

# --- Fungsi Preprocessing ---
def preprocess_text(text):
    # 1. Case Folding
    text = text.lower()
    
    # 2. Cleaning / Filtering (Hapus angka & tanda baca)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. Stopword Removal
    text = stopword_remover.remove(text)
    
    # 4. Tokenizing
    tokens = text.split()
    
    # 5. Stemming (Proses paling lama)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Filter token kosong
    stemmed_tokens = [t for t in stemmed_tokens if len(t) > 0]
    
    return stemmed_tokens

# --- Fungsi Perhitungan VSM ---
def calculate_vsm(documents, query):
    # A. Preprocessing Semua Dokumen
    processed_docs = {}
    for filename, content in documents.items():
        processed_docs[filename] = preprocess_text(content)
        
    # B. Preprocessing Query
    processed_query = preprocess_text(query)
    
    # C. Buat Vocabulary (Daftar Kata Unik)
    all_terms = set(processed_query)
    for tokens in processed_docs.values():
        all_terms.update(tokens)
    all_terms = sorted(list(all_terms))
    
    # D. Hitung TF (Term Frequency)
    tf_data = {}
    for term in all_terms:
        tf_data[term] = {'Query': processed_query.count(term)}
        for doc_id in documents.keys():
            tf_data[term][doc_id] = processed_docs[doc_id].count(term)
    
    df_tf = pd.DataFrame(tf_data).T
    
    # E. Hitung DF & IDF
    N = len(documents)
    idf_data = {}
    for term in all_terms:
        # Hitung di berapa dokumen kata muncul (Query tidak ikut hitung IDF)
        df_count = sum([1 for doc_id in documents.keys() if term in processed_docs[doc_id]])
        
        # Rumus IDF + Smoothing agar tidak error div by zero
        idf_val = math.log10(N / (df_count + 1)) + 1 
        idf_data[term] = idf_val
        
    # F. Hitung Bobot W (TF * IDF)
    w_data = {}
    for term in all_terms:
        w_data[term] = {'Query': df_tf.loc[term, 'Query'] * idf_data[term]}
        for doc_id in documents.keys():
            w_data[term][doc_id] = df_tf.loc[term, doc_id] * idf_data[term]
            
    df_w = pd.DataFrame(w_data).T
    
    return processed_docs, processed_query, df_tf, idf_data, df_w

# --- Fungsi Cosine Similarity ---
def calculate_cosine(df_w):
    similarities = {}
    query_col = 'Query'
    doc_cols = [c for c in df_w.columns if c != query_col]
    
    # Panjang Vektor Query
    q_vec_len = np.sqrt((df_w[query_col] ** 2).sum())
    
    for doc in doc_cols:
        # Panjang Vektor Dokumen
        d_vec_len = np.sqrt((df_w[doc] ** 2).sum())
        
        # Dot Product
        dot_product = (df_w[query_col] * df_w[doc]).sum()
        
        # Rumus Cosine
        if q_vec_len * d_vec_len == 0:
            sim = 0
        else:
            sim = dot_product / (q_vec_len * d_vec_len)
            
        similarities[doc] = sim
        
    return similarities

# ==========================================
# 2. USER INTERFACE (FRONTEND)
# ==========================================

st.title("🔍 Aplikasi Temu Balik Dokumen")
st.markdown("Algoritma: **Vector Space Model (VSM)** | Stemming: **Sastrawi**")

# --- SIDEBAR (INPUT FILE) ---
with st.sidebar:
    st.header("1. Upload Dokumen")
    uploaded_files = st.file_uploader(
        "Upload file (PDF, DOCX, TXT)", 
        type=['txt', 'pdf', 'docx'], 
        accept_multiple_files=True
    )
    
    st.info("Catatan: Upload minimal 2 dokumen agar perbandingan terlihat.")

# --- MAIN AREA ---
if uploaded_files:
    # 1. BACA FILE
    documents = {}
    for file in uploaded_files:
        ext = file.name.split('.')[-1].lower()
        try:
            if ext == 'txt': text = read_txt(file)
            elif ext == 'pdf': text = read_pdf(file)
            elif ext == 'docx': text = read_docx(file)
            documents[file.name] = text
        except Exception as e:
            st.error(f"Gagal membaca {file.name}: {e}")

    st.success(f"Berhasil memuat {len(documents)} dokumen!")
    
    # Expander untuk intip isi dokumen
    with st.expander("📄 Lihat Isi Dokumen Asli"):
        tabs = st.tabs(list(documents.keys()))
        for i, (name, content) in enumerate(documents.items()):
            with tabs[i]:
                st.text_area("Isi Teks:", content, height=150)

    st.divider()

    # 2. INPUT QUERY
    st.header("2. Pencarian (Query)")
    query = st.text_input("Masukkan kata kunci:", placeholder="Contoh: politik ekonomi indonesia")
    
    if st.button("🔎 Cari Dokumen") and query:
        with st.spinner('Sedang melakukan Preprocessing & Stemming... (Mohon tunggu)'):
            
            # --- PROSES BACKEND ---
            proc_docs, proc_query, df_tf, idf_data, df_w = calculate_vsm(documents, query)
            similarities = calculate_cosine(df_w)
            
            # --- TAMPILAN HASIL (TABS) ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "🛠️ Preprocessing", 
                "📊 TF-IDF Matrix", 
                "🧮 Perhitungan VSM", 
                "🏆 Hasil Ranking"
            ])
            
            # TAB 1: HASIL PREPROCESSING
            with tab1:
                st.subheader("Hasil Stemming Query")
                st.code(f"Original: {query}\nStemmed : {proc_query}")
                
                st.subheader("Statistik Kata Dasar Dokumen")
                doc_stats = []
                for name, tokens in proc_docs.items():
                    doc_stats.append({
                        "Nama Dokumen": name,
                        "Jumlah Kata Dasar": len(tokens),
                        "Contoh Token (Awal)": ", ".join(tokens[:10]) + "..."
                    })
                st.dataframe(pd.DataFrame(doc_stats))

            # TAB 2: MATRIKS TF-IDF
            with tab2:
                st.write("#### 1. Term Frequency (TF)")
                st.dataframe(df_tf.style.highlight_max(axis=0))
                
                st.write("#### 2. Bobot Akhir (W = TF * IDF)")
                st.dataframe(df_w.style.background_gradient(cmap="Blues"))
                
                with st.expander("Lihat Nilai IDF"):
                    st.json(idf_data)

            # TAB 3: DETAIL PERHITUNGAN (FIXED)
            with tab3:
                st.write("Detail perhitungan **Cosine Similarity**:")
                
                q_vec_len = np.sqrt((df_w['Query'] ** 2).sum())
                
                st.markdown(f"""
                **Rumus Cosine Similarity:**
                $$
                Sim(Q, D) = \\frac{{Q \cdot D}}{{|Q| \\times |D|}} = \\frac{{\\sum (W_q \cdot W_d)}}{{\\sqrt{{\\sum W_q^2}} \cdot \\sqrt{{\\sum W_d^2}}}}
                $$
                
                **Nilai Referensi:**
                * Panjang Vektor Query ($|Q|$): `{q_vec_len:.4f}`
                """)
                
                st.divider()

                for name, sim in similarities.items():
                    st.markdown(f"### 📄 Dokumen: {name}")
                    
                    d_vec_len = np.sqrt((df_w[name] ** 2).sum())
                    
                    term_details = []
                    dot_prod_accum = 0
                    
                    # PERBAIKAN DI SINI: Iterasi berdasarkan Index (Term), bukan Columns
                    for term in df_w.index:
                        w_q = df_w.loc[term, 'Query']  # Akses yang benar: [Baris, Kolom]
                        w_d = df_w.loc[term, name]     # Akses yang benar: [Baris, Kolom]
                        
                        product = w_q * w_d
                        dot_prod_accum += product
                        
                        # Hanya tampilkan jika bobot query > 0 (kata yang dicari)
                        if w_q > 0: 
                            term_details.append({
                                "Term": term,
                                "W_Query": f"{w_q:.4f}",
                                "W_Doc": f"{w_d:.4f}",
                                "Product": f"{product:.4f}"
                            })
                    
                    if term_details:
                        st.caption("Rincian Perkalian Bobot (Dot Product) untuk Kata Kunci:")
                        st.dataframe(pd.DataFrame(term_details), hide_index=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Dot Product (A)", f"{dot_prod_accum:.4f}")
                    with col2:
                        st.metric("Panjang Vektor |D| (B)", f"{d_vec_len:.4f}")
                    with col3:
                        denominator = q_vec_len * d_vec_len
                        final_res = dot_prod_accum / denominator if denominator != 0 else 0
                        st.metric("Similarity Score", f"{final_res:.4f}")
                    
                    st.divider()

            # TAB 4: RANKING AKHIR
            with tab4:
                st.subheader("Hasil Pencarian Terbaik")
                sorted_res = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                
                for i, (name, score) in enumerate(sorted_res, 1):
                    if score > 0:
                        st.success(f"Peringkat #{i}: {name} (Skor: {score:.4f})")
                    else:
                        st.warning(f"Peringkat #{i}: {name} (Skor: 0.0000 - Tidak Relevan)")

elif not uploaded_files:
    st.info("Silakan upload file dokumen di sidebar sebelah kiri untuk memulai.")