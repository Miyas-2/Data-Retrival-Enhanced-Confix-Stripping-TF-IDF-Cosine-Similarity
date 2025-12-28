from flask import Flask, render_template, request, jsonify, session
import os
import math
import pandas as pd
import numpy as np
import PyPDF2
import docx
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from werkzeug.utils import secure_filename
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Sastrawi tools (ECS Algorithm)
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()

# Global storage for documents (in production, use database)
corpus_data = {
    'documents': pd.DataFrame(columns=['filename', 'content', 'upload_date']),
    'processed': False,
    'vocabulary': [],
    'tf': None,
    'idf': None,
    'tfidf': None,
    'doc_tokens': {}
}

# ============= DOCUMENT READING FUNCTIONS (from main2.ipynb) =============
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ============= PREPROCESSING FUNCTIONS (Exact from main2.ipynb) =============
def preprocess(text, verbose=False):
    """
    Complete preprocessing pipeline using Sastrawi (ECS algorithm)
    Returns intermediate steps for educational visualization
    """
    # 1. Case Folding
    text_lower = text.lower()
    
    # 2. Cleaning (Remove numbers and punctuation)
    text_clean = re.sub(r'[^a-z\s]', '', text_lower)
    
    # 3. Stopword Removal
    text_stop = stopword_remover.remove(text_clean)
    
    # 4. Stemming (ECS)
    text_stemmed = stemmer.stem(text_stop)
    
    # 5. Tokenization
    tokens = text_stemmed.split()
    
    steps = {
        'original': text[:200] if len(text) > 200 else text,
        'case_folding': text_lower[:200] if len(text_lower) > 200 else text_lower,
        'cleaned': text_clean[:200] if len(text_clean) > 200 else text_clean,
        'stopword_removed': text_stop[:200] if len(text_stop) > 200 else text_stop,
        'stemmed': text_stemmed[:200] if len(text_stemmed) > 200 else text_stemmed,
        'tokens': tokens[:50],  # First 50 tokens for display
        'all_tokens': tokens
    }
    
    return steps

# ============= TF-IDF CALCULATION (Exact from main2.ipynb) =============
def build_tfidf_model(df_docs):
    """Build TF-IDF model exactly as in main2.ipynb"""
    global corpus_data
    
    # Preprocess all documents
    tokens_list = []
    doc_tokens = {}
    
    for index, row in df_docs.iterrows():
        processed = preprocess(row['content'])
        tokens_list.append(processed['all_tokens'])
        doc_tokens[row['filename']] = processed
    
    df_docs['tokens'] = tokens_list
    
    # 1. Create Vocabulary
    all_tokens = [token for sublist in tokens_list for token in sublist]
    vocabulary = sorted(list(set(all_tokens)))
    
    # 2. Term Frequency (TF) - Raw Count
    tf_data = []
    for doc_tokens_list in tokens_list:
        doc_tf_vector = []
        for term in vocabulary:
            doc_tf_vector.append(doc_tokens_list.count(term))
        tf_data.append(doc_tf_vector)
    
    df_tf = pd.DataFrame(tf_data, columns=vocabulary, index=df_docs['filename'])
    
    # 3. Inverse Document Frequency (IDF)
    # Formula: log10(Total Documents / Document Frequency of term)
    N = len(df_docs)
    idf_data = []
    
    for term in vocabulary:
        df_count = sum([1 for tokens in tokens_list if term in tokens])
        idf_val = math.log10(N / df_count) if df_count > 0 else 0
        idf_data.append(idf_val)
    
    df_idf = pd.DataFrame([idf_data], columns=vocabulary, index=['IDF'])
    
    # 4. TF-IDF Matrix
    df_tfidf = df_tf.mul(df_idf.iloc[0], axis=1)
    
    # Store in global corpus
    corpus_data['documents'] = df_docs
    corpus_data['vocabulary'] = vocabulary
    corpus_data['tf'] = df_tf
    corpus_data['idf'] = df_idf
    corpus_data['tfidf'] = df_tfidf
    corpus_data['doc_tokens'] = doc_tokens
    corpus_data['processed'] = True
    
    return {
        'vocab_size': len(vocabulary),
        'total_tokens': len(all_tokens),
        'num_docs': N
    }

# ============= QUERY PROCESSING & SIMILARITY (Exact from main2.ipynb) =============
def process_query(query):
    """Process query exactly as in main2.ipynb"""
    if not corpus_data['processed']:
        return None
    
    # Preprocess query
    query_steps = preprocess(query)
    query_tokens = query_steps['all_tokens']
    
    vocabulary = corpus_data['vocabulary']
    df_idf = corpus_data['idf']
    
    # Calculate Query TF
    query_tf = []
    for term in vocabulary:
        query_tf.append(query_tokens.count(term))
    
    # Calculate Query TF-IDF
    query_tfidf = np.array(query_tf) * df_idf.iloc[0].values
    
    return {
        'steps': query_steps,
        'vector': query_tfidf,
        'tokens': query_tokens
    }

def cosine_similarity(vec_a, vec_b):
    """Exact cosine similarity from main2.ipynb"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def search_documents(query):
    """Search exactly as in main2.ipynb"""
    if not corpus_data['processed']:
        return None
    
    query_data = process_query(query)
    if not query_data:
        return None
    
    query_vector = query_data['vector']
    df_tfidf = corpus_data['tfidf']
    
    # Calculate similarity with all documents
    results = []
    for filename, doc_vector in df_tfidf.iterrows():
        sim_score = cosine_similarity(query_vector, doc_vector.values)
        results.append({
            'filename': filename,
            'similarity': float(sim_score)
        })
    
    # Sort by similarity descending
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    return {
        'query_steps': query_data['steps'],
        'results': results
    }

# ============= ROUTES =============
@app.route('/')
def index():
    """Home page"""
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with stats"""
    stats = {
        'num_docs': len(corpus_data['documents']),
        'vocab_size': len(corpus_data['vocabulary']),
        'total_tokens': sum([len(tokens) for tokens in corpus_data['doc_tokens'].values()]) if corpus_data['doc_tokens'] else 0,
        'processed': corpus_data['processed']
    }
    
    doc_list = corpus_data['documents'].to_dict('records') if not corpus_data['documents'].empty else []
    
    return render_template('demo.html', stats=stats, documents=doc_list)

@app.route('/preprocessing')
def preprocessing():
    """Preprocessing explanation page"""
    return render_template('preprocessing.html')

@app.route('/tfidf')
def tfidf():
    """TF-IDF explanation page"""
    if not corpus_data['processed']:
        return render_template('tfidf_info.html', processed=False)
    
    # Sample data for display
    tf_sample = corpus_data['tf'].head(5).to_dict() if corpus_data['tf'] is not None else {}
    idf_sample = corpus_data['idf'].head(10).to_dict() if corpus_data['idf'] is not None else {}
    
    return render_template('tfidf_info.html', 
                         processed=True,
                         tf_sample=tf_sample,
                         idf_sample=idf_sample,
                         vocab_size=len(corpus_data['vocabulary']))

@app.route('/cosine')
def cosine():
    """Cosine similarity explanation page"""
    return render_template('similarity.html')

# ============= API ENDPOINTS =============
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload document endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use .txt, .pdf, or .docx'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Read file content
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'txt':
        content = read_txt(filepath)
    elif ext == 'pdf':
        content = read_pdf(filepath)
    elif ext == 'docx':
        content = read_docx(filepath)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    # Add to corpus
    new_doc = pd.DataFrame([{
        'filename': filename,
        'content': content,
        'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    corpus_data['documents'] = pd.concat([corpus_data['documents'], new_doc], ignore_index=True)
    corpus_data['processed'] = False  # Mark for reprocessing
    
    return jsonify({
        'success': True,
        'filename': filename,
        'size': len(content),
        'message': 'Document uploaded successfully'
    })

@app.route('/api/load_dataset', methods=['POST'])
def load_dataset():
    """Load documents from dataset folder"""
    base_path = request.json.get('path', 'dataset')
    
    if not os.path.exists(base_path):
        return jsonify({'error': f'Dataset path {base_path} does not exist'}), 400
    
    documents = []
    loaders = {
        'texts': ('.txt', read_txt),
        'pdf': ('.pdf', read_pdf),
        'doc': ('.docx', read_docx)
    }
    
    for folder, (ext, loader_func) in loaders.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue
        
        for filename in os.listdir(folder_path):
            if filename.endswith(ext):
                filepath = os.path.join(folder_path, filename)
                try:
                    content = loader_func(filepath)
                    documents.append({
                        'filename': filename,
                        'content': content,
                        'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    if not documents:
        return jsonify({'error': 'No documents found in dataset'}), 400
    
    corpus_data['documents'] = pd.DataFrame(documents)
    corpus_data['processed'] = False
    
    return jsonify({
        'success': True,
        'num_docs': len(documents),
        'message': f'Loaded {len(documents)} documents from dataset'
    })

@app.route('/api/process', methods=['POST'])
def process_corpus():
    """Process all documents and build TF-IDF model"""
    if corpus_data['documents'].empty:
        return jsonify({'error': 'No documents to process'}), 400
    
    try:
        stats = build_tfidf_model(corpus_data['documents'])
        return jsonify({
            'success': True,
            'stats': stats,
            'message': 'Corpus processed successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint"""
    query = request.json.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    if not corpus_data['processed']:
        return jsonify({'error': 'Corpus not processed. Please process documents first.'}), 400
    
    try:
        result = search_documents(query)
        if not result:
            return jsonify({'error': 'Search failed'}), 500
        
        return jsonify({
            'success': True,
            'query': query,
            'query_steps': result['query_steps'],
            'results': result['results'],
            'num_results': len(result['results'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preprocess_demo', methods=['POST'])
def preprocess_demo():
    """Demo preprocessing for a text snippet"""
    text = request.json.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    try:
        steps = preprocess(text, verbose=True)
        return jsonify({
            'success': True,
            'steps': {
                'original': steps['original'],
                'case_folding': steps['case_folding'],
                'cleaned': steps['cleaned'],
                'stopword_removed': steps['stopword_removed'],
                'stemmed': steps['stemmed'],
                'tokens': steps['tokens']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get corpus statistics"""
    stats = {
        'num_docs': len(corpus_data['documents']),
        'vocab_size': len(corpus_data['vocabulary']),
        'processed': corpus_data['processed'],
        'documents': corpus_data['documents'][['filename', 'upload_date']].to_dict('records') if not corpus_data['documents'].empty else []
    }
    return jsonify(stats)

@app.route('/api/document/<filename>', methods=['GET'])
def get_document(filename):
    """Get document details"""
    if corpus_data['documents'].empty:
        return jsonify({'error': 'No documents in corpus'}), 404
    
    doc = corpus_data['documents'][corpus_data['documents']['filename'] == filename]
    if doc.empty:
        return jsonify({'error': 'Document not found'}), 404
    
    doc_data = doc.iloc[0].to_dict()
    
    # Add preprocessing info if available
    if filename in corpus_data['doc_tokens']:
        doc_data['preprocessing'] = {
            'tokens': corpus_data['doc_tokens'][filename]['tokens'],
            'num_tokens': len(corpus_data['doc_tokens'][filename]['all_tokens'])
        }
    
    return jsonify(doc_data)

if __name__ == '__main__':
    # Load initial dataset on startup
    if os.path.exists('dataset - Copy'):
        print("Loading initial dataset...")
        try:
            documents = []
            loaders = {
                'texts': ('.txt', read_txt),
                'pdf': ('.pdf', read_pdf),
                'doc': ('.docx', read_docx)
            }
            
            for folder, (ext, loader_func) in loaders.items():
                folder_path = os.path.join('dataset - Copy', folder)
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.endswith(ext):
                            filepath = os.path.join(folder_path, filename)
                            try:
                                content = loader_func(filepath)
                                documents.append({
                                    'filename': filename,
                                    'content': content,
                                    'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                                print(f"Loaded: {filename}")
                            except Exception as e:
                                print(f"Error loading {filename}: {e}")
            
            if documents:
                corpus_data['documents'] = pd.DataFrame(documents)
                print(f"\nLoaded {len(documents)} documents. Processing...")
                stats = build_tfidf_model(corpus_data['documents'])
                print(f"✅ Processed! Vocabulary: {stats['vocab_size']} terms, Total tokens: {stats['total_tokens']}")
        except Exception as e:
            print(f"Error during initial load: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)