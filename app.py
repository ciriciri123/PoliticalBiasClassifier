import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from spacy import displacy 
from gensim.models import Word2Vec
from newspaper import Article 
import io
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Political Bias Detector", page_icon="🏛️", layout="wide")
class NGramLanguageModel:
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.vectorizer = TfidfVectorizer(ngram_range=(n1, n2), max_features=3000)

    def fit_transform(self, corpus):
        return self.vectorizer.fit_transform(corpus)

    def transform(self, corpus):
        return self.vectorizer.transform(corpus)

@st.cache_resource
def load_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nlp_model = spacy.load("en_core_web_sm")
    
    models = {
        "Random Forest": pickle.load(open("model/rf_model.pkl", "rb")),
        "Logistic Regression": pickle.load(open("model/lr_model.pkl", "rb")),
        "Support Vector Machine": pickle.load(open("model/svm_model.pkl", "rb")),
        "Naive Bayes": pickle.load(open("model/nb_model.pkl", "rb")) 
    }
    scalers = {
        "Logistic Regression": pickle.load(open("model/lr_scaler.pkl", "rb")),
        "Support Vector Machine": pickle.load(open("model/svm_scaler.pkl", "rb"))
    }
    
    ngram_model = pickle.load(open("model/ngram_model.pkl", "rb"))
    w2v_model = pickle.load(open("model/w2v_model.pkl", "rb"))
    ner_vectorizers = pickle.load(open("model/ner_vectorizers.pkl", "rb"))
    label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
    
    return nlp_model, models, scalers, ngram_model, w2v_model, ner_vectorizers, label_encoder

nlp, models, scalers, ngram_model, w2v_model, ner_vectorizers, label_encoder = load_resources()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(cleaned_tokens)

def get_mean_vector(w2v_model, words):
    words = [word for word in words if word in w2v_model.wv.index_to_key]
    if len(words) >= 1:
        return np.mean(w2v_model.wv[words], axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

NER_TYPES = ['ORG', 'PERSON', 'GPE', 'NORP']

def get_entity_dictionary(text):
    doc = nlp(text)
    ent_dict = {t: [] for t in NER_TYPES}
    for ent in doc.ents:
        if ent.label_ in NER_TYPES:
            token = ent.text.strip().lower().replace(" ", "_")
            ent_dict[ent.label_].append(token)
    return {k: " ".join(v) for k, v in ent_dict.items()}

def extract_ner_features(text, ner_vectorizers):
    ent_dict = get_entity_dictionary(text)
    parts = []
    for ner_type in NER_TYPES:
        strings = [ent_dict[ner_type]]
        vec = ner_vectorizers[ner_type]
        mat = vec.transform(strings).toarray()
        parts.append(mat)
    return np.hstack(parts)

def run_analysis(text_input, selected_model, compare_all=False):
    with st.spinner('Processing NLP Pipeline...'):
        try:
            cleaned_str = clean_text(text_input)
            tokens = cleaned_str.split()
            
            w2v_feat = get_mean_vector(w2v_model, tokens).reshape(1, -1)
            tfidf_feat = ngram_model.transform([cleaned_str]).toarray()
            ner_feat = extract_ner_features(text_input, ner_vectorizers) 
            
            combined_features = np.hstack((w2v_feat, tfidf_feat, ner_feat)) 
            
            report_lines = [
                "POLITICAL BIAS NLP ANALYSIS REPORT",
                "\n",
                "ANALYZED TEXT (TITLE + SOURCE):",
                text_input + "\n",
                "EXTRACTED NAMED ENTITIES"
            ]
            
            ent_dict = get_entity_dictionary(text_input)
            for k, v in ent_dict.items():
                report_lines.append(f"{k}: {v if v else 'None detected'}")
            report_lines.append("\nMODEL PREDICTIONS")

            st.divider()

            if compare_all:
                st.subheader("🤖 Multi-Model Consensus Board")
                comparison_data = []
                
                for name, model in models.items():
                    if name == "Naive Bayes":
                        min_val = np.min(combined_features)
                        feats = combined_features - min_val if min_val < 0 else combined_features
                    else:
                        feats = scalers[name].transform(combined_features) if name in scalers else combined_features
                    
                    pred_idx = model.predict(feats)[0]
                    pred_label = label_encoder.inverse_transform([pred_idx])[0]
                    
                    confidence = "N/A"
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(feats)[0]
                        confidence = f"{np.max(probs) * 100:.1f}%"
                        
                    comparison_data.append({
                        "Model": name,
                        "Spectrum": pred_label.upper(),
                        "Confidence": confidence
                    })
                    report_lines.append(f"- {name}: {pred_label.upper()} (Confidence: {confidence})")
                
                st.table(pd.DataFrame(comparison_data))
                
            else:
                st.subheader(f"📊 Results: {selected_model}")
                
                active_model = models[selected_model]
                if selected_model == "Naive Bayes":
                    min_val = np.min(combined_features)
                    feats = combined_features - min_val if min_val < 0 else combined_features
                else:
                    feats = scalers[selected_model].transform(combined_features) if selected_model in scalers else combined_features
                
                pred_idx = active_model.predict(feats)[0]
                predicted_label = label_encoder.inverse_transform([pred_idx])[0]
                
                st.success(f"Predicted Bias Spectrum: **{predicted_label.upper()}**")
                report_lines.append(f"- {selected_model}: {predicted_label.upper()}")
                
                if hasattr(active_model, "predict_proba"):
                    probabilities = active_model.predict_proba(feats)[0]
                    classes = label_encoder.classes_
                    prob_df = pd.DataFrame({"Spectrum": classes, "Confidence": probabilities}).set_index("Spectrum")
                    st.bar_chart(prob_df)
                    report_lines.append(f"Confidence: {np.max(probabilities)*100:.1f}%")

            st.subheader("🔍 Context & Entity Analysis")
            st.markdown("The highlighted entities below (People, Organizations, Locations) influenced the model's decision.")
            
            doc = nlp(text_input)
            options = {"ents": NER_TYPES, "colors": {"PERSON": "#ffcccb", "ORG": "#add8e6", "GPE": "#90ee90", "NORP": "#f0e68c"}}
            html_highlighter = displacy.render(doc, style="ent", options=options)
            st.markdown(html_highlighter, unsafe_allow_html=True)
            
            st.divider()
            st.subheader("💾 Export Findings")
            final_report_text = "\n".join(report_lines)
            st.download_button(
                label="Download Analytical Report (.txt)",
                data=final_report_text,
                file_name="bias_analysis_report.txt",
                mime="text/plain",
                type="primary"
            )

        except Exception as e:
            st.error(f"Pipeline Error: {e}")

st.title("Political Bias Detection")
st.markdown("Analyze news articles to detect latent political bias using classical NLP algorithms.")

st.sidebar.header("Configuration")
compare_all = st.sidebar.checkbox("Compare All Models", value=False)
selected_model = st.sidebar.selectbox("Active Model:", ["Random Forest", "Logistic Regression", "Support Vector Machine", "Naive Bayes"], disabled=compare_all)

if compare_all:
    st.sidebar.info("All 4 models will run simultaneously to provide a consensus audit.")

tab1, tab2, tab3 = st.tabs(["📝 Input Title & Source", "🔗 Analyze URL", "📁 Upload File"])

with tab1:
    st.info("Masukkan Judul Artikel beserta nama Sumber Medianya.")
    col1, col2 = st.columns([3, 1])
    with col1:
        input_title = st.text_input("Article Title:", placeholder="Contoh: New Tax Policy Announced")
    with col2:
        input_source = st.text_input("Media Source:", placeholder="Contoh: usatoday")
        
    if st.button("Analyze Article", type="primary", key="btn_text"):
        if input_title.strip() and input_source.strip():
            combined_input = f"{input_title.strip()} {input_source.strip()}"
            run_analysis(combined_input, selected_model, compare_all)
        else:
            st.warning("Mohon isi kedua kolom (Judul dan Sumber Media).")

with tab2:
    st.info("Paste link artikel. Sistem akan otomatis mengekstrak Judul dan Sumber Media dari URL.")
    url_input = st.text_input("Article URL:", placeholder="https://www.cnn.com/2026/01/01/politics/...")
    if st.button("Scrape & Analyze URL", type="primary", key="btn_url"):
        if url_input.strip():
            try:
                with st.spinner("Downloading article from web..."):
                    article = Article(url_input)
                    article.download()
                    article.parse()
                    scraped_title = article.title
                    domain = urlparse(url_input).netloc
                    domain_parts = domain.replace("www.", "").split(".")
                    scraped_source = domain_parts[0] if len(domain_parts) > 0 else "unknown"
                    
                st.success(f"Successfully scraped: **{scraped_title}** (Source: {scraped_source})")
                combined_input = f"{scraped_title} {scraped_source}"
                run_analysis(combined_input, selected_model, compare_all)
            except Exception as e:
                st.error(f"Could not scrape that URL. Some websites block automated readers. Error: {e}")
        else:
            st.warning("Please enter a valid URL.")

with tab3:
    st.info("Upload a text file (.txt) containing the Article Title and Source on a single line.")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if st.button("Analyze File", type="primary", key="btn_file"):
        if uploaded_file is not None:
            string_data = uploaded_file.getvalue().decode("utf-8").strip()
            run_analysis(string_data, selected_model, compare_all)
        else:
            st.warning("Please upload a file first.")