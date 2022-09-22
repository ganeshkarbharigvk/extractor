import pandas as pd
import streamlit as st
from keybert import KeyBERT
import yake
from keyphrase_vectorizers import KeyphraseCountVectorizer

@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=True)
def load_model():
  model = KeyBERT("AI-Growth-Lab/PatentSBERTa")
  return model
  
model = load_model()
st.title("Patent Text Extractor")
placeholder = st.empty()
text_input = placeholder.text_area("Paste or write text", height=300)
button = st.button("Extract Keywords")
top_n = st.sidebar.slider("Select a number of keywords", 1, 10, 50,20)
min_ngram = st.sidebar.number_input("Minimum number of words in each keyword", 1, 5, 1, 1)
max_ngram = st.sidebar.number_input("Maximum number of words in each keyword", min_ngram, 5, 3, step=1)
st.sidebar.code(f"ngram_range=({min_ngram}, {max_ngram})")

params = {"docs": text_input, "top_n": top_n, "keyphrase_ngram_range": (min_ngram, max_ngram), "stop_words": 'english',"vectorizer":KeyphraseCountVectorizer()}

add_diversity = st.sidebar.checkbox("Adjust diversity of keywords")

if add_diversity:
  method = st.sidebar.selectbox("Select a method", ("Max Sum Similarity", "Maximal Marginal Relevance"))
  if method == "Max Sum Similarity":
        nr_candidates = st.sidebar.slider("nr_candidates", 20, 50, 20, 2)
        params["use_maxsum"] = True
        params["nr_candidates"] = nr_candidates

  elif method == "Maximal Marginal Relevance":
        diversity = st.sidebar.slider("diversity", 0.1, 1.0, 0.6, 0.01)
        params["use_mmr"] = True
        params["diversity"] = diversity

keywords = model.extract_keywords(**params)

if keywords != []:
    st.info("Extracted keywords")
    keywords = pd.DataFrame(keywords, columns=["Keyword", "Score"])
    st.table(keywords)

st.markdown("## **🎈 Check & download results **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

st.header("")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Score"])
    .sort_values(by="Score", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Score",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Score": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
