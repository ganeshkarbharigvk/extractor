import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
# For Flair (Keybert)
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
# For download buttons
from functionforDownloadButtons import download_button
import os
import json

st.set_page_config(
    page_title="Patent Keyword Extractor",
    page_icon="🔑 ",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 2])

with c30:
    # st.image("logo.png", width=500)
    st.title("Patent Keyword Extractor")
    st.header("")



st.markdown("")
st.markdown("## **📌 Paste Patent Text **")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Model",
            ["Patent-Key-BERT(Default)"],
            help="At present, you can choose  AI-Growth-Lab/PatentSBERTa to embed your text. More to come!",
        )

        if ModelType == "Default (AI-Growth-Lab/PatentSBERTa)":
            # kw_model = KeyBERT(model=AI-Growth-Lab/PatentSBERTa)

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT(model=roberta)

            kw_model = load_model()

        else:
            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("AI-Growth-Lab/PatentSBERTa")

            kw_model = load_model()


        top_N = st.slider(
            "# of results",
            min_value=5,
            max_value=50,
            value=50,
            help="You can choose the number of keywords/keyphrases to display. Between 5 and 50, default number is 50.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=3,
            help="""The minimum value for the ngram range.
*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 3) or higher depending on the number of words you would like in the resulting keyphrases.""",
            # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=3,
            min_value=1,
            max_value=3,
            help="""The maximum value for the keyphrase_ngram_range.
*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.
            
Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.
""",
        )

    with c2:
        doc = st.text_area(
            "Paste your text below (max 4000 words)",
            height=410,
        )

        MAX_WORDS = 4000
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 4000 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Extract Keywords!")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()

keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    diversity=Diversity,
    vectorizer=KeyphraseCountVectorizer(),
)

st.markdown("## **Keyword Extraction Results with Score **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([1.5, 1.5, 1.5, 1.5, 1.5])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

st.header("")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
