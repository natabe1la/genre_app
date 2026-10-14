"""
Movie Genre Predictor (Multi-Label SVM)  -  matches CST9_Phase_2_final.ipynb
----------------------------------------------------------------------------
Pipeline: preprocessing -> TF-IDF (1-2 grams, max_df=0.5, min_df=1)
          -> One-vs-Rest linear SVM (class_weight="balanced")
          -> 80/20 split, random_state=42

The TMDB dataset is used ONLY for training. At prediction time you type a brand-new
overview that the model has never seen.

Run:   streamlit run app.py
"""

import ast
import html
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

# ----------------------------------------------------------------
# Settings (same as the notebook)
# ----------------------------------------------------------------
DATA_PATH = Path("tmdb_5000_movies.csv")      # keep the CSV next to this file
MODEL_PATH = Path("genre_svm_model.joblib")   # created automatically after first training
MODEL_VERSION = 3                              # bump to force a retrain of old saved models
RANDOM_STATE = 42
TEST_SIZE = 0.20
SVM_C = 1   # notebook baseline. Change to your GridSearchCV best C if different.

# ----------------------------------------------------------------
# Text preprocessing (identical to the notebook)
# ----------------------------------------------------------------
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

STEMMER = SnowballStemmer("english")


def tokenize_clean(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 2 and t not in STOP_WORDS]


def preprocess_text(text):
    return " ".join(STEMMER.stem(t) for t in tokenize_clean(text))


def parse_genres(value):
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value) if not isinstance(value, list) else value
    except (ValueError, SyntaxError):
        return []
    names = []
    for g in parsed if isinstance(parsed, list) else []:
        if isinstance(g, dict) and g.get("name"):
            names.append(str(g["name"]))
        elif isinstance(g, str):
            names.append(g)
    return names


# ----------------------------------------------------------------
# Training (runs once, then the model is saved to disk)
# ----------------------------------------------------------------
def load_dataset():
    if not DATA_PATH.exists():
        st.error(f"Could not find **{DATA_PATH}**. Put the TMDB CSV in the same folder as app.py and refresh.")
        st.stop()
    try:  # same read options as the final notebook
        movies = pd.read_csv(DATA_PATH, quotechar='"', engine="python", on_bad_lines="warn")
        if "overview" not in movies.columns or "genres" not in movies.columns:
            raise ValueError("missing columns")
    except Exception:
        movies = pd.read_csv(DATA_PATH)
    return movies


def train_model():
    movies = load_dataset()
    df = movies[[c for c in ["id", "overview", "genres"] if c in movies.columns]].copy()
    df = df.drop_duplicates(subset=["id"]) if "id" in df.columns else df.drop_duplicates()
    df = df.dropna(subset=["overview", "genres"])
    df["overview"] = df["overview"].astype(str)
    df = df[df["overview"].str.strip() != ""]
    df["genre_list"] = df["genres"].apply(parse_genres)
    df = df[df["genre_list"].apply(len) > 0].copy()
    df["clean"] = df["overview"].apply(preprocess_text)
    df = df[df["clean"].str.strip() != ""].reset_index(drop=True)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["genre_list"])
    genres = list(mlb.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_df=0.5, min_df=1, ngram_range=(1, 2))),
        ("svm", OneVsRestClassifier(SVC(kernel="linear", C=SVM_C, class_weight="balanced"))),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),  # subset accuracy, as in the notebook
        "Micro Precision": precision_score(y_test, pred, average="micro", zero_division=0),
        "Micro Recall": recall_score(y_test, pred, average="micro", zero_division=0),
        "Micro F1": f1_score(y_test, pred, average="micro", zero_division=0),
        "Hamming Loss": hamming_loss(y_test, pred),
        "Train size": len(X_train),
        "Test size": len(X_test),
        "Features": len(pipeline.named_steps["tfidf"].get_feature_names_out()),
    }
    per_genre = pd.DataFrame({
        "Genre": genres,
        "F1": f1_score(y_test, pred, average=None, zero_division=0),
        "Training movies": y_train.sum(axis=0),
    }).sort_values("F1")
    bundle = {"version": MODEL_VERSION, "pipeline": pipeline, "genres": genres,
              "metrics": metrics, "per_genre": per_genre}
    joblib.dump(bundle, MODEL_PATH)
    return bundle


@st.cache_resource(show_spinner=False)
def get_model():
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if bundle.get("version") == MODEL_VERSION:
                return bundle
        except Exception:
            pass
    with st.spinner("Training the SVM on the dataset (first run only, a few minutes)..."):
        return train_model()


# ----------------------------------------------------------------
# Prediction + explanation
# ----------------------------------------------------------------
def build_stem_lookup(text):
    lookup = {}
    for tok in tokenize_clean(text):
        lookup.setdefault(STEMMER.stem(tok), tok)
    return lookup


def pretty_feature(name, lookup):
    return " ".join(lookup.get(part, part) for part in name.split())


def predict(bundle, text):
    pipe, genres = bundle["pipeline"], bundle["genres"]
    clean = preprocess_text(text)
    if not clean.strip():
        return None
    tfidf = pipe.named_steps["tfidf"]
    ovr = pipe.named_steps["svm"]
    vec = tfidf.transform([clean])
    if vec.nnz == 0:   # none of the words are known to the model
        return None

    scores = ovr.decision_function(vec).ravel()

    feats = tfidf.get_feature_names_out()
    idx, vals = vec.indices, vec.data
    lookup = build_stem_lookup(text)
    raw_names = [feats[i] for i in idx]
    names = [pretty_feature(n, lookup) for n in raw_names]

    contrib = {}
    for g in range(len(genres)):
        est = ovr.estimators_[g]
        if hasattr(est, "coef_"):
            coef = est.coef_
            coef = coef.toarray().ravel() if hasattr(coef, "toarray") else np.ravel(coef)
            contrib[g] = vals * coef[idx]
    return {"scores": scores,
            "names": names, "raw_names": raw_names, "contrib": contrib}


# ----------------------------------------------------------------
# Charts (white card style so they look good in light and dark themes)
# ----------------------------------------------------------------
GREEN, GREY, RED, INK = "#1f9d6b", "#c9ced6", "#e0564f", "#2b2f36"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d0d4da", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eceff3", linewidth=1)
    ax.set_axisbelow(True)


def chart_scores(genres, scores, chosen, threshold=0.0):
    order = np.argsort(scores)
    fig, ax = plt.subplots(figsize=(7, max(4.2, 0.3 * len(genres))), facecolor="white")
    colors = [GREEN if i in chosen else GREY for i in order]
    bars = ax.barh([genres[i] for i in order], scores[order], color=colors, height=0.7)
    ax.axvline(0, color=INK, linewidth=1.2)
    if threshold != 0:
        ax.axvline(threshold, color="#4f46e5", linewidth=1.5, linestyle="--")
    for b, i in zip(bars, order):
        if i in chosen:
            ax.text(b.get_width() + 0.02, b.get_y() + b.get_height() / 2, f"{scores[i]:+.2f}",
                    va="center", fontsize=9, color=GREEN, fontweight="bold")
    ax.set_xlabel("SVM score    ←  “no”   |   “yes”  →")
    style_ax(ax)
    fig.tight_layout()
    return fig


def chart_words(genre, names, contrib, top_n=8):
    order = np.argsort(contrib)
    pos = [o for o in order if contrib[o] > 0][-top_n:]
    neg = [o for o in order if contrib[o] < 0][:3]
    shown = neg + pos
    if not shown:
        return None
    fig, ax = plt.subplots(figsize=(6.2, max(2.4, 0.38 * len(shown) + 0.8)), facecolor="white")
    ax.barh([names[o] for o in shown], [contrib[o] for o in shown],
            color=[GREEN if contrib[o] > 0 else RED for o in shown], height=0.65)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_xlabel("← pushes away from   |   pushes toward →")
    ax.set_title(f"Evidence for “{genre}”", loc="left", fontweight="bold")
    style_ax(ax)
    fig.tight_layout()
    return fig


def chart_heatmap(names, genres_sel, matrix, top_n=10):
    strength = np.abs(matrix).max(axis=1)
    rows = np.argsort(-strength)[:top_n]
    data = matrix[rows]
    lim = max(np.abs(data).max(), 1e-9)
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(genres_sel) + 2.5), 0.42 * len(rows) + 1.6),
                           facecolor="white")
    im = ax.imshow(data, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(genres_sel)), genres_sel, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), [names[r] for r in rows])
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, f"{data[r, c]:+.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="push toward (+) / away (−)")
    fig.tight_layout()
    return fig


def chart_reliability(per_genre):
    fig, ax = plt.subplots(figsize=(7, max(4.2, 0.3 * len(per_genre))), facecolor="white")
    colors = [GREEN if f >= 0.5 else ("#e8a33d" if f >= 0.3 else RED) for f in per_genre["F1"]]
    ax.barh(per_genre["Genre"], per_genre["F1"], color=colors, height=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score on unseen test movies (higher = more reliable)")
    style_ax(ax)
    fig.tight_layout()
    return fig


def highlight_html(text, genre_idx, result):
    """Colour each word of the user's text by how much it supports the genre."""
    c = result["contrib"].get(genre_idx)
    if c is None:
        return html.escape(text)
    unigram = {n: v for n, v in zip(result["raw_names"], c) if " " not in n}
    top = max(np.abs(c).max(), 1e-9)
    out = []
    for piece in re.findall(r"[A-Za-z']+|[^A-Za-z']+", text):
        word = piece.lower()
        stem = STEMMER.stem(word) if piece[0].isalpha() and len(word) > 2 and word not in STOP_WORDS else None
        v = unigram.get(stem)
        if v is None or abs(v) < 0.02 * top:
            out.append(html.escape(piece))
        else:
            a = min(0.15 + 0.75 * abs(v) / top, 0.9)
            rgb = "31,157,107" if v > 0 else "224,86,79"
            out.append(f'<span style="background:rgba({rgb},{a:.2f});padding:1px 3px;'
                       f'border-radius:4px">{html.escape(piece)}</span>')
    return "".join(out)


def strength_label(score):
    if score >= 0.6:
        return "Strong match"
    if score >= 0.25:
        return "Good match"
    return "Borderline"


# ----------------------------------------------------------------
# UI
# ----------------------------------------------------------------
st.set_page_config(page_title="Movie Genre Predictor", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);
       padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size: 2.1rem; color: white;}
.hero p {margin:.4rem 0 0 0; opacity:.92; font-size:1.02rem;}
.chip {display:inline-block; padding:10px 18px; margin:4px 8px 4px 0; border-radius: 14px;
       background: linear-gradient(120deg,#1f9d6b,#16a085); color:white; font-weight:700; font-size:1.05rem;}
.chip small {display:block; font-weight:500; opacity:.9; font-size:.74rem;}
.chip.alt {background: linear-gradient(120deg,#6b7280,#4b5563);}
.stepno {display:inline-block; width:28px; height:28px; line-height:28px; text-align:center;
         border-radius:50%; background:#4f46e5; color:white; font-weight:700; margin-right:8px;}
.explain {background: rgba(79,70,229,.08); border-left: 4px solid #4f46e5; padding: 10px 14px;
          border-radius: 6px; font-size:.93rem; margin: 6px 0 10px 0;}
.hl {line-height: 2.1; font-size: 1.05rem; padding: 12px 14px; border-radius: 10px;
     background: white; color: #2b2f36; border: 1px solid #e5e7eb;}
</style>
<div class="hero">
  <h1>🎬 Movie Genre Predictor</h1>
  <p>Paste a movie overview. A multi-label Support Vector Machine predicts its genres and shows you exactly why.</p>
</div>
""", unsafe_allow_html=True)

bundle = get_model()
genres = bundle["genres"]
m = bundle["metrics"]

with st.sidebar:
    st.header("📊 Model scores")
    st.caption("Measured on the 20% of movies the model never trained on.")
    st.metric("Accuracy (exact match)", f"{m['Accuracy']:.3f}",
              help="Subset accuracy: a movie counts as correct only if EVERY one of its genres is predicted exactly, with none missing or extra. This is a strict measure for multi-label tasks, so it is usually much lower than the other scores. Micro F1 is a fairer overall measure.")
    st.metric("Micro F1", f"{m['Micro F1']:.3f}",
              help="Balance of precision and recall across all genre decisions.")
    c1, c2 = st.columns(2)
    c1.metric("Precision", f"{m['Micro Precision']:.3f}",
              help="Of the genres the model predicted, how many were right.")
    c2.metric("Recall", f"{m['Micro Recall']:.3f}",
              help="Of the true genres, how many the model found.")
    st.metric("Hamming loss", f"{m['Hamming Loss']:.3f}",
              help="Share of individual genre yes/no decisions that were wrong. Lower is better.")
    st.caption(f"Trained on {m['Train size']:,} movies · tested on {m['Test size']:,} · {m['Features']:,} text features")
    st.divider()
    if st.button("🔄 Retrain model from CSV"):
        MODEL_PATH.unlink(missing_ok=True)
        st.cache_resource.clear()
        st.session_state.pop("result", None)
        st.rerun()

tab_predict, tab_model = st.tabs(["🔮 Predict", "🧠 About the model"])

# ---------------- Predict tab ----------------
with tab_predict:
    EXAMPLES = {
        "✍️ Write my own": "",
        "🚀 Space adventure": "A crew of astronauts travels through a wormhole to find a new planet for humanity, but must battle hostile aliens and the vastness of space.",
        "💕 Romantic comedy": "Two coworkers who can't stand each other are forced to pretend they are dating, and soon discover that their fake romance is becoming real.",
        "👻 Horror": "A family moves into an old house and soon realizes that a terrifying demon haunts the halls at night, hunting them one by one.",
        "🔍 Crime thriller": "A detective hunts a serial killer across the city, racing against time as the murderer leaves clues and the police close in.",
    }
    choice = st.selectbox("Start from an example, or write your own", list(EXAMPLES))
    text = st.text_area("Movie overview", value=EXAMPLES[choice], height=140,
                        placeholder="Type or paste a movie overview here (a couple of sentences works best)...",
                        key=f"text_{choice}")

    threshold = st.slider(
        "Sensitivity (score cut-off)", min_value=-1.0, max_value=0.5, value=0.0, step=0.05, key="threshold",
        help="A genre is shown when its SVM score is above this number. 0.00 is exactly how your notebook's SVM decides. "
             "Lower it to see more possible genres, raise it to be stricter.")
    st.caption("0.00 = the notebook's default. Move left to reveal more genres the model finds plausible, right for only its most confident ones.")

    if st.button("🔮 Predict genre", type="primary"):
        if len(text.split()) < 5:
            st.warning("Please enter a longer overview (at least a few words).")
        else:
            res = predict(bundle, text)
            if res is None:
                st.session_state.pop("result", None)
                st.warning("The model didn't recognise any words in that text. Try a more descriptive overview in English.")
            else:
                st.session_state["result"] = {"res": res, "text": text}

    if "result" in st.session_state:
        res, shown_text = st.session_state["result"]["res"], st.session_state["result"]["text"]
        scores = res["scores"]
        chosen = sorted([i for i in range(len(genres)) if scores[i] > threshold], key=lambda i: -scores[i])
        fallback = not chosen
        if fallback:
            chosen = [int(np.argmax(scores))]
        f1_lookup = dict(zip(bundle["per_genre"]["Genre"], bundle["per_genre"]["F1"]))

        # --- Result chips
        st.markdown("### Predicted genres")
        chips = "".join(
            f'<span class="chip">{html.escape(genres[i])}<small>{strength_label(scores[i])} · score {scores[i]:+.2f}</small></span>'
            for i in chosen)
        st.markdown(chips, unsafe_allow_html=True)
        if fallback:
            st.info("No genre passed the cut-off, so this is only the **closest** match. Treat it with caution, or lower the sensitivity slider.")
        top = chosen[0]
        st.caption(f"The model's usual reliability for “{genres[top]}” on unseen movies: "
                   f"F1 = {f1_lookup.get(genres[top], 0):.2f}.")

        # --- Step 1: score chart
        with st.container(border=True):
            st.markdown('<span class="stepno">1</span><b>How strongly did each genre score?</b>', unsafe_allow_html=True)
            st.markdown(
                '<div class="explain">The SVM runs one separate yes/no test for <b>every</b> genre. '
                'A bar to the <b>right of the black line</b> means “yes, this genre fits”; the farther right, the more confident. '
                'If you moved the sensitivity slider, the <b style="color:#4f46e5">dashed purple line</b> is your cut-off and genres right of it are shown. '
                'Because each genre is tested independently, a movie can get several genres at once (that is what “multi-label” means). '
                'Grey bars are genres the model rejected.</div>', unsafe_allow_html=True)
            st.pyplot(chart_scores(genres, scores, set(chosen), threshold))
            runner = [i for i in np.argsort(-scores) if i not in chosen][:1]
            msg = f"**{genres[chosen[0]]}** scored highest ({scores[chosen[0]]:+.2f})."
            if len(chosen) > 1:
                msg += " It was joined by " + ", ".join(f"**{genres[i]}** ({scores[i]:+.2f})" for i in chosen[1:]) + "."
            if runner:
                msg += f" The closest rejected genre was {genres[runner[0]]} ({scores[runner[0]]:+.2f})."
            st.markdown(msg)

        # --- Step 2: word evidence
        with st.container(border=True):
            st.markdown('<span class="stepno">2</span><b>Which words convinced the model?</b>', unsafe_allow_html=True)
            st.markdown(
                '<div class="explain">For each predicted genre, these bars show the words (or two-word phrases) from <b>your text</b> '
                'that mattered most. <b style="color:#1f9d6b">Green</b> = the word is typical of that genre in the training movies and pushed the model toward it. '
                '<b style="color:#e0564f">Red</b> = the word is more typical of other genres and pulled away from it. '
                'Longer bars mean stronger influence.</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for k, g in enumerate(chosen):
                if g not in res["contrib"]:
                    continue
                c = res["contrib"][g]
                fig = chart_words(genres[g], res["names"], c)
                with cols[k % 2]:
                    if fig:
                        st.pyplot(fig)
                        best = [res["names"][o] for o in np.argsort(-c)[:3] if c[o] > 0]
                        if best:
                            st.caption(f"**{genres[g]}** was chosen mainly because of: " + ", ".join(f"“{w}”" for w in best) + ".")
                    else:
                        st.caption(f"No single word stood out for {genres[g]}.")

        # --- Step 3: highlighted text
        with st.container(border=True):
            st.markdown('<span class="stepno">3</span><b>See it in your own text</b>', unsafe_allow_html=True)
            st.markdown(
                '<div class="explain">The same evidence, painted onto your overview. Pick a genre to see which words '
                'supported it (<b style="color:#1f9d6b">green</b>) or worked against it (<b style="color:#e0564f">red</b>). '
                'Darker colour = stronger influence. Uncoloured words did not affect this genre. '
                'Common filler words like “the” and “and” are ignored by the model.</div>', unsafe_allow_html=True)
            pick = st.selectbox("Highlight words for:", [genres[i] for i in chosen], key="hl_pick")
            st.markdown(f'<div class="hl">{highlight_html(shown_text, genres.index(pick), res)}</div>',
                        unsafe_allow_html=True)

        # --- Step 4: heatmap (only when several genres)
        usable = [g for g in chosen if g in res["contrib"]]
        if len(usable) >= 2:
            with st.container(border=True):
                st.markdown('<span class="stepno">4</span><b>Why more than one genre?</b>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="explain">Each row is a word from your text, each column a predicted genre. '
                    'Green cells push toward that genre, red cells push away. '
                    'When a row is green in <b>several columns</b>, that word is shared evidence, which is exactly why the movie '
                    'landed in more than one genre. A word that is green in just one column is what separates that genre from the others.</div>',
                    unsafe_allow_html=True)
                matrix = np.column_stack([res["contrib"][g] for g in usable])
                st.pyplot(chart_heatmap(res["names"], [genres[g] for g in usable], matrix))

# ---------------- About tab ----------------
with tab_model:
    st.markdown("### How this model works")
    st.markdown(
        "1. **Clean the text**: lowercase, remove symbols, drop very short words and stopwords, reduce words to their stems (e.g. “hunting” → “hunt”).\n"
        "2. **TF-IDF**: turn each overview into numbers that reward distinctive words and word pairs.\n"
        "3. **One-vs-Rest linear SVM**: train one yes/no classifier per genre, so a movie can receive several genres.\n"
        "4. **Predict**: your new overview goes through the same steps, and each genre classifier votes.\n\n"
        "The dataset (TMDB 5000) is used only for training. Your input is never part of it.")
    with st.container(border=True):
        st.markdown('<b>How reliable is the model for each genre?</b>', unsafe_allow_html=True)
        st.markdown(
            '<div class="explain">Each bar is the F1 score for one genre on movies the model never saw during training '
            '(<b style="color:#1f9d6b">green</b> ≥ 0.5 solid, <b style="color:#e8a33d">orange</b> 0.3–0.5 so-so, '
            '<b style="color:#e0564f">red</b> &lt; 0.3 weak). Genres with few training movies, or vague ones like “Drama”, tend to be harder. '
            'Use this to judge how much to trust a prediction.</div>', unsafe_allow_html=True)
        st.pyplot(chart_reliability(bundle["per_genre"]))
        st.dataframe(bundle["per_genre"].sort_values("F1", ascending=False).round(3),
                     hide_index=True)