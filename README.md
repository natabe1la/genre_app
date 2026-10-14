🎬 Movie Genre Predictor

A web app that predicts the **genres of a movie from its overview text** and shows **why**
with graphs. It is a **multi-label** classifier: one movie can receive several genres at once.

The model is a One-vs-Rest linear **Support Vector Machine (SVM)** trained on the
TMDB 5000 Movie Dataset. The training data is used only for training. At prediction time you
type a brand-new overview that the model has never seen.

Developed for **CST9, Phase 2**. The full experiment is in
[`notebooks/CST9_Phase_2_final.ipynb`](notebooks/CST9_Phase_2_final.ipynb).

## Features

- Paste any movie overview and get the predicted genres, with a match strength for each
- **Sensitivity slider** to show more or fewer genres (0.00 = the notebook's default decision rule)
- Four explained graphs:
  1. **Genre scores:** the SVM score of every genre
  2. **Word evidence:** which words pushed toward or away from each predicted genre
  3. **Highlighted text:** your own overview colour-coded by influence
  4. **Heatmap:** why more than one genre was chosen (shown when several genres are predicted)
- Model scores in the sidebar (Accuracy, Micro Precision, Micro Recall, Micro F1, Hamming Loss)
  measured on the held-out test split
- "About the model" tab with a reliability chart (F1 score per genre)

## How it works

| Step | Details |
|------|---------|
| Text cleaning | lowercase, remove HTML/URLs/non-letters, drop words with 2 or fewer characters, remove English stopwords, Snowball stemming |
| Features | TF-IDF with `max_df=0.5`, `min_df=1`, `ngram_range=(1, 2)` |
| Classifier | One-vs-Rest `SVC(kernel="linear", C=1, class_weight="balanced")` |
| Split | 80% train / 20% test, `random_state=42` |
| Prediction | a genre is predicted when its SVM decision score is above the cut-off (default 0) |

The explanation graphs come straight from the trained SVM: a genre's score is the sum of
(word TF-IDF value × learned word weight) plus an offset, so each bar is a real part of the score.

## Project structure

```
genre_predictor/
├── app.py                  # Streamlit app: training, prediction, graphs
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── data/
│   ├── README.md           # data setup instructions
│   └── tmdb_5000_movies.csv   # you add this file (see Data setup)
└── notebooks/
    └── CST9_Phase_2_final.ipynb   # full experiment: EDA, SVM, tuning, model comparison
```

## Requirements

- Python 3.9 or newer
- An internet connection on the first run (NLTK downloads its English stopword list once)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows (cmd or PowerShell)

# 3. Install the dependencies
pip install -r requirements.txt
```

## Data setup

There is no database server. The data is one CSV file. Download
`tmdb_5000_movies.csv` and place it in the `data/` folder. See [`data/README.md`](data/README.md)
for the download link, the columns used and the cleaning steps.

## Usage

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>.

1. **First launch only:** the app trains the model from the CSV and saves it as
   `genre_svm_model.joblib`. This can take a few minutes while the page shows
   "Training the SVM...". Later launches load the saved model in seconds.
2. In the **Predict** tab, pick an example or paste your own movie overview.
3. Optionally adjust the **Sensitivity** slider.
4. Press **Predict genre** and read the chips and the four graphs, each with its own explanation.
5. Open **About the model** to see how reliable the model is for each genre.
6. Use **Retrain model from CSV** in the sidebar after changing the data or settings.

### Changing the model settings

Settings sit at the top of `app.py` (`SVM_C`, `TEST_SIZE`, `RANDOM_STATE`). They match the
notebook. After changing one, press **Retrain model from CSV**.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `command not found: python` (macOS) | use `python3` instead |
| `streamlit: command not found` | activate the environment first, or run `python3 -m streamlit run app.py` |
| "Could not find tmdb_5000_movies.csv" | place the CSV in the `data/` folder |
| Page is blank on the first run | the model is still training; wait a few minutes |
| NLTK stopwords error | connect to the internet once, or run `python -c "import nltk; nltk.download('stopwords')"` |

## Limitations

- The model sees only the overview text, and only words learned from the training movies.
- Rare genres (for example TV Movie, Foreign, Documentary) have very few training examples, so
  predictions for them are unreliable.
- SVM scores are not probabilities, and the word evidence reflects patterns in the training
  data, not the meaning of a word.
- English text only.

## Dataset

TMDB 5000 Movie Dataset (Kaggle), derived from [The Movie Database (TMDB)](https://www.themoviedb.org/).
This product uses the TMDB data but is not endorsed or certified by TMDB.
