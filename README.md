# DNLP-KeyBert

The comprehensive project report, which provides an in-depth explanation of our methodology, experimental setup, and detailed results, is available for review **[here](METTERE LINK AL PAPER CHE STA DENTRO AL GITHUB)**.

## Overview

This repository explores the problem of keyword extraction from user-generated movie reviews, with the goal of enhancing the quality and interpretability of the extracted terms through targeted extensions of an established baseline model.

The baseline is built on KeyBERT, a method for unsupervised keyword extraction that uses pre-trained sentence-transformers to generate contextual embeddings of both the document and candidate phrases. These embeddings are compared using cosine similarity, and the most semantically relevant phrases are selected as keywords. While effective in identifying terms closely related to the document content, KeyBERT does not incorporate sentiment polarity or external contextual features, which are often crucial for understanding subjective texts such as reviews.

To address these limitations, this work introduces two main extensions:
- A **sentiment-aware variant**, in which keyword selection is conditioned on the polarity of the review. This is implemented either as a post-hoc reranking step based on sentiment scores or through sentiment-informed embeddings in the extraction phase.
- A **metadata-aware variant**, which enriches the document representation with auxiliary information such as review length or usefulness votes, allowing the model to account for context beyond textual content.

The effectiveness of these extensions is evaluated on a dataset of IMDB movie reviews, using both traditional keyword extraction metrics (e.g., precision, recall, nDCG) and a custom validation scheme based on sentiment alignment between keywords and overall review polarity.

By incorporating additional dimensions of meaning and context, the proposed approach improves the relevance and interpretability of extracted keywords, offering a more nuanced tool for opinion mining and content analysis in the film domain.

## Technologies Used

| Category                   | Libraries |
|----------------------------|-----------|
| Data Handling              | ![Pandas](https://img.shields.io/badge/pandas-1.3.3-green) ![NumPy](https://img.shields.io/badge/numpy-1.21.0-blue) ![tqdm](https://img.shields.io/badge/tqdm-4.62.3-blue) |
| Visualization             | ![Matplotlib](https://img.shields.io/badge/matplotlib-3.4.3-yellowgreen) ![Seaborn](https://img.shields.io/badge/seaborn-0.11.2-lightblue) |
| Deep Learning & NLP       | ![PyTorch](https://img.shields.io/badge/pytorch-1.12-orange) ![Transformers](https://img.shields.io/badge/transformers-4.12.0-yellow) ![Sentence Transformers](https://img.shields.io/badge/sentence--transformers-2.2.0-purple) ![KeyBERT](https://img.shields.io/badge/keybert-0.7.0-cyan) |
| NLP Toolkits              | ![spaCy](https://img.shields.io/badge/spaCy-3.5.0-blueviolet) ![NLTK](https://img.shields.io/badge/nltk-3.6.7-purple) ![Evaluate](https://img.shields.io/badge/huggingface--evaluate-0.4.0-yellow) ![VADER Sentiment](https://img.shields.io/badge/vaderSentiment-3.3.2-brightgreen) |
| ML & Utility              | ![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange) ![Autocorrect](https://img.shields.io/badge/autocorrect-2.6.1-red) ![wordfreq](https://img.shields.io/badge/wordfreq-2.5.1-lightgrey) |
| Web Scraping              | ![Selenium](https://img.shields.io/badge/selenium-4.1.0-green) ![WebDriver Manager](https://img.shields.io/badge/webdriver--manager-3.5.4-lightgrey) ![Requests](https://img.shields.io/badge/requests-2.26.0-darkgreen) ![BeautifulSoup](https://img.shields.io/badge/beautifulsoup-4.9.3-yellow) ![IMDbPY](https://img.shields.io/badge/imdbpy-2021.4.18-orange) |
| Built-in Modules          | ![Pickle](https://img.shields.io/badge/pickle-built--in-lightgrey) ![re](https://img.shields.io/badge/re-built--in-lightgrey) ![time](https://img.shields.io/badge/time-built--in-lightgrey) |

## Dataset

### Source

The dataset used in this project was curated from a selection of well-known movies, with a particular focus on the *Star Wars* saga. It includes both classic and contemporary titles to ensure variety in content and style. The selected films are:

- **Star Wars Series**:
  - Episode I – The Phantom Menace (1999)
  - Episode II – Attack of the Clones (2002)
  - Episode III – Revenge of the Sith (2005)
  - Episode IV – A New Hope (1977)
  - Episode V – The Empire Strikes Back (1980)
  - Episode VI – Return of the Jedi (1983)
  - Episode VII – The Force Awakens (2015)
  - Episode VIII – The Last Jedi (2017)
  - Episode IX – The Rise of Skywalker (2019)

- **Additional Films**:
  - Parasite (2019)
  - The Good, the Bad and the Ugly (1966)
  - Harry Potter and the Sorcerer’s Stone (2001)
  - Oppenheimer (2023)
  - La La Land (2016)
  - Raiders of the Lost Ark (1981)

Each dataset contains user reviews scraped from IMDb, with the following structured features:
- `Review_ID`, `Movie_ID`, `Movie_Title`
- `Rating` (numerical score from 1 to 10)
- `Review_Title`, `Review_Text`, `Review_Date`
- `Helpful_Votes`, `Total_Votes`

### Preprocessing

Text preprocessing was designed with the specific goal of preparing inputs for Transformer-based models, in particular **KeyBERT** with the `all-MiniLM-L6-v2` embedding backend. Since BERT-style models handle tokenization, casing, and subword semantics internally, the preprocessing applied was **minimal but targeted**, aiming to improve surface-level quality while preserving natural language structure.

The following steps were applied to produce the `Preprocessed_Review` column:

1. **Punctuation Spacing Normalization**  
   Ensures punctuation marks like `.`, `!`, and `?` are followed by a space when appropriate (e.g., `"hello.great"` → `"hello. great"`), without affecting numbers or abbreviations.

2. **Typo Correction**  
   Uses the `autocorrect` library along with a custom strategy:
   - Preserves valid English words based on `wordfreq` frequency data
   - Reduces repeated letters (e.g., `"amazzing"` → `"amazing"`)
   - Protects capitalized proper nouns from being altered
   - Performs multi-stage correction only when necessary

3. **Nonsense and Empty Review Removal**  
   Reviews consisting mostly of symbols, emojis, or random characters were filtered out using a heuristic based on the ratio of alphabetic to non-alphabetic content.

> **Lemmatization and stop word removal were deliberately not applied.**  
> Lemmatization may interfere with BERT’s pretraining vocabulary and reduce embedding quality.  
> Stop word handling is delegated to the **keyword extraction pipeline**, which allows for context-specific filtering.

### Keyword Evaluation Ground Truth

To evaluate the performance of the keyword extraction models, a custom **keyword annotation framework** was developed, combining two complementary sources:

1. **Plot Keywords**  
   These are predefined keywords manually associated with each film (from IMDb), typically focused on factual or plot-related elements such as *"Jedi"*, *"galaxy"*, or *"betrayal"*. Each keyword is associated with two metrics:
   - `Helpful_Votes`: the number of users who found the keyword useful
   - `Not_Helpful_Votes`: the number of users who did not find it useful

   Keywords are **ranked** based on these values, and this ranking plays a central role in evaluation:  
   > A keyword extraction model is considered more effective if it tends to recover keywords that were highly rated (helpful) by users, reflecting real-world interpretability and usefulness.

2. **AI-Generated Thematic Summaries**  
   For many films, thematic summaries were generated using AI language models based on aggregated review content. These summaries focused on deeper aspects such as cinematography, acting, direction, and emotional impact. From these summaries, **new keywords** were extracted using KeyBERT and added to the evaluation set.

   To reflect their direct connection to user opinions and narrative perception, these keywords were:
   - **Placed at the top of the ranking**
   - Assigned **artificial `Helpful_Votes` values** higher than any of the plot keywords for that film  
   - Given progressively increasing vote scores to maintain consistent ordering

   > This approach ensures that **audience-centered thematic keywords**—which often capture the essence of the review text—are correctly prioritized during evaluation.

For a few films (*e.g., The Phantom Menace*, *Attack of the Clones*, *La La Land*), AI-generated summaries were not available. In these cases, evaluation relied solely on plot keywords and their vote-based ranking.

This **hybrid ground truth** enables both:
- **Semantic evaluation**: measuring the overlap and relevance between extracted keywords and known useful terms
- **Sentiment-aware validation**: analyzing the alignment between the sentiment of extracted keywords and the polarity of the original review

## System Components

1. **KeyBERTSentimentReranker**
   - *Base*: KeyBERT with `all-MiniLM-L6-v2` sentence-transformer
   - *Extension*: Applies a post-hoc reranking strategy using sentiment polarity scores computed with a Transformer-based sentiment classifier.
   - *Model*: Uses `cardiffnlp/twitter-roberta-base-sentiment` to assess the polarity of the full review and each candidate keyword. Optionally, `nlptown/bert-base-multilingual-uncased-sentiment` can be used for multilingual or domain-agnostic scenarios.
   - *Function*: After keyword extraction, candidate phrases are reranked by combining cosine similarity (semantic relevance) and sentiment alignment between each keyword and the review. This improves interpretability by promoting emotionally consistent keywords.

2. **KeyBERTSentimentAware**
   - *Base*: KeyBERT with `all-MiniLM-L6-v2` sentence-transformer
   - *Extension*: Integrates sentiment prediction directly into keyword selection. Each candidate phrase is passed through a sentiment classifier to estimate its emotional tone.
   - *Model*: Uses `cardiffnlp/twitter-roberta-base-sentiment` as default due to its training on informal text similar to reviews. Comparative tests include `nlptown/bert-base-multilingual-uncased-sentiment`.
   - *Function*: Selects keywords based on a joint score combining semantic relevance and sentiment compatibility with the full review, enabling context-aware and emotionally aligned keyword extraction.

3. **KeyBERTMetadata**
   - *Base*: KeyBERT with `all-MiniLM-L6-v2`, enriched document embeddings
   - *Extension*: Enhances the document embedding by incorporating structured metadata (e.g., length, helpfulness).
   - *Function*: Generates a context-aware representation that reflects not only the textual content but also the user’s evaluation of the review, improving keyword relevance and personalization.

All modules share a common keyword generation pipeline (n-gram candidate extraction and stopword-aware filtering) and are evaluated against a hybrid ground truth combining curated plot keywords and AI-derived thematic keywords.

## Repository Structure

```
Dataset/
   ├── keywords_ground_truth.pkl         
   ├── summary_IA.pkl                    
   ├── sw_reviews.pkl                    
   ├── others_reviews.pkl                
   ├── Reviews_By_Movie/
   │   ├── LaLaLand.pkl, SW_Episode1.pkl, ...
   └── Extracted_Keywords/
       ├── kw_LaLaLand.pkl, kw_SW_Episode1.pkl, ...

Dataset Creation/
   ├── dataset_creation.ipynb            
   ├── IMDB_database_analysis.ipynb      
   ├── keywords_summaryIA.ipynb          
   └── Retriever.py

Evaluation/
   ├── keyword_extraction.ipynb          
   ├── evaluation_metadata.ipynb         
   ├── evaluation_reranker.ipynb         
   └── evaluation_sentiment.ipynb                       

KeyBERTMetadata/
   ├── KeyBertMetadata.py                
   └── PCA_analysis.ipynb                

KeyBERTSentimentAware/
   ├── models/
   │   ├── KeyBertSentimentAware.py      
   │   ├── KeyBertSentimentReranker.py   
   │   └── SentimentModel.py             
   └── tests/
       ├── test_sentiment.ipynb          
       └── test_reranker.ipynb

Preprocessing/
   ├── preprocessing.py
   ├── preprocessing_tests.ipynb
   ├── dataset_preprocessing.ipynb         
   └── Further_Analysis/
       ├── remove_movie_words.ipynb      
       ├── TF-IDF_Analysis.ipynb             

Keyword Extraction with BERT – Towards Human-Aligned Relevance.pdf  # Report
README.md
```

#### Description of the Repository Content

- **`Dataset/`**  
  Contains review datasets and keyword extractions. Includes:
  - `Reviews_By_Movie/`: `.pkl` files of raw reviews per movie.
  - `Extracted_Keywords/`: extracted keyword files (e.g., `kw_LaLaLand.pkl`).
  - Global files like `keywords_ground_truth.pkl`, `summary_IA.pkl`, `others_reviews.pkl`, `sw_reviews.pkl`

- **`Dataset Creation/`**  
  Scripts and notebooks for building the dataset from raw reviews and summaries.  
  Includes scraping, parsing IMDb keywords and AI-generated summary processing.

- **`Evaluation/`**  
  Comparative evaluation of all models (baseline KeyBERT, metadata-based, sentiment-aware, reranker).  

- **`Preprocessing/`**  
  Code for text normalization and cleaning.  
  - `Further_Analysis/` includes TF-IDF studies and movie-specific stopword filtering.

- **`KeyBERTMetadata/`**  
  Implementation of a modified KeyBERT that incorporates review metadata (length, polarity, ...).  
  Also includes PCA-based dimensionality reduction and visual analysis.

- **`KeyBERTSentimentAware/`**  
  Two sentiment-based extensions of KeyBERT:
  - `KeyBertSentimentAware.py`: filters keywords during extraction using RoBERTa-based sentiment.
  - `KeyBertSentimentReranker.py`: post-processing reranker based on keyword sentiment.
  - `SentimentModel.py`: wraps the RoBERTa classifier fine-tuned for movie reviews (using `cardiffnlp/twitter-roberta-base-sentiment`).
  - `tests/`: notebooks to test both reranker and sentiment-aware extractor.

- **`README.md`**  
  Main documentation file that describes the purpose, structure, and usage of the repository.

- **`Keyword Extraction with BERT – Towards Human-Aligned Relevance.pdf`**  
  The original report describing how KeyBERT works.

## Usage

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Manuele23/DNLP-KeyBert.git
   cd DNLP-KeyBert
   ```

2. **Run the notebooks individually**:  
   Each Jupyter notebook in the project is self-contained. If any external library is required, the corresponding installation and import commands are already provided in the first cells of the notebook.

3. **Model downloads**:  
   If needed, pretrained models (e.g., `all-MiniLM-L6-v2`, `cardiffnlp/twitter-roberta-base-sentiment`) will be automatically downloaded when executing the relevant code cells.

> No global environment setup is required.  
> We recommend using a Python 3.9+ environment and executing notebooks sequentially for best results.

## Conclusion

This project explored the enhancement of keyword extraction from movie reviews by extending the KeyBERT framework with sentiment-aware and metadata-driven components. Through the integration of transformer-based models, contextual sentiment analysis, and a hybrid ground truth constructed from plot keywords and AI-generated thematic summaries, the system was able to improve the relevance, diversity, and emotional alignment of extracted keywords.

The modular design of the framework enables specialization across different analytical dimensions, such as emphasizing emotional tone, thematic content, or narrative structure. Moreover, the integration with transformer-based models opens the possibility of generating high-level summaries or evaluative comments that reflect the collective opinion expressed in user reviews—similar to the editorial summaries found on platforms like IMDb.

This flexibility positions the system not only as a keyword extractor, but also as a tool for producing interpretable, content-driven insights that adapt to different genres, audiences, or information needs.

## Contributors
- **[Bianca Bartoli](https://github.com/BiancaBartoli)**
- **[Alessandro Coco](https://github.com/0c0c)**
- **[Francesca Geusa](https://github.com/FrancescaGeusa)** 
- **[Manuele Mustari](https://github.com/Manuele23)**  

## License
This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.

## Acknowledgments

- [Politecnico di Torino](https://www.polito.it)
- [Dipartimento di Automatica e Informatica - Politecnico di Torino](https://www.dauin.polito.it/)
