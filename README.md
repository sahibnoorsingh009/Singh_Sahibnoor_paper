# Offensive Language on Twitter During the 2024 U.S. Election

## Overview
This project investigates the dynamics of offensive language on Twitter during the 2024 U.S. Presidential Election. The study leverages machine learning models to detect offensive content in tweets and analyzes the distribution, temporal trends, and political affiliations associated with offensive language. The findings provide insights into the patterns of offensive language and its implications for social cohesion, democratic discourse, and platform governance.

## Key Objectives
1. **Develop Predictive Models**: Build and evaluate machine learning models to detect offensive language in tweets.
2. **Temporal Analysis**: Analyze how offensive language evolves over time during the election period.
3. **Political Affiliation Analysis**: Examine the distribution of offensive language across different political parties.
4. **User Engagement Analysis**: Investigate whether offensive tweets are concentrated among a small subset of users or widespread.

## Datasets
1. **OLID Dataset**: Contains 14,100 tweets labeled as offensive (1) or non-offensive (0). Used for training and validating machine learning models.
   - [Updated_OLID_Dataset_with_Binary_subtask_a.csv](https://github.com/sahibnoorsingh009/CSS_Report/blob/main/Updated_OLID_Dataset_with_Binary_subtask_a.csv)
   - [OLID.csv](https://drive.google.com/file/d/1I8bGTAJtmfEwcfPZcF1xA2-HlgAlO0UN/view?usp=share_link)
2. **2024 U.S. Election Dataset**: A corpus of approximately 250,000 tweets related to the 2024 U.S. Presidential Election. Used for applying the trained models to analyze offensive language in a real-world political context.
   - [output.csv](https://drive.google.com/file/d/1_Ew2E4XR-hO3TRn-KWje8-j2OFODmshP/view?usp=share_link)
   - [new_dataset_with_predictions.csv](https://drive.google.com/file/d/1GW5I79CcQNAjl2J_X_PZVD8IBDYU09Fv/view?usp=share_link)
   - [sorted_output.csv](https://drive.google.com/file/d/1-ZIbGYN6DneqK1umJ8Os4sXVpBhxt7cH/view?usp=share_link)
   - [election2024_data.csv](https://drive.google.com/file/d/1--1QtFJ0yv8LGs_e4LNqmt2SU3lXkTdG/view?usp=share_link)
   - [new_data_with_sentiment.csv](https://drive.google.com/file/d/1K2bjNkGK1tJtfZhxTlbXBLvNKed4HXpC/view?usp=share_link)

## Methodology
### Data Preprocessing
- **Data Cleaning**: Removed duplicates, missing values, and irrelevant columns. Normalized text by converting to lowercase, removing URLs, mentions, hashtags, and special characters.
- **Feature Engineering**: Applied **TF-IDF Vectorization** and **Word Embeddings** to transform text data into numerical features suitable for machine learning models.

### Model Training and Evaluation
- **Models Used**: Logistic Regression, Random Forest, Multinomial Naive Bayes, and XGBoost.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
- **Best Performing Model**: XGBoost achieved the highest accuracy (77.19%) and balanced precision and recall.

### Analysis
- **Exploratory Data Analysis (EDA)**: Visualized sentiment distribution, word clouds, and tweet length analysis.
- **Temporal Analysis**: Tracked offensive sentiment trends over time, particularly around key political events.
- **Party-Based Analysis**: Analyzed offensive language distribution across political parties (Republican, Democratic, Independent, Green, Libertarian).

## Results
1. **Model Performance**: XGBoost outperformed other models in detecting offensive language, with balanced precision and recall.
2. **Temporal Trends**: Offensive language decreased as the election approached, suggesting evolving societal norms or increased moderation.
3. **Political Affiliation**: Tweets referencing the Republican Party exhibited the highest levels of offensive sentiment, while smaller parties had lower and more stable levels.
4. **User Engagement**: Offensive tweets were disproportionately directed toward high-profile political accounts like GOP and KamalaHQ.

## How to Reproduce the Results
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sahibnoorsingh009/CSS_Report.git
   cd offensive-language-twitter-2024
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebooks**:
   - Open the Jupyter notebooks in the `notebooks/` directory to preprocess data, train models, and analyze results.
   - Start with `eda1.py`.
   - Proceed to `olid_dat_training_-2.py` for model training and evaluation.
   - Finally, use `party_classifier.py` for temporal and political affiliation analysis.


## Dependencies
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, nltk, spacy, wordcloud, gensim

## References
1. Zampieri, M., et al. (2019). SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media.
2. McCombs, M. E., & Shaw, D. L. (1972). The agenda-setting function of mass media.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.

## Acknowledgments
- Supervised by Paul Bauer.
- Datasets sourced from Huggingface, Kaggle, and USC-X-24-US-Election GitHub repository.
