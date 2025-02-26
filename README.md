# PubMed NLP Classification

## Dataset

The dataset used in this project is derived from PubMed, containing text data for Natural Language Processing (NLP) classification tasks. The dataset includes:

- **Text Data:** Biomedical text extracted from PubMed articles.
- **Labels:** Categorized into 5 different classes for classification (BACKGROUNDS, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS).

## Models Implemented

The following models were trained and evaluated:

### Model 0: Naïve Bayes with TF-IDF Encoder (Baseline)

- Traditional machine learning approach using a TF-IDF vectorizer and Naïve Bayes classifier.

### Model 1: Conv1D with Token Embeddings

- Uses convolutional layers to process token-based embeddings.

### Model 2: TensorFlow Hub Pretrained Feature Extractor

- Leverages a pretrained model from TensorFlow Hub as a feature extractor.

### Model 3: Conv1D with Character Embeddings

- Uses character-level embeddings and convolutional layers for classification.

### Model 4: Pretrained Token Embeddings + Character Embeddings

- Combines the approaches of Model 2 and Model 3.

### Model 5: Pretrained Token Embeddings + Character Embeddings + Positional Embeddings

- Enhances Model 4 with positional embeddings for better sequence representation.

### Model 6: Pretrained Token Embeddings + Character Embeddings + Relative Positional Embeddings

- Introduces relative positional embeddings for improved contextual understanding.

## Model Performance

Each model was evaluated based on the following metrics:

- **Accuracy**: Measures the overall correctness of predictions.
- **Precision**: Measures the proportion of correctly predicted positive observations.
- **Recall**: Measures the proportion of actual positives correctly predicted.
- **F1 Score**: Harmonic mean of precision and recall.

### Results Summary

| Model                                        | Accuracy | Precision | Recall | F1 Score |
| -------------------------------------------- | -------- | --------- | ------ | -------- |
| Model 0 (Naïve Bayes)                        | 71.67%      | 0.712432        | 0.716675     | 0.692510       |
| Model 1 (Conv1D Token)                       | 79.52%      | 0.794529        | 0.795222     | 0.791915       |
| Model 2 (TF Hub)                             | 72.80%      | 0.724341        | 0.727958     | 0.722599       |
| Model 3 (Conv1D Character)                   | 47.23%      | 0.429898        | 0.472308     | 0.429068       |
| Model 4 (Token + Char)                       | 74.10%      | 0.736320        | 0.740999     | 0.735125       |
| Model 5 (Token + Char + Positional)          | 81.52%      | 0.817906        | 0.815165     | 0.811884       |
| Model 6 (Token + Char + Relative Positional) | 83.34%      | 0.836608        | 0.833449     | 0.828667       |

(Replace XX with actual values from model evaluation.)

## How to Run the Notebook

1. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas scikit-learn matplotlib
   ```
2. Run the notebook in Jupyter or Google Colab.
3. Ensure the dataset is properly loaded and preprocessed before training models.

## Conclusion

This project explores various NLP classification models applied to biomedical text from PubMed. The results indicate that advanced deep learning approaches outperform traditional machine learning models like Naïve Bayes.

