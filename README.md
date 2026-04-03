# Ottoman Turkish Emotion Classification (NLP Project)

This project is a Python-based NLP system that detects emotions in Ottoman Turkish / Classical Turkish-style expressions.

It is designed as a portfolio-ready end-to-end NLP project, including data processing, model training, evaluation, and an interactive demo.

## Features

- Emotion classification for Ottoman Turkish / Classical Turkish-style text
- Data cleaning and normalization pipeline
- TF-IDF + LinearSVC model
- Evaluation on a balanced real-world test set
- Confusion matrix and error analysis
- Interactive Gradio demo

## Supported Emotion Classes

The model predicts the following 8 emotions:

- cekince (hesitation)
- istek (request/desire)
- kizginlik (anger)
- merak (curiosity)
- minnettarlik (gratitude)
- saygi (respect)
- umut (hope)
- uzuntu (sadness)

## Project Structure

```
intent_emotion_project/

app.py  
train.py  
predict.py  
requirements.txt  
README.md  

data.csv  
test_real.csv  
emotion_model.pkl  

reports/  
  classification_report.txt  
  sample_errors.csv  
  confusion_matrix.png  
  demo.png (optional)
```

## Model

This project uses the following components:

TF-IDF Vectorizer:
- unigram + bigram features
- sublinear term frequency
- maximum 10,000 features

LinearSVC:
- class balancing enabled
- effective for sparse text data

## Results

Evaluation on a balanced test set:

Accuracy: 0.856  
Macro F1-score: 0.86  

### Classification Report

```
              precision    recall  f1-score   support

     cekince       0.53      0.85      0.65        40
       istek       0.95      0.53      0.68        40
   kizginlik       1.00      0.85      0.92        40
       merak       0.85      1.00      0.92        40
minnettarlik       0.98      1.00      0.99        40
       saygi       0.97      0.85      0.91        40
        umut       0.84      0.80      0.82        40
      uzuntu       1.00      0.97      0.99        40

    accuracy                           0.86       320
   macro avg       0.89      0.86      0.86       320
weighted avg       0.89      0.86      0.86       320
```

## Key Findings

The model performs strongly on:
- minnettarlik
- uzuntu
- merak
- kizginlik

The most difficult distinctions occur between:
- cekince and istek
- cekince and umut

This is due to stylistic and semantic overlap in classical Turkish expressions.

## Example Inputs

- İnşallah murad olunan netice hâsıl olur.
- Bu hususta bir miktar tereddüdüm vardır.
- Zât-ı âlinize şükranlarımı arz ederim.
- Meramımı arz etmeme müsaade buyurur musunuz?

## Training

Run:

```
python train.py
```

This will:
- train the model
- evaluate on test data
- generate reports

## CLI Prediction

Run:

```
python predict.py
```

## Gradio Demo

Run:

```
python app.py
```

## Installation

```
python -m pip install -r requirements.txt
```

## Author

Personal NLP portfolio project.
