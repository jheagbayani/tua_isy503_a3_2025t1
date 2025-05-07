# ISY503 Intelligent Systems - Group 2 Final Project: Sentiment Analysis of Product Reviews

## Project Overview

This project is submitted as part of the ISY503 Intelligent Systems course. Our team (Group 2) has developed a Natural Language Processing (NLP) model to perform sentiment analysis on Amazon product reviews. The goal is to classify reviews as either positive or negative based on their text content.

The core of this project is a Jupyter Notebook (`ISY503_Group_2_Assessment_3_Final_Project.ipynb`) that details the entire machine learning pipeline, including:
* Data loading and preprocessing (text cleaning, spelling correction, negation handling, tokenization, padding).
* Automated hyperparameter tuning using Keras Tuner.
* Training a Bidirectional LSTM model.
* Evaluating the model's performance.
* An inference function to predict sentiment on new review text.
* Options to load a pre-trained model or train a new one.

This repository also includes assets for a demonstration of a simple web user interface (WebUI) that would interact with the sentiment analysis model.

## Team Members

* Jay-ar Anacleto Agbayani
* Archkymuel llao Bautista
* Nguyen Bui
* Hoang Tuan Ngo

## Files in This Repository

* **`ISY503_Group_2_Assessment_3_Final_Project.ipynb`**: The main Jupyter Notebook containing all the Python code for data processing, model training, evaluation, and inference.
* **Pre-trained Model & Parameters (Suffix `--TRUE_TRUE` indicates settings: `REMOVE_STOPWORDS = True`, `APPLY_LEMMATIZATION = True`):**
    * `final_best_sentiment_model--TRUE_TRUE.h5`: The saved Keras model weights.
    * `model_params--TRUE_TRUE.json`: Contains `max_length` and `actual_vocab_size` parameters corresponding to the pre-trained model.
    * `tokenizer--TRUE_TRUE.pkl`: The saved Keras Tokenizer object.
* **`demo_video.mp4`**: (short video file demonstrating the WebUI)
* **`webui_images`**:
    * `Arki.jpeg`
    * `Jay.jpeg`
    * `Lux.jpeg`
    * `Tuan.jpeg`
    * `Torrens Logo.jpeg`
* **`data/`**: (This directory will be created by the notebook to store downloaded datasets and spelling dictionaries if training a new model).
* **`keras_tuner_dir/`**: (This directory will be created by the notebook if hyperparameter tuning is run).
* **`README.md`**: This file.

## Setup and Usage

### Prerequisites

Ensure you have Python 3.x installed. The notebook uses several Python libraries. You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn nltk tensorflow keras-tuner scikit-learn wget symspellpy requests
```

The notebook will also attempt to install symspellpy and keras-tuner if they are not found when it first runs.

### Running the Notebook
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```
2. Place Pre-trained Files (Optional but Recommended for Quick Start):
- If you intend to use the provided pre-trained model, ensure `final_best_sentiment_model--TRUE_TRUE.h5`, `model_params--TRUE_TRUE.json`, and `tokenizer--TRUE_TRUE.pkl` are in the root directory of the cloned project.
3. Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
# or
jupyter lab
```
4. Open the `ISY503_Group_2_Assessment_3_Final_Project.ipynb` file.
5. Configure Flags (See below).
6. Follow the instructions and run the cells sequentially.
### Configuration Flags (in Section 0 of the Notebook)
These flags are crucial for controlling how the notebook operates:

- `LOAD_PRETRAINED_MODEL` (Boolean):
    - `True`: Recommended for quick testing, inference, or if you want to use the provided model for a WebUI.
        - The notebook will skip data download, preprocessing, hyperparameter tuning, and model training.
        - It will attempt to load the model specified by `SAVED_MODEL_PATH` (e.g., `final_best_sentiment_model--TRUE_TRUE.h5`), `SAVED_TOKENIZER_PATH`, and `SAVED_PARAMS_PATH`.
        - You can proceed directly to Section 4 (Load Model and Evaluate) and Section 5 (Model Inference).
        - For WebUI: A WebUI would typically use a pre-trained model. Setting this flag to True simulates the state where the model is ready for such an application. The WebUI would load the `.h5`, `.pkl`, and `.json` files.
    - `False` (default): For training a new model or re-running the entire pipeline.
        - The notebook will perform all steps: data download, preprocessing, hyperparameter tuning, model training, and evaluation. This is a lengthy process.
        - The trained model, tokenizer, and parameters will be saved (you can adjust SAVED_MODEL_PATH etc. if you wish to save with a different name).
- Preprocessing Flags: These flags define how text is processed.
    - `APPLY_SPELLING_CORRECTION` (Boolean)
    - `HANDLE_NEGATION_TOKENS` (Boolean)
    - `REMOVE_STOPWORDS` (Boolean)
    - `APPLY_LEMMATIZATION` (Boolean)

    Important Notes on Preprocessing Flags:
    1. When `LOAD_PRETRAINED_MODEL = True`: The preprocessing flags defined in Section 0 of the notebook must match the settings used when the pre-trained model was originally saved for the inference to work correctly with new text. The provided model `final_best_sentiment_model--TRUE_TRUE.h5` was trained with `APPLY_SPELLING_CORRECTION = True`, `HANDLE_NEGATION_TOKENS = True`, `REMOVE_STOPWORDS = True`, and `APPLY_LEMMATIZATION = True`. The notebook is set up to use these flags for inference when loading this specific model.
    2. When `LOAD_PRETRAINED_MODEL = False`: These flags will determine how the new data is preprocessed before training a new model. You can experiment with different combinations.
### Workflow Scenarios:
- Testing/Inference with Provided Pre-trained Model (Fastest Scenario, also for WebUI preparation):
    1. Ensure pre-trained files (`.h5`, `.json`, `.pkl` with `--TRUE_TRUE` suffix) are in the root directory.
    2. In the notebook (Section 0), set `LOAD_PRETRAINED_MODEL = True`.
    3. Ensure `APPLY_SPELLING_CORRECTION = True` and `HANDLE_NEGATION_TOKENS = True` (and others are `False`) to match the pre-trained model's conditions.
    4. Run cells. You can jump to Section 5 for inference. The model is now ready to be used by an external application like a WebUI.
- Training a New Model (Lengthy Scenario):
    1. In the notebook (Section 0), set `LOAD_PRETRAINED_MODEL = False`.
    2. Set your desired preprocessing flags (`APPLY_SPELLING_CORRECTION`, `HANDLE_NEGATION_TOKENS`, etc.).
    3. Run all cells from the beginning. The notebook will download data, preprocess, tune, train, and save the new model.
