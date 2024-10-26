
# AI Text Detection Mini Research Project

This project aims to explore and develop methodologies for detecting AI-generated text. Utilizing advanced NLP techniques and a trained RoBERTa model, this analysis investigates various text patterns and statistical markers to differentiate human-written text from AI-generated content.
We have tried with three models ,each with it's own characteristics,

#### Models
1. Linguistic Features Model
2. Vectorization Distance Model
3. Transferring Learning Model (`Model binary files skipped cause of large size in the repo`)


## Project Structure

```
├── app.py                 # Main application file
├── helpers.py             # Helper functions for data processing and analysis
├── roberta-model          # Directory containing the 2500 paragraphs trained RoBERTa model
│   ├── config.json
│   └── model.safetensors
├── roberta-model-2        # Alternative RoBERTa model with 50k paragraphs trained
│   ├── config.json
│   └── model.safetensors
├── roberta-tokenizer      # Tokenizer files for RoBERTa model
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── roberta-tokenizer-2    # Tokenizer files for the alternative model
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── sample.html            # Sample HTML output for analysis results
├── statsmodel.sav         # Saved statistical aka linguistic model for pattern analysis
├── vectorizermodel.sav    # Saved vectorization model for feature extraction
└── vectorizer.sav         # Additional vectorizer for feature engineering
```

## Research Workflow

1. **Data Preprocessing**:
   - Processed text data with NLP techniques, including POS tagging and tokenization.
   - Reduced dataset to approximately 300 words to optimize for analysis.

2. **Data Extraction:**
    - Extracted the Linguistic features such count of `pos` and other nlp features.

2. **Model Training**:
   - Trained a RoBERTa model (`roberta-model` and `roberta-model-2`) on the preprocessed dataset.
   - Models saved in `safetensors` format for efficient deployment.

3. **Feature Extraction and Analysis**:
   - Employed vectorization methods and statistical modeling to analyze linguistic patterns.
   - Models saved in `.sav` format (`statsmodel.sav`, `vectorizermodel.sav`) for reproducibility.

4. **Detection Pipeline**:
   - Combined tokenizer, vectorizer, and statistical models to generate predictions on whether text is AI-generated or human-written.

## Getting Started

To run the project:

1. Clone the repository.
2. Install the dependencies via ```pip install -r requirements.txt```
3. Command in terminal `python app.py` to start the ***FASTHtml*** App.

### Future Exploration
May try to understand and use embeddings with llms in order to understand more relations among the text corpus.