# Converting Your Google Colab Project to Production

This guide walks you through converting your Google Colab fake news detector into a production-ready Docker application.

## Step 1: Export Your Models from Colab

In your Google Colab notebook, export each trained model as a pickle file:

```python
import pickle

# For each of your trained models, save them:
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(your_logistic_regression_model, f)

with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(your_naive_bayes_model, f)

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(your_svm_model, f)

# If you have a vectorizer/preprocessor, save that too:
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Download the files
from google.colab import files
files.download('logistic_regression_model.pkl')
files.download('naive_bayes_model.pkl')
files.download('svm_model.pkl')
files.download('tfidf_vectorizer.pkl')
```

## Step 2: Add Models to Repository

1. Create a `models/` folder in your repository if it doesn't exist
2. Upload all your `.pkl` files to the `models/` directory

```bash
models/
├── logistic_regression_model.pkl
├── naive_bayes_model.pkl
├── svm_model.pkl
└── tfidf_vectorizer.pkl
```

## Step 3: Update the Application Code

### If you use a vectorizer/preprocessor:

Edit `app.py` and update the `predict_fake_news()` function to include preprocessing:

```python
# At the top of app.py, add vectorizer loading:
def load_models():
    """Load all trained models from pickle files"""
    models = {}
    vectorizer = None
    
    if MODELS_DIR.exists():
        for pkl_file in MODELS_DIR.glob("*.pkl"):
            try:
                with open(pkl_file, 'rb') as f:
                    if 'vectorizer' in pkl_file.stem.lower():
                        vectorizer = pickle.load(f)
                    else:
                        models[pkl_file.stem] = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {pkl_file.stem}: {e}")
    
    return models, vectorizer

# Then update predict_fake_news to use vectorizer:
def predict_fake_news(text):
    if not text or len(text.strip()) == 0:
        return {"error": "Please enter some text to analyze"}
    
    models, vectorizer = load_models()
    
    if not models or vectorizer is None:
        return {"error": "Models or vectorizer not loaded"}
    
    try:
        # Vectorize the input text
        text_vectorized = vectorizer.transform([text])
        
        predictions = {}
        scores = []
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(text_vectorized)[0]
                    fake_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                    predictions[model_name] = float(fake_prob)
                    scores.append(fake_prob)
                elif hasattr(model, 'predict'):
                    pred = model.predict(text_vectorized)[0]
                    predictions[model_name] = float(pred)
                    scores.append(float(pred))
            except Exception as e:
                print(f"Error with {model_name}: {e}")
        
        # Calculate ensemble
        valid_scores = [s for s in scores if isinstance(s, (int, float))]
        if valid_scores:
            ensemble_score = np.mean(valid_scores)
            prediction = "🚨 FAKE NEWS" if ensemble_score > 0.5 else "✓ REAL NEWS"
        
        return {
            "Prediction": prediction,
            "Confidence": f"{ensemble_score:.2%}",
            "Individual Model Scores": predictions
        }
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
```

## Step 4: Update requirements.txt

If you used specific libraries in Colab (like NLTK, spaCy, etc.), add them to `requirements.txt`:

```
gradio==4.26.0
flask==3.0.0
scikit-learn==1.4.1
numpy==1.24.3
pandas==2.0.3
nltk==3.8.1  # If you used NLTK
spacy==3.5.0  # If you used spaCy
```

## Step 5: Deploy with Docker

### Start the application:

```bash
docker compose up
```

The interface will be available at `http://localhost:7860`

### Verify it's working:

- Visit `http://localhost:7860` in your browser
- Try a news article to get predictions
- Check the logs for any errors

## Troubleshooting

### Issue: "No models loaded"

**Solution:**
1. Ensure your `.pkl` files are in the `models/` directory
2. Check file permissions: `ls -la models/`
3. Verify pickle files aren't corrupted:
   ```python
   import pickle
   with open('models/your_model.pkl', 'rb') as f:
       model = pickle.load(f)
   ```

### Issue: "Model prediction error"

**Possible causes:**
- Input text format doesn't match training data format
- Vectorizer preprocessing is different in production
- Model expects specific input shape

**Solution:** Ensure preprocessing in `app.py` matches your Colab notebook exactly

### Issue: Container won't start

**Check logs:**
```bash
docker compose logs -f
```

**Common issues:**
- Port 7860 already in use: Change in `docker-compose.yml`
- Missing `models/` directory: Create it with `mkdir models`
- Requirements conflict: Try rebuilding: `docker compose build --no-cache`

## Testing Locally (Without Docker)

Before deploying with Docker, test locally:

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run app
python app.py
```

## Next Steps

1. ✅ Models exported from Colab
2. ✅ Models added to `models/` directory
3. ✅ Application code verified
4. ✅ Docker deployment tested
5. ✅ Ready for instructor review!

## File Reference

- **app.py** - Main Gradio web application
- **api.py** - Optional REST API (for API access)
- **requirements.txt** - Python dependencies
- **Dockerfile** - Docker image definition
- **docker-compose.yml** - Docker Compose configuration
- **models/** - Directory for your `.pkl` model files

## Git Push to GitHub

Once everything works locally:

```bash
git add .
git commit -m "Add production-ready fake news detector with Docker"
git push origin main
```

Your instructor can then:
```bash
git clone <your-repo-url>
cd fake-news-detect
docker compose up
# Open http://localhost:7860
```

That's it! 🎓
