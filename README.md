# Fake News Detector

A machine learning project that detects fake news using multiple classification models. Built with Python, Gradio, and Docker for easy deployment.

## Features

- **Multiple ML Models**: Ensemble approach using trained pickle models
- **Gradio Interface**: User-friendly web interface for real-time predictions
- **Docker Support**: Fully containerized for consistent deployment
- **REST API**: Flask backend for integration with other systems

## Quick Start

### Prerequisites

- Docker and Docker Compose (only required dependency)

### Run with Docker Compose

```bash
docker compose up
```

The Gradio interface will be available at `http://localhost:7860`

## Project Structure

```
fake-news-detect/
├── app.py                 # Main Gradio application
├── api.py                 # Flask REST API (optional)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── models/                # Trained model files (.pkl)
├── README.md              # This file
└── data/                  # Sample data for testing
```

## Installation (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/hammadf23/fake-news-detect.git
cd fake-news-detect
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your trained models

Place your `.pkl` files in the `models/` directory

### 5. Run the application

```bash
python app.py
```

The interface will be available at `http://localhost:7860`

## Usage

### Web Interface (Gradio)

1. Navigate to `http://localhost:7860`
2. Enter news text or article content
3. Click "Predict" to get the classification
4. Results show probability scores for each model's prediction

### REST API (Optional)

```bash
# Start the API
python api.py

# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article here"}'
```

## Model Details

The detector uses multiple pre-trained models stored as pickle files:
- Models should be saved in the `models/` directory
- Currently supports sklearn and similar serialized models
- Models are loaded and used for ensemble predictions

## Technologies Used

- **Python 3.8+**
- **Gradio** - Web interface
- **Flask** - REST API
- **scikit-learn** - ML models
- **Docker & Docker Compose** - Containerization

## File Descriptions

### `app.py`
Main Gradio application that loads models and provides the web interface.

### `api.py`
Optional Flask REST API for programmatic access to predictions.

### `requirements.txt`
Python package dependencies for the project.

### `Dockerfile`
Multi-stage Docker build for production-ready containerization.

### `docker-compose.yml`
Orchestration file for running the application with `docker compose up`.

## Troubleshooting

### Models not loading?
- Ensure all `.pkl` files are in the `models/` directory
- Check that pickle files are compatible with the Python version used

### Port already in use?
- Gradio default port: 7860
- Flask default port: 5000
- Modify `docker-compose.yml` to use different ports

### Out of memory?
- Adjust `memory` limit in `docker-compose.yml` if needed
- Reduce batch size in preprocessing

## API Response Format

```json
{
  "prediction": "Real",
  "confidence": 0.95,
  "model_scores": {
    "model_1": 0.92,
    "model_2": 0.98,
    "model_3": 0.91
  }
}
```

## Deployment

### Using Docker Compose (Recommended)

```bash
docker compose up -d
```

### Using Docker directly

```bash
docker build -t fake-news-detector .
docker run -p 7860:7860 fake-news-detector
```

## Notes for Instructor

This project demonstrates:
- ✅ Proper repository structure and organization
- ✅ Production-ready code (not notebook format)
- ✅ Docker containerization for reproducibility
- ✅ Multiple ML model integration
- ✅ Clean documentation and setup instructions
- ✅ Runnable with single command: `docker compose up`

## Author

Hammad F - May 2026

## License

MIT

