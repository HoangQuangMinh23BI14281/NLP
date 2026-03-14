# NLP Project: Information Extraction From IT Job Descriptions

Our NLP group project extracte structured information from unstructured IT job descriptions using Transformer-based Named Entity Recognition (NER).

The system is built around five recruitment-oriented entity types:

- ROLE
- SKILL
- LOC
- EXP
- SALARY

The project includes:

- Synthetic data generation for the IT recruitment domain
- Training and evaluation notebooks for multiple Transformer backbones
- Saved experiment outputs and model checkpoints
- A LaTeX report documenting methodology and results
- A demo application with FastAPI backend and browser-based frontend

## Repository Structure

```text
NLP/
├── data/
│   ├── data_generator.py        # Synthetic dataset generation script
│   └── dataset.jsonl            # Generated dataset
├── data_analyze/
│   └── data_analyze.ipynb       # Data exploration notebook
├── demo/
│   ├── api/                     # FastAPI backend for inference and matching
│   ├── backend/                 # Demo model folders
│   ├── frontend/                # Browser UI for extraction and matching
│   └── test_roberta_crf.py      # Local model test script
├── report latex/
│   ├── main.tex                 # Main LaTeX file
│   └── sections/                # Report sections
├── result/                      # Saved training results/checkpoints
│   ├── bert/
│   └── roberta/
├── train/
│   ├── bert_train(2).ipynb
│   ├── distillbert_train.ipynb
│   └── roberta_train.ipynb
├── requirement.txt              # Python dependencies
└── README.md
```

## Models and Experiments

The project compares several Transformer backbones and training objectives.

Backbones:

- BERT
- DistilBERT
- RoBERTa

## Installation

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

On Command Prompt:

```cmd
python -m venv env
env\Scripts\activate.bat
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

## Running the Demo

The demo application is located in [demo](demo) and consists of:

- FastAPI backend in [demo/api](demo/api)
- static frontend in [demo/frontend](demo/frontend)
- model files in [demo/backend](demo/backend)

### Download demo model

If the model folder is not already available locally, download it from Google Drive:

- Model download: [Google Drive link](https://drive.google.com/drive/folders/1epqMNAe6Fm7BydWjXoYwZMWt0Q1Fpj-W?fbclid=IwY2xjawQhtfJleHRuA2FlbQIxMQBzcnRjBmFwcF9pZA80Mzc2MjYzMTY5NzM3ODgAAR7aWNwopxl5bDcUDygLd6woaSzLo48NxOzAA-lczr8Ub0FuimVqx4s2n2Schg_aem_Lno5FVBkGAtBrpqrveCavA)

After downloading, place the model folder inside [demo/backend](demo/backend).

### Start the demo API

```bash
cd demo/api
python run_demo.py
```

### Open the final demo

After the API starts, open the frontend at:

- `http://127.0.0.1:8000/` or `http://localhost:8000/static/index.html`

Optional endpoints:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Labels metadata: `http://127.0.0.1:8000/labels`
- Match API: `http://127.0.0.1:8000/match`

### Demo behavior

The demo does two things:

1. Extracts entities from job-description style text
2. Shows heuristic matching against a small set of sample companies

### Example input

```text
We are looking for a Senior Java Developer with 5 years of experience in Spring Boot and AWS, located in Berlin, salary up to $5000/month.
```

## License

This repository includes an MIT-style license file at [LICENSE](LICENSE).
