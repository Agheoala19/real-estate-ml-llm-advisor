# California Housing Valuation & AI Market Analyst

An end-to-end hybrid AI application that combines **classical Machine Learning** (ensemble regression modeling) with **Generative AI** (Large Language Models) to deliver data-driven real estate valuations alongside structured investment analysis reports.

---

## Architecture & Workflow

Large Language Models excel at synthesis and reasoning, yet they frequently hallucinate or miscalculate numerical estimates on tabular datasets. Conversely, classical ML models produce accurate statistical predictions but lack business narrative. This project bridges both capabilities:

```text
[User Input via Streamlit Dashboard]
                 │
                 ▼
[Scikit-Learn Random Forest Regressor] ──► Calculates Numeric Valuation ($)
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
             [OpenAI GPT-4o-mini via Structured Prompt]
                                     │
                                     ▼
          [Actionable Real Estate Investment Report]
```

### Core Modules
1. **ML Engine (`ml_engine.py`)**: Handles data ingestion, model training on the California Housing dataset, evaluation benchmarking, model serialization (`joblib`), and inference.
2. **GenAI Analyst (`llm_analyst.py`)**: Uses context injection and prompt engineering to pass both property metrics and the ML-predicted valuation to OpenAI's API, generating an executive investment report.
3. **Interactive Web UI (`app.py`)**: A responsive Streamlit interface allowing users to adjust neighborhood parameters dynamically and trigger real-time analysis.

---

## Model Performance & Benchmarking

During development, baseline linear modeling was benchmarked against an ensemble tree-based model using an 80/20 train/test split (`random_state=42`):

| Model | $R^2$ Score | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) |
|---|---|---|---|
| **Linear Regression (Baseline)** | `0.575` | `0.556` | ~$74,500 |
| **Random Forest Regressor (Final)** | **`0.805`** | **`0.255`** | **~$50,500** |

The **Random Forest Regressor** (100 estimators) improved variance explanation ($R^2$) by over **23%**, capturing non-linear relationships and geographical interactions (such as coastal proximity and neighborhood income levels).

---

## Tech Stack

* **Language:** Python 3.10+
* **Machine Learning & Data Processing:** `scikit-learn`, `pandas`, `numpy`, `joblib`
* **Generative AI & API Integration:** `openai` (GPT-4o-mini), `python-dotenv`
* **Frontend & Visualization:** `streamlit`

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/california-housing-ai-analyst.git
cd california-housing-ai-analyst
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Train and serialize the ML model
Run the ML pipeline once to train the Random Forest model and generate the `.pkl` file locally:
```bash
python ml_engine.py
```

### 6. Launch the web application
```bash
streamlit run app.py
```
The dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## Project Structure

```text
california-housing-ai-analyst/
│
├── ml_engine.py          # Model training, evaluation, serialization & inference
├── llm_analyst.py        # OpenAI client setup & structured prompt engineering
├── app.py                # Interactive Streamlit web dashboard
├── housing_model.pkl     # Serialized Random Forest model (generated locally)
├── requirements.txt      # Python package dependencies
├── .gitignore            # Excludes .env, venv, __pycache__, and .pkl files
└── README.md             # Project documentation
```

---

## Future Improvements

* **Hyperparameter Tuning:** Implement `GridSearchCV` or `RandomizedSearchCV` / `XGBoost` to push $R^2$ beyond `0.85`.
* **Geospatial Visualization:** Integrate interactive map rendering (`pydeck` or `folium`) directly inside the Streamlit dashboard based on selected coordinates.
* **Structured Output Validation:** Enforce Pydantic schema validation on the LLM response for guaranteed JSON/UI card parsing.