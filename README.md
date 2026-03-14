# NeuroSQL

**NeuroSQL** is an advanced neural Text-to-SQL generator based on T5/Transformers.  
Convert natural language questions and database table schemas into accurate SQL queries.

---

## Installation

NeuroSQL is modular: you can install only the components you need.

#### Install for production/basic usage (core dependencies only):

```bash
pip install neurosql
```

#### Install with API support (FastAPI, uvicorn, pydantic):

```bash
pip install neurosql[api]
```

#### Install for development (includes notebooks, testing, plotting, ML utilities):

```bash
pip install neurosql[dev,api]
```

#### For exact environment reproduction (e.g., full research replication):

```bash
pip install -r requirements.txt
```

---

## Quick Start

**Python Example:**
```python
from neurosql import NeuroSQLModel

model = NeuroSQLModel.from_pretrained("0xV4h3/neurosql")
sql = model.generate(
    query="Show all sales in Armenia for 2020",
    context="CREATE TABLE sales(id INT, country VARCHAR(20), year INT, sales INT);"
)
print(sql)  # Example: SELECT * FROM sales WHERE country='Armenia' AND year=2020;
```

---

## Command-Line Interface (CLI)

Run SQL generation in the terminal:

```bash
neurosql --query "Show sales in Armenia for 2020" --context "CREATE TABLE sales(id INT, country VARCHAR(20), year INT, sales INT);"
```

---

## API

Integrate NeuroSQL into your service using the built-in FastAPI server.  
See usage examples in the [examples/](examples/) directory.

---

## Notebooks & Training

Find Jupyter notebooks for fine-tuning and experimenting with the model in the `notebooks/` directory.  
You can further customize NeuroSQL for your own datasets and domains.

---

## Model Card

The pretrained and fine-tuned model is available on Hugging Face:  
[Hugging Face Model Card](https://huggingface.co/0xV4h3/neurosql)

---

## Requirements

- Python 3.8+
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.10.0
- tqdm
- For API: fastapi, uvicorn, pydantic
- For development: notebook, pytest, matplotlib, scikit-learn, etc.
- See [requirements.txt](requirements.txt) for the complete development environment.

---

## License

MIT License