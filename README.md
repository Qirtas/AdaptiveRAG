# RAGForDatamite

Prototype for a Retrieval-Augmented Generation (RAG) feature in DATAMITE using LangChain. 

What Does This Project Do?

This project builds a basic pipeline that:
- Preprocesses structured CSV files (content for KB)
- Converts them into LangChain-compatible `Document` objects
- Generates vector embeddings for LangChain docs
- Stores those embeddings in a vector DB
- Can retrieve documents based on users question
- Can provide LLM (Claude or Deepseek) generated answers for users questions

In this prototype, we also expose the pipeline through a FastAPI service. This allows us to run the API locally, ask questions via the CLI or endpoint and receive answers directly from the chosen LLM along with the supporting source documents.  

## How to Run

This project supports running the RAG Fast API with **two different LLM providers**:
- [Anthropic Claude](https://www.anthropic.com/)  
- [DeepSeek (via OpenRouter)](https://openrouter.ai/)  

You can switch between them without changing code by setting one environment variable.

---

## What you’ll do

	1.	Get the code
	2.	Create a Python environment
	3.	Install the required packages
	4.	Set your LLM provider and API key
	5.	Run a one-shot test by sending API request
	6.	Ask more questions (without rebuilding)



### Prerequisites

- **Python**: version **3.10** is required.  
- **Conda** (For environment management).  

---
### 1. Get the code

Open a terminal (PowerShell on Windows, Terminal on macOS) and run:

```bash
git clone --recursive https://github.com/eduardovyhmeister/Datamite.git
cd Datamite
git submodule update --init --recursive --remote
```

Open the submodule folder in your editor:
Datamite/RAGDataMite

If using PyCharm, open RAGDataMite directly as the project.

---

### 2. Create a Python environment

```bash
conda create -n datamite_env python=3.10 -y
conda activate datamite_env
```
---

### 3. Install required packages

From inside RAGDataMite:
```bash
pip install -r requirements.txt
```
---

### 4. Set your LLM provider and API key

Choose one provider.

####  Option A — Claude (Anthropic)
1. Sign in to the [Anthropic Console](https://console.anthropic.com/).
2. Navigate to **Settings** → **API Keys**
3. Click **Create Key**, give it a name (e.g., “datamite_test”), and copy the generated key. **Save it safely**.

Next, set this API key in your environment variables

##### Mac/Linus OS
```bash
export LLM_PROVIDER=claude
export ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
```
##### Windows

```bash
$env:LLM_PROVIDER = "claude"
$env:ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_KEY"
```

####  Option B — DeepSeek (via OpenRouter)

1. Go to [OpenRouter’s website](https://openrouter.ai/) and **sign up or log in**.
2. Open the menu (top-right), go to **Settings** → **API Keys**, and click **Create Key**.
3. Name the key and then copy it and store it safely—this key grants API access to DeepSeek models via OpenRouter.

Next, set this API key in your environment variables

##### Mac/Linus OS
```bash
export LLM_PROVIDER=deepseek
export OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
```

##### Windows

```bash
$env:LLM_PROVIDER = "deepseek"
$env:OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"
```

Note: On Windows you must include quotes whenever setting an environment variable , e.g. "sk-or-..."

---

### 5. Run a one-shot test

Build the index, start the server, send a query, and stop the server, all in one step.

```bash
python cli.py ask --index --reset-index --question "What is CAPEX?" --port 8000
```
Subsequent tests (do NOT index again):

```bash
python cli.py ask --question "What is CAPEX?" --port 8000 --no-index
```

---

## Switching providers later

Just reset the environment variables for the provider you want (Claude or DeepSeek) and re-run the test:

```bash
python cli.py ask --question "What is CAPEX?" --port 8000 --no-index
```

---

## Example response

When successful, you should see something like this:

```bash
Starting Uvicorn on port 8000 ...
POST http://127.0.0.1:8000/ask
{
  "answer": "Based on the provided context, CAPEX (also known as Capital Cost) refers to one-time, upfront investments made to acquire, upgrade, or maintain long-term physical and technological assets...",
  "sources": [
    {
      "name": "CAPEX",
      "score": null,
      "url": null,
      "snippet": "KPI Name: CAPEX\nAlternative Names: Capital Cost, Hardware Costs...",
      "metadata": {
        "source": "RAG/Content/ProcessedFiles/clean_KPIs.csv",
        "type": "KPI",
        "name": "CAPEX"
      }
    }
  ],
  "meta": {
    "k": 3,
    "min_similarity": 0.28
  }
}
Stopping server (PID 10424)
```

	•	answer → model’s response based on your documents
	•	sources → which document(s) were used
	•	meta → retrieval settings used

---

## Notes for Windows vs macOS

	•	Setting variables:
	•	macOS/Linux → export NAME=value
	•	Windows PowerShell → $env:NAME = "value"
