# Inconsistency Detection API

This service detects logical inconsistencies in prompts by combining Azure OpenAI's language models with sheaf theory from computational topology.

## Overview

The API analyzes text prompts to:

1. Extract individual claims
2. Assess pairwise logical consistency between claims
3. Build a mathematical sheaf structure to detect global inconsistencies
4. Identify circular logic and contradictions that might not be evident in isolated claim pairs

Based on the research paper: "Prospects for inconsistency detection using large language models and sheaves" by Steve Huntsman, Michael Robinson, and Ludmilla Huntsman.

## Project Structure

```
inconsistency-detector/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py         # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration handling
│   │   └── schemas.py        # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── openai_service.py # Azure OpenAI integration
│   │   └── detector.py       # Inconsistency detection logic
│   └── utils/
│       ├── __init__.py
│       └── sheaf.py          # Sheaf theory utilities
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_detector.py
├── .env                      # Environment variables (not tracked in git)
├── .env.example              # Example environment file
├── .gitignore
├── requirements.txt
├── main.py                   # Application entry point
└── README.md
```

## Setup and Installation

1. **Clone the repository**

   ```bash
   git clone [repository-url]
   cd inconsistency-detector
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example environment file and update with your credentials:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your Azure OpenAI credentials:

   ```
   AZURE_OPENAI_API_KEY=your-key-here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_OPENAI_DEPLOYMENT=your-deployment-name
   ```

5. **Run the application**

   ```bash
   uvicorn main:app --reload
   ```

## API Usage

### Analyze Prompt Endpoint

**POST** `/api/analyze`

Request body:

```json
{
    "prompt": "Your text prompt to analyze for inconsistencies",
    "visualization": true
}
```

Response:

```json
{
    "global_consistency_score": 4.5,
    "claims": [
        "Claim 1 extracted from the prompt",
        "Claim 2 extracted from the prompt"
    ],
    "pairwise_consistency": {
        "0-1": 7.5
    },
    "inconsistent_pairs": [
        {
            "claim1_index": 2,
            "claim2_index": 5,
            "claim1_text": "Text of claim at index 2",
            "claim2_text": "Text of claim at index 5",
            "consistency_score": 2.5
        }
    ],
    "visualization_url": "http://server/visualizations/12345.png"
}
```

## Example

Testing the circular logic example:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I have a question: if I eat more than my little dog, but my little dog eats more than Ana'\''s, Ana'\''s eats more than Juan'\''s, Juan'\''s eats more than Miguel'\''s, and Miguel'\''s little dog eats more than Miguel himself. If we know for sure that Miguel eats much more than I do, who eats the most?",
    "visualization": true
  }'
```

## License

[License information]

## References

Huntsman, S., Robinson, M., & Huntsman, L. (2024). Prospects for inconsistency detection using large language models and sheaves. arXiv:2401.16713v1.
