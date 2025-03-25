# Inconsistency Detection API

This service detects logical inconsistencies in text prompts using advanced language models to identify contradictions, circular logic, and other logical flaws.

## Overview

The API analyzes text prompts to:

1. Extract individual claims using language model capabilities
2. Assess logical relationships and consistency between claims
3. Detect circular inconsistencies and contradictory patterns
4. Generate visualizations of claim relationships with consistency scores

Based on the research paper: "Prospects for inconsistency detection using large language models and sheaves" by Steve Huntsman, Michael Robinson, and Ludmilla Huntsman.

## Key Features

- **LLM-Driven Analysis**: Uses GPT models to understand semantic relationships without relying on rigid pattern matching
- **Comprehensive Consistency Evaluation**: Evaluates both direct and transitive inconsistencies
- **Interactive Visualizations**: Generates color-coded graphs showing relationships and inconsistency cycles
- **Detailed Scoring**: Provides numerical consistency scores for all claim pairs

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
│   │   ├── openai_service.py # Language model integration
│   │   └── detector.py       # Inconsistency detection logic
│   └── utils/
│       ├── __init__.py
│       └── sheaf.py          # Graph analysis utilities
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

   Edit `.env` with your Azure OpenAI or OpenAI credentials:

   ```
   AZURE_OPENAI_API_KEY=your-key-here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_OPENAI_DEPLOYMENT=your-deployment-name
   ```

   Or for standard OpenAI:

   ```
   OPENAI_API_KEY=your-key-here
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
    "consistency_score": 4.5,
    "claims": [
        "Claim 1 extracted from the prompt",
        "Claim 2 extracted from the prompt"
    ],
    "cycles": [
        [0, 1, 2]
    ],
    "inconsistent_pairs": [
        {
            "cycle": [0, 1, 2],
            "description": "Claim 1 → Claim 2 → Claim 3 → Claim 1"
        }
    ],
    "pairwise_consistency": {
        "0-1": 8.5,
        "0-2": 3.2,
        "1-2": 7.0
    },
    "visualization_url": "http://localhost:8000/visualizations/12345.png"
}
```

### Response Fields

- **consistency_score**: Overall consistency rating from 0-10 (0=inconsistent, 10=consistent)
- **claims**: List of individual claims extracted from the prompt
- **cycles**: Lists of claim indices that form inconsistency cycles
- **inconsistent_pairs**: Detailed descriptions of detected inconsistencies
- **pairwise_consistency**: Consistency scores for each analyzed claim pair
- **visualization_url**: Complete URL to the generated visualization image

## Examples

### Simple Contradiction Example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I am older than you and you are older than me",
    "visualization": true
  }'
```

### Circular Logic Example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I have a question: if I eat more than my little dog, but my little dog eats more than Ana'\''s, Ana'\''s eats more than Juan'\''s, Juan'\''s eats more than Miguel'\''s, and Miguel'\''s little dog eats more than Miguel himself. If we know for sure that Miguel eats much more than I do, who eats the most?",
    "visualization": true
  }'
```

## Visualization Features

The API generates interactive visualizations showing:

- **Green edges**: Consistent relationships (score ≥ 5/10)
- **Orange edges**: Inconsistent relationships (score < 5/10)
- **Red edges**: Inconsistency cycle edges
- **Edge labels**: Numerical consistency scores (e.g., "7.5/10")

## Implementation Details

- **Language Model Integration**: Uses GPT-4o mini or similar models to evaluate logical consistency
- **Advanced JSON Parsing**: Robust handling of LLM responses with error correction
- **Graph-Based Analysis**: Uses NetworkX for graph construction and cycle detection
- **Matplotlib Visualizations**: Generates clear, informative visualizations of claim relationships

## License

MIT License

## References

Huntsman, S., Robinson, M., & Huntsman, L. (2024). Prospects for inconsistency detection using large language models and sheaves. arXiv:2401.16713v1.