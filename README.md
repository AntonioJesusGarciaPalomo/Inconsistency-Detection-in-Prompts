# Inconsistency Detection API

This service detects logical inconsistencies in text prompts using advanced language models to identify contradictions, circular logic, and other logical flaws.

## Overview

The API analyzes text prompts to:

1. Extract individual claims using language model capabilities
2. Assess logical relationships and consistency between claims
3. Detect circular inconsistencies and contradictory patterns
4. Generate static or interactive visualizations of claim relationships

Based on the research paper: "Prospects for inconsistency detection using large language models and sheaves" by Steve Huntsman, Michael Robinson, and Ludmilla Huntsman.

## Key Features

- **LLM-Driven Analysis**: Uses GPT models to understand semantic relationships without relying on rigid pattern matching
- **Comprehensive Consistency Evaluation**: Evaluates both direct and transitive inconsistencies
- **Multiple Visualization Options**: Generates both static and interactive visualizations of inconsistency networks
- **Detailed Scoring**: Provides numerical consistency scores for relationships between claims
- **Pattern Detection**: Uses linguistic pattern analysis to detect comparative and transitive relationships

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
│       ├── graph_analyzer.py           # Static visualization generator
│       └── interactive_graph_analyzer.py # Interactive visualization generator
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

   This includes PyVis for interactive network visualizations.

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

   The API will be available at `http://localhost:8000`.

## API Usage

### Analyze Prompt Endpoint

**POST** `/api/analyze`

#### Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| prompt | string | The text prompt to analyze for inconsistencies |
| visualization | boolean | Whether to generate a visualization (default: false) |
| visualization_type | string | Type of visualization to generate ('static' or 'interactive', default: 'static') |

#### Example Request

```json
{
    "prompt": "I like chocolate and I don't like sugar and sugar is the most delicious thing for me",
    "visualization": true,
    "visualization_type": "interactive"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| consistency_score | float | Overall consistency rating from 0-10 (0=inconsistent, 10=consistent) |
| claims | array | List of individual claims extracted from the prompt |
| cycles | array | Lists of claim indices that form inconsistency cycles |
| inconsistent_pairs | array | Detailed descriptions of detected inconsistencies |
| visualization_url | string | URL to the generated visualization |
| visualization_type | string | Type of visualization generated ('static' or 'interactive') |
| error | string | Error message (null if no error occurred) |

#### Example Response

```json
{
    "consistency_score": 4.0,
    "claims": [
        "I like chocolate.",
        "I don't like sugar.",
        "Sugar is the most delicious thing for me."
    ],
    "cycles": [[1, 2]],
    "inconsistent_pairs": [
        {
            "cycle": [1, 2],
            "description": "I don't like sugar. → Sugar is the most delicious thing for me. → I don't like sugar."
        }
    ],
    "visualization_url": "http://localhost:8000/visualizations/inconsistency_8643dc23.html",
    "visualization_type": "interactive",
    "error": null
}
```

### Visualization Endpoint

**GET** `/visualizations/{filename}`

Returns the visualization file (PNG or HTML) for the given filename.

## Visualization Types

The API supports two types of visualizations:

### 1. Static Visualizations (PNG)

- Generated using Matplotlib and NetworkX
- Shows claims as nodes with text labels
- Green edges represent consistent relationships
- Red dashed edges represent inconsistent relationships
- Edge labels show consistency scores

Example request:
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I like chocolate and I don't like sugar and sugar is the most delicious thing for me",
    "visualization": true,
    "visualization_type": "static"
  }'
```

### 2. Interactive Visualizations (HTML)

- Generated using PyVis (based on vis.js)
- Allows dragging, zooming, and interactive exploration of the graph
- Hovering over nodes shows the full text of claims
- Green edges represent consistent relationships
- Red dashed edges represent inconsistent relationships
- All relationships are undirected (no arrows)
- Includes a detailed explanation of detected inconsistency cycles
- Contains a full listing of all claims for reference

Example request:
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I like chocolate and I don't like sugar and sugar is the most delicious thing for me",
    "visualization": true,
    "visualization_type": "interactive"
  }'
```

## Inconsistency Detection Process

The API follows these steps to detect inconsistencies:

1. **Claim Extraction**: Uses LLM to extract individual claims from the prompt
2. **Consistency Analysis**: Evaluates logical relationships between claims
3. **Cycle Detection**: Identifies circular inconsistencies in the relationship network
4. **Visualization**: Generates a visual representation of the claim network

The system employs multiple strategies for detecting inconsistencies:

- **Direct LLM Analysis**: Primary method using language model to evaluate logical consistency
- **Pattern Detection**: Identifies comparative patterns in claims (e.g., "more than", "better than")
- **Graph Analysis**: Uses network analysis to detect cycles in the relationship graph

## Examples

### Simple Contradiction Example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I am older than you and you are older than me",
    "visualization": true,
    "visualization_type": "interactive"
  }'
```

### Circular Logic Example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I have a question: if I eat more than my little dog, but my little dog eats more than Ana'\''s, Ana'\''s eats more than Juan'\''s, Juan'\''s eats more than Miguel'\''s, and Miguel'\''s little dog eats more than Miguel himself. If we know for sure that Miguel eats much more than I do, who eats the most?",
    "visualization": true,
    "visualization_type": "interactive"
  }'
```

### Preference Contradiction Example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I like chocolate and I like sweets and I don'\''t like sugar and sugar is the most delicious thing for me",
    "visualization": true,
    "visualization_type": "interactive"
  }'
```

## Working with Visualizations

After receiving the API response, you can view the visualization by:

1. Opening the URL provided in the `visualization_url` field in your web browser
2. For interactive visualizations, you can:
   - Drag nodes to rearrange the graph
   - Zoom in/out using the mouse wheel
   - Hover over nodes to see the full text of claims
   - Double-click on a node to focus on its connections

NOTE: If the URL in the response contains `0.0.0.0`, replace it with `localhost` to access the visualization.

## Implementation Details

- **Language Model Integration**: Uses GPT models to evaluate logical consistency
- **Advanced JSON Parsing**: Robust handling of LLM responses with multi-stage parsing and error correction
- **Graph-Based Analysis**: Uses NetworkX for graph construction and cycle detection
- **Visualization**: 
  - Static: Uses Matplotlib for generating PNG visualizations
  - Interactive: Uses PyVis for generating interactive HTML visualizations

## License

MIT License

## References

Huntsman, S., Robinson, M., & Huntsman, L. (2024). Prospects for inconsistency detection using large language models and sheaves. arXiv:2401.16713v1.
