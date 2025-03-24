import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, AsyncMock

from main import app
from app.core.schemas import AnalyzePromptResponse, InconsistentPair

client = TestClient(app)

@pytest.fixture
def mock_detector():
    """Create a mock for the inconsistency detector"""
    with patch('app.services.detector.InconsistencyDetector.analyze_prompt') as mock:
        yield mock

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_prompt_success(mock_detector):
    """Test the analyze prompt endpoint with a successful response"""
    # Setup the mock response
    mock_result = AnalyzePromptResponse(
        global_consistency_score=4.5,
        claims=["I eat more than my dog", "My dog eats more than Ana's dog"],
        pairwise_consistency={"0-1": 8.5},
        inconsistent_pairs=[
            InconsistentPair(
                claim1_index=0,
                claim2_index=1,
                claim1_text="I eat more than my dog",
                claim2_text="My dog eats more than Ana's dog",
                consistency_score=8.5
            )
        ],
        visualization_url="/visualizations/test.png"
    )
    mock_detector.return_value = mock_result

    # Call the API
    response = client.post(
        "/api/analyze",
        json={"prompt": "Test prompt", "visualization": True}
    )
    
    # Check the response
    assert response.status_code == 200
    result = response.json()
    assert result["global_consistency_score"] == 4.5
    assert len(result["claims"]) == 2
    assert result["visualization_url"] == "/visualizations/test.png"
    
    # Verify the mock was called correctly
    mock_detector.assert_called_once_with(prompt="Test prompt", generate_visualization=True)

def test_analyze_prompt_error(mock_detector):
    """Test the analyze prompt endpoint with an error response"""
    # Setup the mock to raise an exception
    mock_detector.side_effect = Exception("Test error")
    
    # Call the API
    response = client.post(
        "/api/analyze",
        json={"prompt": "Test prompt", "visualization": True}
    )
    
    # Check the response
    assert response.status_code == 500
    assert "Test error" in response.json()["detail"]