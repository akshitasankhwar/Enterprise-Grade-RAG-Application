"""Tests - API endpoint tests using pytest"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_status_code(self):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self):
        """Test health response has required fields"""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "message" in data
        assert "services" in data
    
    def test_health_services(self):
        """Test health includes service status"""
        response = client.get("/health")
        data = response.json()
        
        assert "qdrant" in data["services"]
        assert "settings" in data["services"]


class TestQueryEndpoint:
    """Test query endpoint"""
    
    def test_query_basic(self):
        """Test basic query"""
        response = client.post("/query", json={
            "query": "Hello"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "classification" in data
        assert "sources" in data
    
    def test_query_empty(self):
        """Test empty query rejection"""
        response = client.post("/query", json={
            "query": ""
        })
        assert response.status_code == 400
    
    def test_query_too_long(self):
        """Test query length validation"""
        long_query = "a" * 2001
        response = client.post("/query", json={
            "query": long_query
        })
        assert response.status_code == 400
    
    def test_query_with_session_id(self):
        """Test query with session ID"""
        response = client.post("/query", json={
            "query": "Test query",
            "session_id": "test_session_123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_123"
    
    def test_query_response_structure(self):
        """Test query response structure"""
        response = client.post("/query", json={
            "query": "What is AI?"
        })
        data = response.json()
        
        required_fields = ["query", "response", "classification", "sources", "reasoning"]
        for field in required_fields:
            assert field in data


class TestInfoEndpoint:
    """Test info endpoint"""
    
    def test_info_status_code(self):
        """Test info endpoint returns 200"""
        response = client.get("/info")
        assert response.status_code == 200
    
    def test_info_response_structure(self):
        """Test info response has required fields"""
        response = client.get("/info")
        data = response.json()
        
        assert "api_name" in data
        assert "version" in data
        assert "project" in data
        assert "qdrant" in data


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_status_code(self):
        """Test root endpoint returns 200"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_has_endpoints(self):
        """Test root response documents endpoints"""
        response = client.get("/")
        data = response.json()
        
        assert "endpoints" in data or "message" in data


# Manual test runner
if __name__ == "__main__":
    print("Running API tests...\n")
    
    # Test health
    print("Testing /health...")
    response = client.get("/health")
    print(f"  Status: {response.status_code}")
    print(f"  Data: {response.json()}\n")
    
    # Test root
    print("Testing /...")
    response = client.get("/")
    print(f"  Status: {response.status_code}\n")
    
    # Test info
    print("Testing /info...")
    response = client.get("/info")
    print(f"  Status: {response.status_code}\n")
    
    # Test query
    print("Testing /query...")
    response = client.post("/query", json={
        "query": "Hello, how are you?"
    })
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Query: {data['query']}")
        print(f"  Classification: {data['classification']}")
        print(f"  Response: {data['response'][:100]}...")
    else:
        print(f"  Error: {response.json()}")
