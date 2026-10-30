"""Tests - API endpoint tests using pytest"""

import uuid

from fastapi.testclient import TestClient
from app.main import app
from app.agents.nodes.responder import NOT_FOUND_RESPONSE

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
        """Test empty query rejection (schema validation)"""
        response = client.post("/query", json={
            "query": ""
        })
        assert response.status_code == 422

    def test_query_whitespace(self):
        """Test whitespace-only query rejection"""
        response = client.post("/query", json={
            "query": "   "
        })
        assert response.status_code == 400

    def test_query_too_long(self):
        """Test query length validation (schema validation)"""
        long_query = "a" * 2001
        response = client.post("/query", json={
            "query": long_query
        })
        assert response.status_code == 422
    
    def test_query_with_session_id(self):
        """Test query with session ID"""
        response = client.post("/query", json={
            "query": "Test query",
            "session_id": "test_session_123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_123"

    def test_query_remembers_session(self):
        """Queries with the same session ID remember earlier turns; other sessions don't"""
        session_id = f"pytest_{uuid.uuid4().hex}"
        client.post("/query", json={"query": "Hi! My name is Priya.", "session_id": session_id})

        response = client.post("/query", json={"query": "What did I say my name was?", "session_id": session_id})
        assert response.status_code == 200
        assert "Priya" in response.json()["response"]

        response = client.post("/query", json={
            "query": "What did I say my name was?",
            "session_id": f"pytest_{uuid.uuid4().hex}"
        })
        assert response.status_code == 200
        assert "Priya" not in response.json()["response"]

    def test_query_response_structure(self):
        """Test query response structure"""
        response = client.post("/query", json={
            "query": "What is AI?"
        })
        data = response.json()
        
        required_fields = ["query", "response", "classification", "sources", "reasoning"]
        for field in required_fields:
            assert field in data


class TestDocumentEndpoints:
    """Test upload → list → query → delete"""

    FILENAME = "pytest_upload_sample.txt"
    CONTENT = b"Project Heron kickoff is on 2 June 2027. The project lead is Dana Ortiz."

    def upload(self):
        return client.post("/upload", files=[("files", (self.FILENAME, self.CONTENT, "text/plain"))])

    def test_document_lifecycle(self):
        """Uploaded documents are listed, answer questions, re-upload replaces, delete removes"""
        try:
            response = self.upload()
            assert response.status_code == 200
            assert response.json()["files"] == [{"filename": self.FILENAME, "chunks": 1}]

            # Re-uploading replaces instead of duplicating
            assert self.upload().status_code == 200
            documents = client.get("/documents").json()["documents"]
            assert {"filename": self.FILENAME, "chunks": 1} in documents

            # Answers come from the uploaded document and cite it as [1]
            session_id = f"pytest_{uuid.uuid4().hex}"
            response = client.post("/query", json={
                "query": "When is the Project Heron kickoff?",
                "session_id": session_id
            })
            assert response.status_code == 200
            data = response.json()
            assert data["sources"][0]["source"] == self.FILENAME
            assert "2027" in data["response"]
            assert "[1]" in data["response"]

            # A follow-up in the same session is rewritten to stand alone, then answered from the document
            response = client.post("/query", json={"query": "Who leads that project?", "session_id": session_id})
            assert response.status_code == 200
            data = response.json()
            assert "Read follow-up as" in data["reasoning"]
            assert data["sources"][0]["source"] == self.FILENAME
            assert "Dana Ortiz" in data["response"]

            # Questions the library can't answer get a not-found reply, not general knowledge
            response = client.post("/query", json={
                "query": "What does the uploaded document say about submarine engine warranties?"
            })
            assert response.status_code == 200
            data = response.json()
            assert data["classification"] == "TECHNICAL"
            assert data["sources"] == []
            assert data["response"] == NOT_FOUND_RESPONSE
        finally:
            response = client.delete(f"/documents/{self.FILENAME}")

        assert response.status_code == 200
        assert response.json()["deleted_chunks"] == 1
        names = [d["filename"] for d in client.get("/documents").json()["documents"]]
        assert self.FILENAME not in names

    def test_upload_unsupported_type(self):
        """Test unsupported files are rejected"""
        response = client.post("/upload", files=[("files", ("image.png", b"\x89PNG", "image/png"))])
        assert response.status_code == 400

    def test_delete_missing_document(self):
        """Test deleting an unknown document returns 404"""
        response = client.delete("/documents/does_not_exist.pdf")
        assert response.status_code == 404


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
        assert "qdrant" in data
        assert "point_count" in data["qdrant"]


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
