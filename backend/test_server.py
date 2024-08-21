import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_upload_file():
    with open("test.pdf", "rb") as f:
        response = client.post("/uploadfile/", files={"file": f})
    assert response.status_code == 200
    assert response.json()["filename"] == "test.pdf"