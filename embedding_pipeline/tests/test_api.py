import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def make_pubsub_body(expedition_ids: list[int]) -> dict:
    """
    Build a Pub/Sub push envelope with the given expedition IDs.
    """
    data = base64.b64encode(
        json.dumps({"expedition_ids": expedition_ids}).encode()
    ).decode()
    return {
        "message": {
            "data": data,
            "messageId": "test-msg-001",
            "publishTime": "2026-02-22T10:00:00Z",
        },
        "subscription": "projects/deepsummit/subscriptions/test-sub",
    }


@pytest.fixture
def client():
    from main import app, get_session

    # Override the DB session dependency so API tests need no real database.
    app.dependency_overrides[get_session] = lambda: MagicMock()
    yield TestClient(app)
    app.dependency_overrides.clear()


@patch("main.pipeline.process")
def test_valid_message_returns_204(mock_process, client):
    mock_process.return_value = None

    response = client.post("/", json=make_pubsub_body([1, 2, 3]))

    assert response.status_code == 204
    mock_process.assert_called_once()


@patch("main.pipeline.process")
def test_empty_expedition_ids_returns_204_without_processing(mock_process, client):
    response = client.post("/", json=make_pubsub_body([]))

    assert response.status_code == 204
    mock_process.assert_not_called()


def test_missing_message_key_returns_400(client):
    response = client.post("/", json={"not_a_message": {}})

    assert response.status_code == 400


def test_invalid_base64_data_returns_400(client):
    response = client.post("/", json={
        "message": {"data": "!!!not-base64!!!", "messageId": "x"}
    })

    assert response.status_code == 400


def test_missing_expedition_ids_key_returns_400(client):
    data = base64.b64encode(json.dumps({"wrong_key": []}).encode()).decode()
    response = client.post("/", json={
        "message": {"data": data, "messageId": "x"}
    })

    assert response.status_code == 400
