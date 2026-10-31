import os
import tempfile

os.environ["CHURNGUARD_API_KEY"] = "test-key"
os.environ["CHURNGUARD_LOG_DIR"] = tempfile.mkdtemp(prefix="churnguard_test_logs_")

import pytest

from churnguard import config


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def headers():
    return {"X-API-Key": "test-key"}


@pytest.fixture
def customer():
    return dict(config.SAMPLE_CUSTOMER)
