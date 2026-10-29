import pathlib

import pytest

from querypilot import evalset, warehouse
from querypilot.catalog import Catalog
from querypilot.executor import Executor
from querypilot.policy import PolicyEngine


@pytest.fixture(scope="session")
def db(tmp_path_factory) -> pathlib.Path:
    return warehouse.build(tmp_path_factory.mktemp("wh") / "warehouse.duckdb")


@pytest.fixture(scope="session")
def catalog(db):
    return Catalog.load(db)


@pytest.fixture(scope="session")
def policies(catalog):
    return PolicyEngine.load(catalog)


@pytest.fixture(scope="session")
def executor(db):
    return Executor(db)


@pytest.fixture(scope="session")
def eval_files(tmp_path_factory):
    return evalset.build(tmp_path_factory.mktemp("eval"))
