import json
from pathlib import Path
from unittest.mock import patch

from src.utils.config_manager import ConfigManager


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_config_manager_falls_back_to_example_file(tmp_path):
    write_json(
        tmp_path / "neo4j_config.example.json",
        {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "CHANGE_ME_LOCAL_ONLY",
            "database": "neo4j",
        },
    )

    with patch.dict("os.environ", {}, clear=True):
        manager = ConfigManager(config_dir=str(tmp_path))
        assert manager.neo4j_config["password"] == "CHANGE_ME_LOCAL_ONLY"


def test_config_manager_prefers_local_file_and_env_overrides(tmp_path):
    write_json(tmp_path / "neo4j_config.example.json", {"password": "example"})
    write_json(tmp_path / "neo4j_config.local.json", {"password": "local"})

    with patch.dict("os.environ", {"CITEWEAVE_NEO4J_PASSWORD": "env"}, clear=True):
        manager = ConfigManager(config_dir=str(tmp_path))
        assert manager.neo4j_config["password"] == "env"
