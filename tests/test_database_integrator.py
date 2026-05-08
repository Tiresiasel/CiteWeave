from src.storage.database_integrator import DatabaseIntegrator


def test_database_integrator_keeps_config_path_for_vector_indexer():
    integrator = DatabaseIntegrator(config_path="config", storage_root="data/papers")

    assert integrator.config_path == "config"
    assert integrator.storage_root == "data/papers"
