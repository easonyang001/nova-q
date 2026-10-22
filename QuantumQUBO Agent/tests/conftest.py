import os

# unit tests never load the embedding model or the harvested catalogue index (qubo_auto/annealing/catalog.py)
os.environ.setdefault("QUBO_NO_CATALOG", "1")
# the existing adapter tests drive the free-code path with scripted answers; plan mode has its own tests
os.environ.setdefault("QUBO_ADAPTER_MODE", "code")
