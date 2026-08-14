PY := .venv/bin/python
PYTEST := .venv/bin/pytest

.PHONY: test
test:
	$(PYTEST) -v -m "not slow"

.PHONY: test-all
test-all:
	$(PYTEST) -v

.PHONY: clean-features
clean-features:
	rm -rf data/features
