# --------------------------
# DEFAULT
# --------------------------

all: install

# --------------------------
# INSTALL
# --------------------------

install:
	uv sync

# --------------------------
# RUN
# --------------------------

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

# --------------------------
# CLEAN
# --------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# --------------------------
# FCLEAN
# --------------------------

fclean: clean
	@rm -rf .venv
	@echo "💣 Virtual environment removed"

# --------------------------
# LINT
# --------------------------

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# --------------------------
# PHONY
# --------------------------

.PHONY: all install run debug fclean clean lint lint-strict