# Testing layers

The repository intentionally separates dependency-light correctness checks from
execution stacks that pull in larger third-party runtimes.

## Core baseline

```bash
python -m unittest discover -s tests -v
```

This layer avoids Paddle, CUDA, PyMuPDF, and Pillow. It covers configuration,
path safety, naming/schema compatibility, manifests, atomic writes, and CLI
surfaces that do not execute PDF/OCR engines.

## PDF execution smoke

```bash
python -m pip install '.[pdf]'
python -m unittest discover -s tests_pdf -v
```

This layer creates all fixtures at runtime and verifies actual PDF rendering,
searchable-PDF reconstruction, hidden-text extraction, CLI wiring, incomplete
input rejection, and transactional output behavior.

## Future execution layers

Paddle/PaddleX CPU smoke tests and GPU/self-hosted benchmarks remain separate so
public CI does not pretend to validate CUDA environments it does not have.
