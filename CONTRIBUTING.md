# Contributing to Synthetic Data Quality Assurance

Thanks for your interest in contributing to Synthetic Data Quality Assurance! Follow these guidelines to set up your environment and streamline your contributions.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mostly-ai/mostlyai-qa.git
   cd mostlyai-qa
   ```
   If you don’t have direct write access to `mostlyai-qa`, fork the repository first and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/mostlyai-qa.git
   cd mostlyai-qa
   ```

2. **Install `uv` (if not installed already)**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   For alternative installation methods, visit the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Create a virtual environment and install dependencies**:
   ```bash
   uv sync --frozen --python=3.10
   source .venv/bin/activate
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Ensure your local `main` branch is up to date**:
   ```bash
   git checkout main
   git reset --hard origin/main
   git pull origin main
   ```

2. **Create a new feature or bugfix branch**:
   ```bash
   git checkout -b my-feature-branch
   ```

3. **Implement your changes.**

4. **Run tests and pre-commit hooks**:
   ```bash
   pytest
   pre-commit run
   ```

5. **Commit your changes with a descriptive message**:
   ```bash
   git add .
   git commit -m "feat: add a clear description of your feature"
   ```
   Follow the [Conventional Commits](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13) format.

6. **Push your changes**:
   ```bash
   git push origin my-feature-branch
   ```

7. **Open a pull request on GitHub.**
