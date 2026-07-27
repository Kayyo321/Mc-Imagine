# Contributing to Mc-Imagine

Welcome! We're thrilled that you'd like to contribute to Mc-Imagine. This document provides guidelines for contributing to both the Minecraft mod and the Python training pipeline.

## How to Report Bugs
When filing an issue, please include:
- A clear, descriptive title.
- Your Minecraft version, Mod Loader (Fabric/Forge) version, and Mc-Imagine version.
- Detailed steps to reproduce the bug.
- Any crash reports or log files (`latest.log`).

## How to Suggest Features
We are open to new ideas! Open a feature request issue and explain:
- What the feature is.
- Why it would be useful.
- How you envision it working.

## Development Setup
### Prerequisites
- **Java**: JDK 17
- **Build Tool**: Gradle
- **Python**: Python 3.11 or higher

### Mod Setup
Follow the standard Architectury mod setup instructions in the `mod/` directory.

### Pipeline Setup
```bash
cd pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Code Style
- **Java**: We follow the [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html).
- **Python**: We use [Black](https://github.com/psf/black) for formatting and [Ruff](https://github.com/astral-sh/ruff) for linting. Please ensure your code passes both before submitting.

## Pull Request Process
1. Fork the repository and create your branch from `main`.
2. Write clean, documented code and include tests where appropriate.
3. Ensure the build passes.
4. Open a Pull Request with a clear description of the changes.
5. Participate in the code review process.

## License Agreement
By contributing to this project, you agree that your contributions will be licensed under its GNU General Public License v3.0 (GPLv3).
