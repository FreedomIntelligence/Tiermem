# Contributing to TierMem

Thank you for your interest in contributing to TierMem! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Features

We love feature suggestions! Please open an issue with:
- A clear description of the feature
- Use cases and benefits
- Any implementation ideas (optional)

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/

   # Test on a small benchmark
   python test_TierMem_locomo_multi.py --limit 5
   ```

4. **Commit your changes**
   - Use clear, descriptive commit messages
   - Reference any related issues

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/TierMem.git
cd TierMem

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Start Qdrant for testing
./start_qdrant.sh
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

- Add tests for new features
- Ensure existing tests pass
- Aim for good test coverage

## Questions?

Feel free to open an issue if you have questions about contributing!

---

Thank you for helping make TierMem better!
