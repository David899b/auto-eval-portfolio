# {{cookiecutter.project_name}}

Auto-generated evaluation infrastructure via `auto-eval-platform`.

## Quick Start
```bash
# 1. Install
pip install -e .

# 2. Configure
cp configs/config.yaml.example configs/config.yaml
# Edit configs/config.yaml with your model endpoint, golden set path, etc.

# 3. Run evaluation
python -m auto_eval.cli evaluate --split test

# 4. View dashboard
streamlit run dashboard/app.py
```

## Structure
```
├── configs/           # Configuration (gitignored secrets)
├── data/              # Golden sets (versioned, frozen)
├── src/               # Custom evaluators, judges
├── tests/             # Regression tests + compliance suite
├── dashboard/         # Streamlit dashboard
└── .github/workflows/ # CI/CD gates
```

## Agents Enabled
- Golden Set Agent: {{cookiecutter.include_golden_set_agent}}
- Red Team Agent: {{cookiecutter.include_red_teaming}}
- Drift Agent: {{cookiecutter.include_drift_monitoring}}
- Compliance Agent: {{cookiecutter.include_compliance}}
