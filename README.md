# Synthetic Data Quality Assurance 🔎

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-qa/) [![stats](https://pepy.tech/badge/mostlyai-qa)](https://pypi.org/project/mostlyai-qa/) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai-qa) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-qa) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-qa)

[Documentation](https://mostly-ai.github.io/mostlyai-qa/) | [Sample Reports](#sample-reports) | [Technical White Paper](https://arxiv.org/abs/2504.01908)

Assess the fidelity and novelty of synthetic samples with respect to original samples:

1. calculate a rich set of accuracy, similarity and distance [metrics](https://mostly-ai.github.io/mostlyai-qa/api/#mostlyai.qa.metrics.ModelMetrics)
2. visualize statistics for easy comparison to training and holdout samples
3. generate a standalone, easy-to-share, easy-to-read HTML summary report

...all with a few lines of Python code 💥.

https://github.com/user-attachments/assets/b27e270a-f19c-4059-b4f2-ed209c9a26b9

## Installation

The latest release of `mostlyai-qa` can be installed via pip:

```bash
pip install -U mostlyai-qa
```

On Linux, one can explicitly install the CPU-only variant of torch together with `mostlyai-qa`:

```bash
pip install -U torch==2.6.0+cpu torchvision==0.21.0+cpu mostlyai-qa --extra-index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

```python
import pandas as pd
import webbrowser
from mostlyai import qa

# initialize logging to stdout
qa.init_logging()

# fetch original + synthetic data
base_url = "https://github.com/mostly-ai/mostlyai-qa/raw/refs/heads/main/examples/quick-start"
syn = pd.read_csv(f"{base_url}/census2k-syn_mostly.csv.gz")
# syn = pd.read_csv(f'{base_url}/census2k-syn_flip30.csv.gz') # a 30% perturbation of trn
trn = pd.read_csv(f"{base_url}/census2k-trn.csv.gz")
hol = pd.read_csv(f"{base_url}/census2k-hol.csv.gz")

# calculate metrics
report_path, metrics = qa.report(
    syn_tgt_data=syn,
    trn_tgt_data=trn,
    hol_tgt_data=hol,
)

# pretty print metrics
print(metrics.model_dump_json(indent=4))

# open up HTML report in new browser window
webbrowser.open(f"file://{report_path.absolute()}")
```

## Basic Usage

```python
from mostlyai import qa

# initialize logging to stdout
qa.init_logging()

# analyze single-table data
report_path, metrics = qa.report(
    syn_tgt_data = synthetic_df,
    trn_tgt_data = training_df,
    hol_tgt_data = holdout_df,  # optional
)

# analyze sequential data
report_path, metrics = qa.report(
    syn_tgt_data = synthetic_df,
    trn_tgt_data = training_df,
    hol_tgt_data = holdout_df,  # optional
    tgt_context_key = "user_id",
)

# analyze sequential data with context
report_path, metrics = qa.report(
    syn_tgt_data = synthetic_df,
    trn_tgt_data = training_df,
    hol_tgt_data = holdout_df,  # optional
    syn_ctx_data = synthetic_context_df,
    trn_ctx_data = training_context_df,
    hol_ctx_data = holdout_context_df,  # optional
    ctx_primary_key = "id",
    tgt_context_key = "user_id",
)
```

## Sample Reports

* [Baseball Players](https://html-preview.github.io/?url=https://github.com/mostly-ai/mostlyai-qa/blob/main/examples/baseball-players.html) (Flat Data)
* [Baseball Seasons](https://html-preview.github.io/?url=https://github.com/mostly-ai/mostlyai-qa/blob/main/examples/baseball-seasons-with-context.html) (Sequential Data)

## Citation

Please consider citing our project if you find it useful:

```bibtex
@misc{mostlyai-qa,
      title={Benchmarking Synthetic Tabular Data: A Multi-Dimensional Evaluation Framework},
      author={Andrey Sidorenko and Michael Platzer and Mario Scriminaci and Paul Tiwald},
      year={2025},
      eprint={2504.01908},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.01908},
}
```
