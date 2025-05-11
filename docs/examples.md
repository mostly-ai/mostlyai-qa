---
hide:
  - navigation
---

# Usage Examples

## Baseball Players (Flat Data)

### Case 1: Players Table only

```python
import pandas as pd
import webbrowser
from mostlyai import qa

repo = "https://github.com/mostly-ai/mostlyai-qa"
path = "/raw/refs/heads/main/examples/baseball-players"

cols = ["birthDate", "deathDate", "weight", "height", "bats", "throws"]
report_path, metrics = qa.report(
    syn_tgt_data=pd.read_parquet(f"{repo}/{path}/synthetic-target.pqt")[cols],
    trn_tgt_data=pd.read_parquet(f"{repo}/{path}/training-target.pqt")[cols],
    hol_tgt_data=pd.read_parquet(f"{repo}/{path}/holdout-target.pqt")[cols],
    report_subtitle=" for Baseball Players",
    report_path="baseball-players.html",
)
print(metrics.model_dump_json(indent=4))

webbrowser.open(f"file://{report_path.absolute()}")
```

👉 [HTML Report](https://html-preview.github.io/?url=https://github.com/mostly-ai/mostlyai-qa/blob/main/examples/baseball-players.html)

### Case 2: Players Table with Context

```python
import pandas as pd
import webbrowser
from mostlyai import qa

repo = "https://github.com/mostly-ai/mostlyai-qa"
path = "/raw/refs/heads/main/examples/baseball-players"

report_path, metrics = qa.report(
    syn_tgt_data=pd.read_parquet(f"{repo}/{path}/synthetic-target.pqt"),
    syn_ctx_data=pd.read_parquet(f"{repo}/{path}/synthetic-context.pqt"),
    trn_tgt_data=pd.read_parquet(f"{repo}/{path}/training-target.pqt"),
    trn_ctx_data=pd.read_parquet(f"{repo}/{path}/training-context.pqt"),
    hol_tgt_data=pd.read_parquet(f"{repo}/{path}/holdout-target.pqt"),
    hol_ctx_data=pd.read_parquet(f"{repo}/{path}/holdout-context.pqt"),
    tgt_context_key="id",
    ctx_primary_key="id",
    report_subtitle=" for Baseball Players",
    report_path="baseball-players-with-context.html",
)
print(metrics.model_dump_json(indent=4))

webbrowser.open(f"file://{report_path.absolute()}")
```
👉 [HTML Report](https://html-preview.github.io/?url=https://github.com/mostly-ai/mostlyai-qa/blob/main/examples/baseball-players-with-context.html)

## Baseball Seasons (Sequential Data)

### Case 1: Seasons Table only

```python
import pandas as pd
import webbrowser
from mostlyai import qa

repo = "https://github.com/mostly-ai/mostlyai-qa"
path = "/raw/refs/heads/main/examples/baseball-seasons"

report_path, metrics = qa.report(
    syn_tgt_data=pd.read_parquet(f"{repo}/{path}/synthetic-target.pqt"),
    trn_tgt_data=pd.read_parquet(f"{repo}/{path}/training-target.pqt"),
    hol_tgt_data=pd.read_parquet(f"{repo}/{path}/holdout-target.pqt"),
    tgt_context_key="players_id",
    report_subtitle=" for Baseball Seasons",
    report_path="baseball-seasons.html",
)
print(metrics.model_dump_json(indent=4))

webbrowser.open(f"file://{report_path.absolute()}")
```

👉 [HTML Report](https://html-preview.github.io/?url=https://github.com/mostly-ai/mostlyai-qa/blob/main/examples/baseball-seasons.html)

### Case 2: Seasons Table with Context

```python
import pandas as pd
import webbrowser
from mostlyai import qa

repo = "https://github.com/mostly-ai/mostlyai-qa"
path = "/raw/refs/heads/main/examples/baseball-seasons"

report_path, metrics = qa.report(
    syn_tgt_data=pd.read_parquet(f"{repo}/{path}/synthetic-target.pqt"),
    syn_ctx_data=pd.read_parquet(f"{repo}/{path}/synthetic-context.pqt"),
    trn_tgt_data=pd.read_parquet(f"{repo}/{path}/training-target.pqt"),
    trn_ctx_data=pd.read_parquet(f"{repo}/{path}/training-context.pqt"),
    hol_tgt_data=pd.read_parquet(f"{repo}/{path}/holdout-target.pqt"),
    hol_ctx_data=pd.read_parquet(f"{repo}/{path}/holdout-context.pqt"),
    tgt_context_key="players_id",
    ctx_primary_key="id",
    report_subtitle=" for Baseball Seasons",
    report_path="baseball-seasons-with-context.html",
)
print(metrics.model_dump_json(indent=4))

webbrowser.open(f"file://{report_path.absolute()}")
```

👉 [HTML Report](https://html-preview.github.io/?url=https://github.com/mostly-ai/mostlyai-qa/blob/main/examples/baseball-seasons-with-context.html)
