---
name: "\U0001F41B Bug Report"
about: Create a report to help us reproduce and fix the bug
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
Please provide a clear and concise description of what the bug is.

**To Reproduce**
Code to reproduce the behavior:
```
# All necessary imports at the beginning
import pandas as pd
from mostlyai import qa
# A succinct reproducing example trimmed down to the essential parts:
df = pd.DataFrame({'x': [1, 2, 3]})
qa.report(syn_tgt_df=df, trn_tgt_df=df)
```

**Expected behavior**
A clear and concise description of what you expected to happen.
