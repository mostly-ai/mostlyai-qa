name: 🐛 Bug Report
description: Create a report to help us reproduce and fix the bug

body:
- type: textarea
  attributes:
    label: 🐛 Describe the bug
    description: |
      Please provide a clear and concise description of what the bug is.

      If relevant, add a minimal example so that we can reproduce the error by running the code. It is very important for the snippet to be as succinct (minimal) as possible, so please take time to trim down any irrelevant code to help us debug efficiently. We are going to copy-paste your code and we expect to get the same result as you did: avoid any external data, and include the relevant imports, etc. For example:

      ```python
      # All necessary imports at the beginning
      import pandas as pd
      from mostlyai import qa

      # A succinct reproducing example trimmed down to the essential parts:
      df = pd.DataFrame({'x': [1, 2, 3]}
      qa.report(syn_tgt_df=df, trn_tgt_df=df)
        ```

        Please also paste or describe the results you observe instead of the expected results. If you observe an error, please paste the error message including the **full** traceback of the exception. It may be relevant to wrap error messages in ```` ```triple quotes blocks``` ````.
    placeholder: |
      A clear and concise description of what the bug is.

      ```python
      # Sample code to reproduce the problem
      ```

      ```
      The error message you got, with the full traceback.
      ```
    validations:
      required: true
- type: textarea
  attributes:
    label: Versions
    description: |
      Please run the following and paste the output below.
      ```sh
-     poetry show
      ```
  validations:
    required: true
- type: markdown
  attributes:
    value: >
      Thanks for contributing 🎉!