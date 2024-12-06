import pandas as pd

from mostlyai.qa.sampling import pull_data_for_embeddings


def test_pull_data_for_embeddings_large_int(tmp_path):
    # regression test for issue with overly large integers
    df = pd.DataFrame({"cc": [123, 1800218404984585216]}, dtype="Int64")
    pull_data_for_embeddings(df_tgt=df)
