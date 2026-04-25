import ast

import pandas as pd


def parse_list_cell(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        return [text]
