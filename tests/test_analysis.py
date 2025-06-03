# tests/test_analysis.py
import pandas as pd
from src.analysis import NewsEDA

def test_init():
    df = pd.DataFrame({
        'headline': ['Example headline'],
        'publisher': ['test@example.com'],
        'date': ['2020-01-01 12:00:00'],
        'url': ['http://example.com'],
        'stock': ['AAPL']
    })
    eda = NewsEDA(df)
    assert not eda.df.empty
    assert 'date' in eda.df.columns
    assert pd.api.types.is_datetime64_any_dtype(eda.df['date'])  # ensure date is parsed
    assert not eda.df['date'].isnull().any()  # no parsing failures
