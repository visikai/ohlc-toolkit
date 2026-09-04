# OHLC Toolkit

[![PyPI](https://img.shields.io/pypi/v/ohlc-toolkit)](https://pypi.org/project/ohlc-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/ohlc-toolkit.svg)](https://pypi.org/project/ohlc-toolkit/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0db6f73fe9bb4e8a8591055a6ea284f2)](https://app.codacy.com/gh/visikai/ohlc-toolkit/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/0db6f73fe9bb4e8a8591055a6ea284f2)](https://app.codacy.com/gh/visikai/ohlc-toolkit/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A flexible Python toolkit for working with OHLC (Open, High, Low, Close) market data.

## Installation

The project is available on [PyPI](https://pypi.org/project/ohlc-toolkit/):

```bash
pip install ohlc-toolkit
```

## Features

- Read OHLC data from CSV files into pandas DataFrames, with built-in data quality checks:

  ```py
    df = read_ohlc_csv(csv_file_path, timeframe="1d")
  ```

- Download BTCUSD 1-minute candle data in one line (using data from [ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data)):

  ```py
    df_1min = DatasetDownloader().download_bitstamp_btcusd_minute_data(bulk=True)
  ```

- Convert timeframe strings to the number of minutes, and vice versa:

  ```py
    # From string to minutes
    parse_timeframe("1h15m") == 75

    # From minutes to string
    format_timeframe(minutes=75) == "1h15m"
  ```

## Examples

See the [examples](examples/README.md) directory for examples of how to use the toolkit.

Run the example script to see how the toolkit works:

```bash
# Clone the repository
git clone https://github.com/visikai/ohlc-toolkit.git
cd ohlc-toolkit

# Install dependencies (creates .venv)
uv sync --all-groups

# Run the example script
uv run python examples/basic_usage.py
```

## Support

If you need any help or have any questions, please feel free to open an issue or contact me directly.

We hope this repo makes your life easier! If it does, please give us a star! ⭐
