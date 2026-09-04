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

- Fetch the published BTCUSD 1-minute history, verifying every byte
  against the release's own manifest (data from
  [ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data)):

  ```py
    result = fetch_snapshot(
        SnapshotRelease(
            repository=BITSTAMP_BTCUSD_1M_REPOSITORY,
            tag="bitstamp-btcusd-1m-2026-08",
        ),
        "data/",
    )
  ```

- Parse and format compact duration strings:

  ```py
    Duration.parse("1h15m").total_seconds == 4500

    str(Duration(4500)) == "1h15m"
  ```

## Support

If you need any help or have any questions, please feel free to open an issue or contact me directly.

We hope this repo makes your life easier! If it does, please give us a star! ⭐
