# Examples

This `examples` directory contains scripts that demonstrate
how to use the toolkit for processing OHLC data.

## Prerequisites

First, clone the `ohlc-toolkit` repository:

```bash
git clone https://github.com/visikai/ohlc-toolkit.git
```

Set up a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install poetry
poetry install
```

### Getting a Sample Dataset

> New: We now have a helper method in the example script which will automatically download a sample dataset for you.
> See below for details. Skip to [Running the Example](#running-the-example) if you just want to get started.

Before running the examples, you need a sample dataset to work with.

On a [separate repo](https://github.com/ff137/bitstamp-btcusd-minute-data), we host up-to-date BTC/USD 1-minute data from Bitstamp,
which can be found at the following link:

<https://github.com/ff137/bitstamp-btcusd-minute-data/blob/main/data/updates/btcusd_bitstamp_1min_latest.csv>

Inspect the above link to verify that you're happy downloading the data.

We recommend the above dataset because it is real, recent, and has passed
data integrity checks. You're of course welcome to use your own dataset instead.

#### Automated Download

We have included a helper class, `DatasetDownloader`, in the toolkit to help automate the download process.

Here is how you can use it (this is included in the example script and doesn't need to be done manually):

```python
from ohlc_toolkit.bitstamp_dataset_downloader import DatasetDownloader

# Initialize the downloader
downloader = DatasetDownloader(data_dir="data")  # Configure your desired output directory

# Download the latest dataset
df = downloader.download_bitstamp_btcusd_minute_data(recent=True, bulk=False)  # Set bulk to True to also download the full historical dataset (~90MB)
```

#### Manual Download

You can also download the data manually using curl:

```bash
cd ohlc-toolkit  # Move to the project
mkdir -p data  # Create a data directory

curl -L -o data/btcusd_bitstamp_1min_latest.csv https://github.com/ff137/bitstamp-btcusd-minute-data/raw/main/data/updates/btcusd_bitstamp_1min_latest.csv
```

## Running the Example

Once you have downloaded the data, you can run the `basic_usage.py` script
to see how the `ohlc-toolkit` processes the data.

### Instructions

Run the following command in your terminal:

```bash
python examples/basic_usage.py
```

### What the Script Does

The [basic_usage.py](basic_usage.py) script performs the following actions:

- Imports the `ohlc_toolkit` module.
- Downloads a sample dataset from the [bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data) repository.
- Reads the OHLC data from the CSV file using the `read_ohlc_csv` function.
- Transforms the OHLC data to different timeframes.

That's it for now! More features will be added soon.
