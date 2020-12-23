# Koopman ML

Because this was developed using Pymp in a Unix environment, some of this code cannot be run on Windows.
Our dataset was found on [Kaggle](https://www.kaggle.com/tencars/392-crypto-currency-pairs-at-minute-resolution)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages.

```bash
pip install -r requirements.txt
```

## Running

To calculate the epsilon values from the gEDMD method:
```bash
python generator.py
```

To list the previously calculated epsilon values:
```bash
python loader.py
```
