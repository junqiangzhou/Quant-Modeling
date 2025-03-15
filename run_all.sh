#!/bin/bash

python3 -m data.data_fetcher .

python3 -m model.train .

python3 -m model.eval .

python3 -m backtest.backtest .