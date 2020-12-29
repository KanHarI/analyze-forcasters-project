# analyze-forcasters-project

Running:
* Create a venv
* Install package
* Run
```
py -3 -m venv venv
pip install .
analyze_forecasts --predictions-csv data\ForecastsAnonymized.csv --truth-csv data\GroundTruth.csv --end-only-1st-pred F217 --start-only-2nd-pred F260 --size-1st-questions 59 --size-2nd-questions 39
```