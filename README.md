# Credit Risk Fuzzy Logic

A Mamdani fuzzy inference system, written from scratch in Python, that scores default risk from 0 to 100. Built for the Johns Hopkins course Reasoning Under Uncertainty. The write up is in `paper.pdf`.

## How it works

- **Inputs:** income, debt to income ratio, and credit score, described with overlapping fuzzy terms such as low, medium and high.
- **Rules:** 8 weighted rules, such as "if DTI is high and credit is poor, risk is high".
- **Output:** rule results are aggregated and turned into one score using centroid, bisector, or mean of maxima defuzzification.

It uses only the Python standard library.

```bash
python fuzzy_credit_risk.py
```

## Output

Running the script reproduces the paper's results table.

| Income ($k) | DTI (%) | Credit score | Centroid | Bisector | Mean of maxima |
| --- | --- | --- | --- | --- | --- |
| 45 | 38 | 610 | 84.06 | 84.25 | 85.75 |
| 150 | 12 | 780 | 14.77 | 14.50 | 11.62 |
| 60 | 25 | 650 | 50.00 | 50.00 | 50.00 |
| 30 | 45 | 580 | 83.36 | 83.25 | 84.25 |
| 90 | 18 | 720 | 16.82 | 16.75 | 16.12 |

## Limitations

This is a demonstration of interpretable reasoning, not a validated risk model. It uses five hand built borrower profiles and no real default outcomes, so predictive accuracy was not measured. The membership functions and rules are subjective and were not fit to data.
