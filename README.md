# qa-dna-batch-testing

Overview
--------
A python simulator test result distributions of microplate assays, based on Ref. [1].

Usage
-----
```
python simulate.py --help
```

Example:
```
python simulate.py --output '/path/to/results.csv'
                   --microplates 10000
                   --shape '(8, 12)'
                   --prevalence 0.16
                   --controls 6
                   --controls-position 'top-left'
```

References
----------
[1] E. N. Beylerian et al, _Statistical Modeling for Quality Assurance of Human Papillomavirus DNA Batch Testing_ (2018) (https://pubmed.ncbi.nlm.nih.gov/29570137/)
