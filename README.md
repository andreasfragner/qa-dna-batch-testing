# qa-dna-batch-testing

Overview
--------
A simulator for microplate assay test result distributions, based on Ref. [1].

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
