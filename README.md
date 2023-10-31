
# Installing Dependencies

```bash
conda env create
conda activate bayes-downstream-env
```

# NAPSU-MQ Source
Most of the code in `lib/` is from our 
[NAPSU-MQ implementation](https://github.com/DPBayes/NAPSU-MQ-experiments)
([Räisä et al. 2023](https://proceedings.mlr.press/v206/raisa23a.html)).

# DP-GLM
The DP-GLM baseline uses code from 
[Kulkarni et al.](http://proceedings.mlr.press/v139/kulkarni21a.html) The 
code is not public, so we could not include it in the repo.

# Running the Code

The Gaussian example is in the notebook `bayesian-inference-gaussian-test.ipynb`.

For the toy data and UCI Adult experiments, we use 
[Snakemake](https://snakemake.readthedocs.io/en/stable/). The experiments can 
be run with 
```bash
snakemake -j 10
```
which runs 10 jobs in parallel. After the command finishes, the plotting 
notebooks `workflow/scripts/toy-data/report.py.ipynb` can
and `workflow/scripts/adult-reduced/report.py.ipynb` be run to 
plot the results.

Note: running this will likely take several days on a single computer.

Figures are saved in the `figures/` directory, and separated into subdirectories 
of each example.