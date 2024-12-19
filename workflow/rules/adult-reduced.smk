
epsilons = [0.25, 0.5, 1.0]
repeats = 20
n_syn_datasets = 100
hierarchical_n_syn_datasets = 200
# repeats = 2
# n_syn_datasets = 2

rule discretise_data:
    input:
        "datasets/adult.csv"
    output:
        "datasets/adult-reduced/adult-reduced-discretised.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="1h",
    script:
        "../scripts/adult-reduced/adult-reduced-discretise-data.py"

rule real_data_results:
    input:
        queries="datasets/adult-reduced/queries.p",
        discretised_data="datasets/adult-reduced/adult-reduced-discretised.csv",
    output: 
        "results/adult-reduced/real-data-results/results.p"
    threads: 4
    resources:
        mem_mb=1000,
        runtime="1h"
    script:
        "../scripts/adult-reduced/real_data_results.py"

rule napsu_mq_posterior:
    input:
        data="datasets/adult-reduced/adult-reduced-discretised.csv",
        queries="datasets/adult-reduced/queries.p",
    output:
        "results/adult-reduced/napsu-mq-posteriors/{repeat}_{epsilon}.p"
    threads: 4
    resources:
        mem_mb=4000,
        runtime="30h",
        slurm_partition="medium",
    script:
        "../scripts/adult-reduced/napsu_mq_posterior.py"

rule downstream_posterior:
    input:
        napsu_mq_posterior="results/adult-reduced/napsu-mq-posteriors/{repeat}_{epsilon}.p",
        real_data_results="results/adult-reduced/real-data-results/results.p",
        queries="datasets/adult-reduced/queries.p",
        data="datasets/adult-reduced/adult-reduced-discretised.csv",
    output:
        "results/adult-reduced/downstream-posteriors/{repeat}_{epsilon}.p"
    threads: 4
    resources:
        mem_mb=16000,
        runtime="24h",
        slurm_partition="medium",
    script:
        "../scripts/adult-reduced/downstream_posterior.py"

rule downstream_posterior_variance_correction:
    input:
        napsu_mq_posterior="results/adult-reduced/napsu-mq-posteriors/{repeat}_{epsilon}.p",
        real_data_results="results/adult-reduced/real-data-results/results.p",
        queries="datasets/adult-reduced/queries.p",
        data="datasets/adult-reduced/adult-reduced-discretised.csv",
    output:
        "results/adult-reduced/downstream-posteriors-variance-correction/{repeat}_{epsilon}.p"
    threads: 4
    resources:
        mem_mb=16000,
        runtime="8h",
        slurm_partition="short",
    script:
        "../scripts/adult-reduced/downstream_posterior_variance_correction.py"

rule hierarchical_napsu_mq_posterior:
    input:
        data="datasets/adult-reduced/adult-reduced-discretised.csv",
        queries="datasets/adult-reduced/queries_with_age_gender_income.p",
    output:
        "results/adult-reduced/hierarchical-napsu-mq-posteriors/{repeat}_{epsilon}.p"
    threads: 4
    resources:
        mem_mb=6000,
        runtime="8h",
        slurm_partition="short",
    script:
        "../scripts/adult-reduced/napsu_mq_posterior.py"

rule hierarchical_downstream_posterior:
    input:
        napsu_mq_posterior="results/adult-reduced/hierarchical-napsu-mq-posteriors/{repeat}_{epsilon}.p",
        real_data_results="results/adult-reduced/real-data-results/results.p",
        queries="datasets/adult-reduced/queries_with_age_gender_income.p",
        data="datasets/adult-reduced/adult-reduced-discretised.csv",
    output:
        "results/adult-reduced/hierarchical-downstream-posteriors/{repeat}_{syn_data_ind}_{epsilon}.p"
    threads: 8
    resources:
        mem_mb=8000,
        runtime="8h",
        slurm_partition="short",
    script:
        "../scripts/adult-reduced/hierarchical_downstream_posterior.py"

rule dpvi_hyperparameter_tuning:
    input:
        discretised_data="datasets/adult-reduced/adult-reduced-discretised.csv",
    output:
        "results/adult-reduced/dpvi-hyperparameters/{epsilon}.p"
    threads: 16
    resources:
        mem_mb=8000,
        runtime="40h",
        slurm_partition="medium",
    script:
        "../scripts/adult-reduced/dpvi_hyperparameters.py"

rule dpvi_posterior:
    input:
        discretised_data="datasets/adult-reduced/adult-reduced-discretised.csv",
        hyperparameters="results/adult-reduced/dpvi-hyperparameters/{epsilon}.p",
    output:
        "results/adult-reduced/dpvi-posteriors/{repeat}_{epsilon}.p"
    threads: 8
    resources:
        mem_mb=4000,
        runtime="15h",
        slurm_partition="medium",
    script:
        "../scripts/adult-reduced/dpvi_posterior.py"

rule synthpop_syn_data:
    input:
        discretised_data="datasets/adult-reduced/adult-reduced-discretised.csv",
    output:
        ["results/adult-reduced/synthpop-syn-data/{{repeat}}_{}.csv".format(syn_data_ind) 
        for syn_data_ind in range(n_syn_datasets)
        ]
    threads: 4
    resources:
        mem_mb=16000,
        runtime="16h",
        slurm_partition="medium",
    shell:
        "Rscript workflow/scripts/adult-reduced/synthpop_syn_data.R {wildcards.repeat}"

rule synthpop_posterior:
    input:
        synthpop_syn_data=[
            "results/adult-reduced/synthpop-syn-data/{{repeat}}_{}.csv".format(syn_data_ind)
            for syn_data_ind in range(n_syn_datasets)
        ],
        real_data_results="results/adult-reduced/real-data-results/results.p",
    output:
        "results/adult-reduced/synthpop-posteriors/{repeat}.p"
    threads: 4
    resources:
        mem_mb=8000,
        runtime="1h",
    script:
        "../scripts/adult-reduced/synthpop_posterior.py"

rule plots_ready:
    input:
        downstream_posteriors=expand(
            "results/adult-reduced/downstream-posteriors/{repeat}_{epsilon}.p",
            repeat=range(repeats), epsilon=epsilons, 
        ),
        downstream_posteriors_variance_correction=expand(
            "results/adult-reduced/downstream-posteriors-variance-correction/{repeat}_{epsilon}.p",
            repeat=range(repeats), epsilon=epsilons, 
        ),
        hierarchical_downstream_posteriors=expand(
            "results/adult-reduced/hierarchical-downstream-posteriors/{repeat}_{syn_data_ind}_{epsilon}.p",
            repeat=range(5), epsilon=[1.0], syn_data_ind=range(hierarchical_n_syn_datasets)
        ),
        real_data_results=expand(
            "results/adult-reduced/real-data-results/results.p",
            repeat=range(repeats)
        ),
        dpvi_posteriors=expand(
            "results/adult-reduced/dpvi-posteriors/{repeat}_{epsilon}.p",
            repeat=range(repeats), epsilon=epsilons,
        ),
        dpvi_hypers=expand(
            "results/adult-reduced/dpvi-hyperparameters/{epsilon}.p",
            epsilon=epsilons,
        ),
        synthpop_posteriors=expand(
            "results/adult-reduced/synthpop-posteriors/{repeat}.p",
            repeat=range(repeats)
        ),
    output:
        "results/adult-reduced/ready-to-plot"
    threads: 1
    resources:
        mem_mb=100,
        runtime="1h",
    shell:
        "touch {output}"
