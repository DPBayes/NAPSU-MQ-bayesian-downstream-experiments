
epsilons = [0.25, 0.5, 1.0]
repeats = 20
# repeats = 1

rule discretise_data:
    input:
        "datasets/adult.csv"
    output:
        "datasets/adult-reduced/adult-reduced-discretised.csv"
    threads: 1
    resources:
        mem_mb=1000,
        time="00:10:00",
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
        time="00:10:00",
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
        mem_mb=1000,
        time="23:00:00",
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
        mem_mb=8000,
        time="20:00:00",
    script:
        "../scripts/adult-reduced/downstream_posterior.py"

rule dpvi_hyperparameter_tuning:
    input:
        discretised_data="datasets/adult-reduced/adult-reduced-discretised.csv",
    output:
        "results/adult-reduced/dpvi-hyperparameters/{epsilon}.p"
    threads: 16
    resources:
        mem_mb=8000,
        time="23:59:59",
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
        mem_mb=2000,
        time="10:00:00"
    script:
        "../scripts/adult-reduced/dpvi_posterior.py"

rule plots_ready:
    input:
        downstream_posteriors=expand(
            "results/adult-reduced/downstream-posteriors/{repeat}_{epsilon}.p",
            repeat=range(repeats), epsilon=epsilons, 
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
        )
    output:
        "results/adult-reduced/ready-to-plot"
    threads: 1
    resources:
        mem_mb=100,
        time="00:10:00",
    shell:
        "touch {output}"
