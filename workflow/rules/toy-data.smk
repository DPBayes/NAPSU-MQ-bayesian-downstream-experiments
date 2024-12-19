
epsilons = [0.1, 0.5, 1.0]
repeats = 100
#epsilons = [1.0]
#repeats = 1

rule real_data_results:
    output: 
        "results/toy-data/real-data-results/{repeat}.p"
    threads: 4
    resources:
        mem_mb=1000,
        runtime="10min",
    script:
        "../scripts/toy-data/real_data_results.py"

rule dp_glm:
    input:
        real_data="results/toy-data/real-data-results/{repeat}.p"
    threads: 4
    resources:
        mem_mb=1000,
        runtime="10min",
    output:
        "results/toy-data/dp-glm-posteriors/{repeat}_{epsilon}.p"
    script:
        "../scripts/toy-data/dp_glm_posterior.py"

rule napsu_mq_posterior:
    input:
        real_data="results/toy-data/real-data-results/{repeat}.p"
    threads: 4
    resources:
        mem_mb=1000,
        runtime="1h",
    output:
        "results/toy-data/napsu-mq-posteriors/{repeat}_{epsilon}.p"
    script:
        "../scripts/toy-data/napsu_mq_posterior.py"

rule downstream_posterior:
    input:
        napsu_mq_posterior="results/toy-data/napsu-mq-posteriors/{repeat}_{epsilon}.p",
        real_data_results="results/toy-data/real-data-results/{repeat}.p",
    threads: 4
    resources:
        mem_mb=8000,
        runtime="10h",
    output:
        "results/toy-data/downstream-posteriors/{repeat}_{epsilon}.p"
    script:
        "../scripts/toy-data/downstream_posterior.py"

rule plots_ready:
    input:
        downstream_posteriors=expand(
            "results/toy-data/downstream-posteriors/{repeat}_{epsilon}.p",
            repeat=range(repeats), epsilon=epsilons, 
        ),
        real_data_results=expand(
            "results/toy-data/real-data-results/{repeat}.p",
            repeat=range(repeats)
        ),
        # dp_glm_posteriors=expand(
        #     "results/toy-data/dp-glm-posteriors/{repeat}_{epsilon}.p",
        #     repeat=range(repeats), epsilon=epsilons,
        # )
    threads: 1
    resources:
        mem_mb=100,
        runtime="10min",
    output:
        "results/toy-data/ready-to-plot"
    shell:
        "touch {output}"
