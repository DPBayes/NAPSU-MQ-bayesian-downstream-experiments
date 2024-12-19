
library(synthpop)


size_mul <- 10
# n_syn_datasets <- 2
n_syn_datasets <- 100

args <- commandArgs(trailingOnly = TRUE)
repeat_ind <- as.integer(args[1])

set.seed(984592 + repeat_ind)

data_filename <- "datasets/adult-reduced/adult-reduced-discretised.csv"
data <- read.obs(data_filename)
# Bootstrap
ind <- sample(nrow(data), nrow(data), replace = TRUE)
data <- data[ind, ]
syn_data_proper <- syn(
    # data, proper = TRUE, m = n_syn_datasets, k = nrow(data) * size_mul
    data, method="parametric", proper = TRUE, m = n_syn_datasets, k = nrow(data) * size_mul
)

for (syn_data_ind in 1:n_syn_datasets){
    output_filename <- sprintf(
        "results/adult-reduced/synthpop-syn-data/%d_%d.csv",
        repeat_ind, 
        syn_data_ind - 1
    )
    write.csv(
        syn_data_proper$syn[[syn_data_ind]], output_filename,
        row.names = FALSE
    )
}