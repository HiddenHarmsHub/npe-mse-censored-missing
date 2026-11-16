## Intercept-support acceptance rate A_m of each structure and the TV distance between π and π·A_m
include("mse_functions.jl")
include("model_selection_functions.jl")
using CSV, DataFrames, Random, Statistics

run = model_selection_run()
n_check = 1000        # conditional-NPE draws per structure
mass = 0.99           # structures covering this classifier mass
n_test = 500          # test datasets per system
output_path = model_selection_output_path(run)
real_data = Dict("A" => load_silverman_data(), "B" => load_king_counts())

acceptance(cnpe, y, mask) = mean(1 .<= raw_draws(cnpe, reshape(cnpe_input(y, mask), :, 1), n_check)[1, :] .<= 10)

## A_m over the leading structures, and the total variation distance it implies
function support_check(classifiers, cnpe, y, K)
    π = model_probabilities(classifiers, y)
    order = sortperm(π, rev = true)
    leading = order[1:searchsortedfirst(cumsum(π[order]), mass)]
    A = [acceptance(cnpe, y, model_mask(c, K)) for c in leading]
    p = π[leading] ./ sum(π[leading])
    q = p .* A
    q ./= sum(q)
    return (leading = leading, π = π[leading], A = A, tv = 0.5 * sum(abs.(p .- q)),
            weighted_A = sum(p .* A), min_A = minimum(A))
end

Random.seed!(2025)
structure_rows, dataset_rows = NamedTuple[], NamedTuple[]
for s in ["A", "B"]
    system = model_selection_systems[s]
    K = system.K
    config = model_selection_config(system, run)
    classifiers, cnpe = load_classifier(config, 0), load_cnpe(config, 0)

    y = encode_counts(real_data[s]; censored = system.censoring_upper > 0)
    r = support_check(classifiers, cnpe, y, K)
    append!(structure_rows, [(system = s, model = c, label = model_label(model_mask(c, K), K), prob = p, acceptance = a)
                             for (c, p, a) in zip(r.leading, r.π, r.A)])
    push!(dataset_rows, (system = s, data = "real", dataset = 0, n_structures = length(r.leading), tv = r.tv, weighted_acceptance = r.weighted_A, min_acceptance = r.min_A))
    println("System $s real data: $(length(r.leading)) structures, weighted A = $(round(r.weighted_A, digits = 4)), min A = $(round(r.min_A, digits = 4)), TV = $(round(r.tv, digits = 4))")

    test = BSON.load(model_selection_test_file(run, system))
    for i in sort(randperm(size(test[:y], 2))[1:n_test])
        r = support_check(classifiers, cnpe, test[:y][:, i], K)
        push!(dataset_rows, (system = s, data = "test", dataset = i, n_structures = length(r.leading), tv = r.tv, weighted_acceptance = r.weighted_A, min_acceptance = r.min_A))
    end
    sims = filter(r -> r.system == s && r.data == "test", DataFrame(dataset_rows))
    println("System $s test data: TV median $(round(median(sims.tv), digits = 4)), 95th percentile $(round(quantile(sims.tv, 0.95), digits = 4)), max $(round(maximum(sims.tv), digits = 4)); weighted A median $(round(median(sims.weighted_acceptance), digits = 4)), 5th percentile $(round(quantile(sims.weighted_acceptance, 0.05), digits = 4))")
end

CSV.write(joinpath(output_path, "support_acceptance.csv"), DataFrame(dataset_rows))
CSV.write(joinpath(output_path, "real_data", "support_acceptance_structures.csv"), DataFrame(structure_rows))
