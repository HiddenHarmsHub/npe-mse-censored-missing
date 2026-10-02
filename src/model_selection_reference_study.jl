## Reference posterior model probabilities for a stratified subset of the model-selection test sets:
## all 2^J structures enumerated, each marginal likelihood by importance sampling (Laplace-t proposal).
## The first n_check datasets also repeat the top structures with a second proposal.
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("mcmc_functions.jl")
@everywhere include("model_selection_functions.jl")
@everywhere include("model_selection_reference.jl")
@everywhere BLAS.set_num_threads(1)

gamma_sd = 4
n_reference = Dict("A" => 200, "B" => 300)
n_check = 20
n_check_models = 5
n_draws = 50_000
B = 4000

output_path = joinpath("output", "model_selection")
reference_path = joinpath(output_path, "reference")
mkpath(reference_path)

## Proportional strata: model size band × intercept band × censored-cell tercile (B only)
function reference_subset(test, system, n)
    J = binomial(system.K, 2)
    size_band = [div(4 * sum(m), J + 1) for m in eachcol(test.masks)]
    intercept_band = clamp.(floor.(Int, (test.pars[1, :] .- 1) ./ 1.8), 0, 4)
    n_censored = vec(sum(test.counts_obs .== -1, dims = 1))
    censored_band = system.censoring_upper > 0 ? searchsortedfirst.(Ref(quantile(n_censored, [1/3, 2/3])), n_censored) : zeros(Int, length(n_censored))
    return stratified_sample(collect(zip(size_band, intercept_band, censored_band)), n; seed = 3)
end

@everywhere function run_reference_dataset(system, i, counts_obs, truth, check, n_check_models, gamma_sd, n_draws, B, reference_path)
    prefix = joinpath(reference_path, "$(system.system)_dataset$(i)")
    isfile("$(prefix)_posteriors.csv") && return
    Random.seed!(i)
    K = system.K
    elapsed = @elapsed ref = reference_model_average(counts_obs, system; gamma_sd, n_draws, B)
    if check
        rm("$(prefix)_check.csv"; force = true)  # an interrupted earlier run may have left partial rows
        for c in sortperm(ref.π, rev = true)[1:n_check_models]
            r = log_evidence_check(counts_obs, model_mask(c, K); system.censoring_lower, system.censoring_upper, gamma_sd, n_draws)
            CSV.write("$(prefix)_check.csv", DataFrame([merge((dataset = i, system = system.system, model = c), r)]); append = isfile("$(prefix)_check.csv"))
        end
    end
    primary = primary_model_prior
    probs = DataFrame(
        dataset = i, system = system.system, model = 1:n_models(K),
        size = [model_size(model_mask(c, K)) for c in 1:n_models(K)],
        log_evidence = ref.log_evidence, log_evidence_se = ref.log_evidence_se, ess = ref.ess,
        prob = ref.π,
    )
    for (name, prior) in sensitivity_model_priors
        probs[!, "prob_$name"] = reweight_model_probs(ref.π, primary, prior, K)
    end
    CSV.write("$(prefix)_models.csv", probs)
    posteriors = DataFrame([
        merge((dataset = i, system = system.system, method = "reference_bma", quantity = String(q), seconds = elapsed),
              posterior_metrics(getproperty(ref, q), truth[q]))
        for q in keys(truth)])
    CSV.write("$(prefix)_posteriors.csv", posteriors)  # written last: marks the dataset as done
end

tasks = []
for s in ["A", "B"]
    system = model_selection_systems[s]
    test = (; (k => v for (k, v) in BSON.load(joinpath(output_path, "test_data_$(s).bson")))...)
    subset = reference_subset(test, system, n_reference[s])
    CSV.write(joinpath(output_path, "reference_datasets_$(s).csv"), DataFrame(dataset = subset))
    for (j, i) in enumerate(subset)
        truth = (alpha = Float64(test.pars[1, i]), lambda0 = exp(Float64(test.pars[1, i])), N0 = test.N0[i], N = test.N[i])
        isfile(joinpath(reference_path, "$(s)_dataset$(i)_posteriors.csv")) ||
            push!(tasks, (system, i, test.counts_obs[:, i], truth, j <= n_check, n_check_models, gamma_sd, n_draws, B, reference_path))
    end
end
println("Reference datasets to fit: ", length(tasks))

pmap(t -> run_reference_dataset(t...), tasks)

println("Finished the reference model-averaging study.")
