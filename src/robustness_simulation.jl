## Robustness study: evaluate every experiment in chunks (Slurm).
using Pkg; Pkg.activate(".")
using Distributed, SlurmClusterManager
addprocs(SlurmManager(); exeflags=["--threads", "1", "--project"])

@everywhere include("mse_functions.jl")
@everywhere include("mcmc_functions.jl")
@everywhere include("model_selection_functions.jl")
@everywhere include("model_selection_reference.jl")
@everywhere include("robustness_functions.jl")

chunk_size = 50
n_draws = 4000
results_path = joinpath(robustness_path, "simulation")
mkpath(results_path)

scenarios = CSV.read(joinpath(robustness_path, "scenarios.csv"), DataFrame)
scenario_names = Dict(s => scenarios.scenario[scenarios.system .== s] for s in ["A", "B"])
experiments = robustness_experiments(scenario_names)
for e in experiments
    isfile(robustness_test_file(e.system, e.scenario)) || error("$(robustness_test_file(e.system, e.scenario)) not found; run src/robustness_test_data.jl first.")
end

@everywhere chunk_file(results_path, e, chunk, kind) = joinpath(results_path, "$(experiment_id(e))_chunk$(chunk)_$(kind).csv")

@everywhere function run_robustness_chunk(e, chunk, chunk_size, n_draws, results_path)
    test = (; (k => v for (k, v) in BSON.load(robustness_test_file(e.system, e.scenario)))...)
    idx = ((chunk - 1) * chunk_size + 1):min(chunk * chunk_size, size(test.counts, 2))
    config = robustness_config(e.system; censored = e.censored)
    K = config.K
    priors = coefficient_priors(beta_sd = e.sigma, gamma_sd = e.sigma)
    fit = (K = K, censoring_lower = e.censored ? test.censoring_lower : 0, censoring_upper = e.censored ? test.censoring_upper : 0)
    X = one_hot_encode_parameters(K)
    nbe = BSON.load(robustness_nbe_file(config, e.sigma))[:estimator]
    npe = BSON.load(robustness_npe_file(config, e.sigma))[:estimator]
    if e.model_average
        ms_config = model_selection_config(model_selection_systems[e.system], model_selection_runs["b4_g4x"])
        classifiers, cnpe = load_classifier(ms_config, 0), load_cnpe(ms_config, 0)
    end
    diagnostics = e.experiment == "main"
    if diagnostics
        reference = BSON.load(robustness_reference_file(e.system))
        embedding = embedding_reference(npe, reference[:y])
    end

    Random.seed!(stable_seed(experiment_id(e)) + chunk)
    dataset_rows, posterior_rows = NamedTuple[], NamedTuple[]
    for i in idx
        counts_obs = e.censored ? test.counts_obs[:, i] : test.counts[:, i]
        y = encode_counts(counts_obs; censored = config.censoring_upper > 0)
        α = Float64(test.pars[1, i])
        truths = (alpha = α, lambda0 = exp(α), N0 = Float64(test.N0[i]), N = test.N[i])
        population(θ) = population_draws(θ, counts_obs; fit.censoring_lower, fit.censoring_upper, X)

        draws = Dict{String, Any}()
        timing = Dict(m => NaN for m in ["nbe", "npe", "npe_nis", "exact", "bma"])
        timing["nbe"] = @elapsed q = nbe_quantiles(nbe, y)
        timing["npe"] = @elapsed draws["npe"] = population(bounded_draws(npe, reshape(y, :, 1), n_draws))
        timing["npe_nis"] = @elapsed nis = npe_importance_sampling(npe, y, counts_obs, fit, priors, n_draws; X)
        draws["npe_nis"] = population(nis.θ)
        timing["exact"] = @elapsed ex = exact_posterior(counts_obs, fit, priors, n_draws)
        draws["exact"] = population(ex.θ)
        bma_top = NaN
        if e.model_average
            timing["bma"] = @elapsed begin
                π = model_probabilities(classifiers, y)
                draws["bma"] = sample_model_averaged_posterior(cnpe, y, π, n_draws, fit; counts_obs)
            end
            bma_top = maximum(π)
        end

        push!(dataset_rows, (experiment = e.experiment, system = e.system, scenario = e.scenario, sigma = e.sigma, dataset = i,
            intercept = α, capped = test.capped[i], tries = test.tries[i], n_zero = count(==(0), counts_obs),
            n_censored = count(==(-1), counts_obs), empty_overlap = has_empty_overlap(counts_obs, K),
            N0_true = truths.N0, N_true = truths.N, exact_ess = ex.ess, exact_retried = ex.retried, nis_ess = nis.ess,
            npe_draws = length(draws["npe"].alpha), bma_top_probability = bma_top,
            summary_tail = diagnostics ? summary_tail(counts_obs, K, reference[:summaries]) : NaN,
            embedding_percentile = diagnostics ? embedding_percentile(embedding, y) : NaN,
            time_nbe = timing["nbe"], time_npe = timing["npe"], time_npe_nis = timing["npe_nis"], time_exact = timing["exact"], time_bma = timing["bma"]))

        meta = (experiment = e.experiment, system = e.system, scenario = e.scenario, sigma = e.sigma, dataset = i)
        push!(posterior_rows, merge(meta, (method = "nbe", quantity = "alpha"), quantile_metrics(q[1, :], α)))
        push!(posterior_rows, merge(meta, (method = "nbe", quantity = "lambda0"), quantile_metrics(exp.(q[1, :]), exp(α))))
        for (method, d) in sort(collect(draws), by = first), quantity in keys(truths)
            push!(posterior_rows, merge(meta, (method = method, quantity = String(quantity)), posterior_metrics(getproperty(d, quantity), truths[quantity])))
        end
    end
    CSV.write(chunk_file(results_path, e, chunk, "posteriors"), DataFrame(posterior_rows))
    CSV.write(chunk_file(results_path, e, chunk, "datasets"), DataFrame(dataset_rows))
end

tasks = [(e, chunk, chunk_size, n_draws, results_path) for e in experiments for chunk in 1:cld(robustness_test_size, chunk_size)
         if !isfile(chunk_file(results_path, e, chunk, "datasets"))]
println("Experiments: ", length(experiments), "; chunks to evaluate: ", length(tasks))

pmap(tasks) do task
    e, chunk = task[1], task[2]
    println("Worker $(myid()) evaluating $(experiment_id(e)), chunk $chunk")
    try
        run_robustness_chunk(task...)
    catch err
        msg = sprint(showerror, err, catch_backtrace())
        println("Chunk failed for $(experiment_id(e)), chunk $chunk:\n$msg")
        write(chunk_file(task[end], e, chunk, "error") * ".txt", msg)
    end
end

println("Finished the robustness simulation study.")
