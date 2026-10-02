## Run from the repo root with: julia --project test/runtests.jl
using Test, Random
include(joinpath(@__DIR__, "..", "src", "mse_functions.jl"))
include(joinpath(@__DIR__, "..", "src", "mcmc_functions.jl"))
include(joinpath(@__DIR__, "..", "src", "model_selection_functions.jl"))
include(joinpath(@__DIR__, "..", "src", "model_selection_reference.jl"))

@testset "design matrix" begin
    X = one_hot_encode_parameters(3)
    # cells 1, 2, 3, 12, 13, 23, 123; parameters α, β1, β2, β3, γ12, γ13, γ23
    @test X == [
        1 1 0 0 0 0 0;
        1 0 1 0 0 0 0;
        1 0 0 1 0 0 0;
        1 1 1 0 1 0 0;
        1 1 0 1 0 1 0;
        1 0 1 1 0 0 1;
        1 1 1 1 1 1 1
    ]

    Random.seed!(1)
    for K in 3:4
        pars = vcat(3.0, 0.5 .* randn(K), 0.3 .* randn(binomial(K, 2)))
        m = 5000
        Z = simulate_data(pars, m, log_transform = false)
        λ = exp.(one_hot_encode_parameters(K) * pars)
        @test all(abs.(vec(mean(Z, dims = 2)) .- λ) .< 5 .* sqrt.(λ ./ m))
    end
end

@testset "likelihood" begin
    X = one_hot_encode_parameters(3)
    pars = [2.0, 0.3, -0.5, 0.1, 0.4, -0.2, 0.05]
    λ = exp.(X * pars)
    y = [7, 3, 12, 0, 5, 2, 1]
    saturated(y) = sum(logfactorial.(y)) - sum(yi > 0 ? yi * log(yi) - yi : 0.0 for yi in y)
    @test likelihood_censored(y, pars, X, 0, 0) - saturated(y) ≈ sum(logpdf.(Poisson.(λ), y))

    for (a, b) in [(0, 10), (1, 4)]
        yc = copy(y)
        yc[[4, 6]] .= -1
        observed = setdiff(1:7, [4, 6])
        expected = sum(logpdf.(Poisson.(λ[observed]), y[observed])) +
            sum(log(cdf(Poisson(λ[i]), b) - cdf(Poisson(λ[i]), a - 1)) for i in [4, 6])
        @test likelihood_censored(yc, pars, X, a, b) - saturated(y[observed]) ≈ expected
    end

    for (y, δ) in [(91995510979395824, 3e-9), (1634, 0.01), (1, -0.5)]
        η = Float64(log(big(y)) + δ)
        exact = y * (big(η) - log(big(y))) - (exp(big(η)) - y)
        # error within a few ulps of η times the slope y δ; the naive y η - exp(η) form is off by ~10
        @test abs(poisson_kernel(y, η) - Float64(exact)) < 10 * abs(y * δ) * eps(η) + 1e-12
    end

    @test log_interval_prob(log(1e6), 0, 10) < -1e5
    @test isfinite(log_interval_prob(log(1e6), 0, 10))
end

@testset "Turing log density" begin
    X = one_hot_encode_parameters(3)
    y = [7, 3, -1, 0, 5, -1, 1]
    pars = [2.0, 0.3, -0.5, 0.1, 0.4, -0.2, 0.05]
    m = mse_model_censored(y, X, Uniform(1, 10), Normal(0, 4), Normal(0, 4), 0, 10)
    lj = logjoint(m, (intercept = pars[1], betas = pars[2:4], gammas = pars[5:7]))
    prior = logpdf(Uniform(1, 10), pars[1]) + sum(logpdf.(Normal(0, 4), pars[2:end]))
    @test lj ≈ prior + likelihood_censored(y, pars, X, 0, 10)
    @test lj ≈ log_posterior(pars, y, X, 0, 10)
end

@testset "Laplace approximation" begin
    X = one_hot_encode_parameters(3)
    y = [124829, 5, -1, 3, 1634, -1, -1]
    mode, L = laplace_approximation(y, X; censoring_lower = 0, censoring_upper = 10)
    f(θ) = log_posterior(θ, y, X, 0, 10)
    H = -ForwardDiff.hessian(f, mode)
    @test maximum(abs.(ForwardDiff.gradient(f, mode) .* sqrt.(diag(L * L')))) < 1e-3
    @test L * L' ≈ inv(H) rtol = 1e-6
end

@testset "test counts" begin
    dir = mktempdir()
    Z_counts = [0 5; 10 11; 123456789 3]
    Z_test = Float32.(log.(Z_counts .+ 1))
    params = zeros(Float32, 7, 2)
    BSON.@save joinpath(dir, "test_data_2.bson") Z_test Z_counts params
    counts, _ = load_test_counts(dir, 2, 0, 10)
    @test counts == [-1 -1; -1 11; 123456789 -1]
    counts, _ = load_test_counts(dir, 2)
    @test counts == Z_counts

    @test !is_capped([5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    @test is_capped([10.0, 12.0, 12.0, 12.0, 0.0, 0.0, 0.0])
end

@testset "benchmark selection" begin
    Random.seed!(2)
    pars = vcat(rand(Uniform(1, 10), 1, 2000), randn(6, 2000))
    counts = rand([-1, 1], 7, 2000)
    idx = select_benchmark_datasets(pars, counts; n = 200)
    @test length(idx) == 200
    @test allunique(idx)
end

@testset "IRLS sampler pieces" begin
    rng = MersenneTwister(3)
    η = log.([0.5, 3.0, 50.0])
    draws = [impute_censored([-1, -1, -1], η, 1, 4; rng = rng) for _ in 1:20000]
    @test all(all(1 .<= d .<= 4) for d in draws)
    p = pdf.(Poisson(3.0), 1:4) ./ (cdf(Poisson(3.0), 4) - cdf(Poisson(3.0), 0))
    freq = [mean(d[2] == k for d in draws) for k in 1:4]
    @test all(abs.(freq .- p) .< 0.015)

    X = one_hot_encode_parameters(3)
    θ = [2.0, 0.3, -0.5, 0.1, 0.4, -0.2, 0.05]
    y = [7, 3, 12, 0, 5, 2, 1]
    m, R = irls_proposal(θ, y, X, vcat(5.5, zeros(6)), 1 ./ vcat(81 / 12, fill(16.0, 6)))
    q = MvNormal(m, Symmetric(inv(R' * R)))
    θ2 = θ .+ 0.1
    @test proposal_logpdf(θ, m, R) - proposal_logpdf(θ2, m, R) ≈ logpdf(q, θ) - logpdf(q, θ2)
end

@testset "model structures and priors" begin
    for K in (3, 4, 5)
        J = binomial(K, 2)
        masks = enumerate_models(K)
        @test size(masks) == (J, 2^J)
        @test allunique(eachcol(masks))
        @test all(model_index(model_mask(c, K)) == c for c in 1:2^J)
        @test model_mask(1, K) == falses(J) && model_mask(2^J, K) == trues(J)
        for prior in (primary_model_prior, values(sensitivity_model_priors)...)
            lp = model_prior_logpmf(prior, K)
            @test sum(exp.(lp)) ≈ 1
        end
        sizes = vec(sum(masks, dims = 1))
        p = exp.(model_prior_logpmf(BetaBinomialModelPrior(1, 1), K))
        @test all(sum(p[sizes .== s]) ≈ 1 / (J + 1) for s in 0:J)
    end
    ## bit j is the j-th pair of enumerate_two_digit_numbers
    @test model_label(model_mask(1 + 2^0 + 2^5, 4), 4) == "12,34"

    Random.seed!(4)
    π = rand(64); π ./= sum(π)
    @test reweight_model_probs(π, primary_model_prior, primary_model_prior, 4) ≈ π
    uniform = reweight_model_probs(π, primary_model_prior, BernoulliModelPrior(0.5), 4)
    expected = π ./ exp.(model_prior_logpmf(primary_model_prior, 4))
    @test uniform ≈ expected ./ sum(expected)

    s = model_summaries(π, 4)
    @test sum(s.size_probs) ≈ 1
    @test s.inclusion ≈ enumerate_models(4) * π
    for L in (0.5, 0.9)
        set = credible_model_set(π, L)
        @test sum(π[set]) >= L && sum(π[set[1:end-1]]) < L
    end
end

@testset "masked simulation and encoding" begin
    K = 4
    X = one_hot_encode_parameters(K)
    θ = Float32[3.0, 0.5, -0.2, 0.1, 0.3, 1.0, -1.0, 0.5, 0.2, -0.4, 0.8]
    mask = Bool[1, 0, 1, 0, 0, 1]
    θ2 = copy(θ); θ2[1 + K .+ findall(.!mask)] .+= 5
    @test X * effective_parameters(θ, mask) ≈ X * effective_parameters(θ2, mask)

    for (cl, cu) in [(0, 0), (1, 4), (0, 10)]
        Random.seed!(5)
        y_sim = simulate_data(effective_parameters(θ, mask), 1; censoring_lower = cl, censoring_upper = cu)
        Random.seed!(5)
        Z = Int.(simulate_data(effective_parameters(θ, mask), 1; log_transform = false))
        @test encode_counts(censor_counts(Z, cl, cu); censored = cu > 0) == y_sim
    end
    king_counts = load_king_counts()
    @test king_counts == [74, 17, 584, 35, -1, 63, -1, 12, -1, 35, 9, -1, 17, -1, -1]
    @test encode_counts(king_counts; censored = true) == load_king_data()
    @test encode_counts(load_silverman_data(); censored = false) == Float32.(log.(load_silverman_data() .+ 1))

    sim = simulate_model_selection_data(50, model_selection_systems["B"])
    @test all(sim.counts_obs[sim.counts_obs .>= 0] .== sim.counts[sim.counts_obs .>= 0])
    @test all(1 .<= sim.counts[sim.counts_obs .== -1] .<= 4)
    @test sim.N == vec(sum(sim.counts, dims = 1)) .+ sim.N0
end

struct StubNPE end
boundedsampleposterior(::StubNPE, Z, N::Integer, lower, upper; kw...) = vcat(fill(5f0, 1, N), randn(Float32, 10, N))

@testset "population draws and model averaging" begin
    rng = MersenneTwister(6)
    system = model_selection_systems["B"]
    θ_eff = repeat(Float32[2.0, 0.5, -0.2, 0.1, 0.3, 0, 0, 0, 0, 0, 0], 1, 2000)
    counts_obs = [74, 17, -1, 584, 63, 12, 9, 35, -1, -1, -1, 35, 17, -1, -1]
    pop = population_draws(θ_eff, counts_obs; censoring_lower = 1, censoring_upper = 4, rng)
    exact = sum(counts_obs[counts_obs .>= 0])
    @test all(exact + 6 .<= pop.N_obs .<= exact + 24)
    @test pop.N == pop.N_obs .+ pop.N0
    @test abs(mean(pop.N0) - exp(2.0)) < 5 * sqrt(exp(2.0) / 2000)

    ## A stub conditional NPE whose posterior ignores the data; a one-hot π must reproduce it
    y = encode_counts(counts_obs; censored = true)
    π = zeros(64); π[7] = 1
    bma = sample_model_averaged_posterior(StubNPE(), y, π, 500, system; counts_obs, rng)
    @test all(bma.models .== 7)
    @test size(bma.θ, 2) == 500
    @test all(bma.θ_eff[1 + 4 .+ findall(.!model_mask(7, 4)), :] .== 0)

    π = fill(1 / 64, 64)
    bma = sample_model_averaged_posterior(StubNPE(), y, π, 6400, system; counts_obs, rng)
    @test length(unique(bma.models)) > 50

    values = randn(rng, 3000) .+ repeat([0.0, 2.0, 5.0], 1000)
    groups = repeat([1, 2, 3], 1000)
    d = variance_decomposition(values, groups)
    @test d.total ≈ var(values; corrected = false)
end

@testset "metrics" begin
    rng = MersenneTwister(7)
    ranks = [randomised_rank(rand(rng, Poisson(3.0), 200), rand(rng, Poisson(3.0)); rng) for _ in 1:4000]
    @test abs(mean(ranks) - 0.5) < 0.02
    @test abs(mean(ranks .< 0.1) - 0.1) < 0.02
    @test interval_score(1.0, 2.0, 1.5, 0.95) == 1.0
    @test interval_score(1.0, 2.0, 3.0, 0.9) ≈ 1.0 + 20
    m = posterior_metrics(randn(rng, 10_000), 0.0)
    @test m.inside_95 && abs(m.lower_95 + 1.96) < 0.1
    @test length(stratified_sample(repeat(1:4, 50), 20)) == 20
end

@testset "reference evidence" begin
    Random.seed!(8)
    K = 3
    X = one_hot_encode_parameters(K)
    y = [1200, 340, 560, 90, 150, 60, 20]
    ## The K keyword reproduces the inferred-K path for the full model
    pars = [2.0, 0.3, -0.5, 0.1, 0.4, -0.2, 0.05]
    @test log_posterior(pars, y, X, 0, 0; K = 3) == log_posterior(pars, y, X, 0, 0)
    ## Null model: evidence matches the Laplace approximation with large counts
    mask = falses(3)
    Xm = X[:, 1:4]
    mode, L = laplace_approximation(y, Xm; K = 3)
    laplace_ev = log_posterior(mode, y, Xm, 0, 0; K = 3) + 0.5 * 4 * log(2π) + logdet(L)
    ev = log_evidence(y, mask; n_draws = 20_000)
    @test abs(ev.log_evidence - laplace_ev) < 0.05
    @test all(ev.θ[5:7, :] .== 0)
    check = log_evidence_check(y, Bool[1, 0, 1]; n_draws = 20_000)
    @test abs(check.laplace - check.pilot) < 0.05

    ## A badly fitting structure for large counts pushes the mode to the intercept's lower bound,
    ## where the logit-scale search used to produce NaN derivatives (test set A, dataset 8060)
    y5 = [6598, 86541, 398, 22178, 1906391, 142, 160, 826365, 302974, 17493, 883781, 14776413, 0, 2412272, 250375, 16,
          89925, 6417, 0, 300897, 2519155, 7, 107888568, 9917693, 5, 0, 37014, 275683, 44, 1381, 26]
    X5 = one_hot_encode_parameters(5)[:, vcat(trues(6), model_mask(41, 5))]
    mode5, L5 = laplace_approximation(y5, X5; K = 5)
    @test all(isfinite, mode5) && all(isfinite, L5)
    @test isfinite(log_evidence(y5, model_mask(41, 5); n_draws = 2000).log_evidence)
end
