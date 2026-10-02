## Run from the repo root with: julia --project test/runtests.jl
using Test, Random
include(joinpath(@__DIR__, "..", "src", "mse_functions.jl"))
include(joinpath(@__DIR__, "..", "src", "mcmc_functions.jl"))

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
