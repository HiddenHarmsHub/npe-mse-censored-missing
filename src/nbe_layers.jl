using Flux: softplus, sigmoid

## Output layer of the NBE: turns 3n raw network outputs into the 0.025, 0.5 and 0.975 posterior quantiles of
## n parameters, stacked [lower; median; upper]. Each quantile adds a positive increment to the one below, so
## they cannot cross. With an intercept support [a, b] the intercept's quantiles are built in the same way on
## the logit scale and then mapped into [a, b].
struct MonotoneQuantiles
    n::Int
    a::Float32
    b::Float32
    bounded::Bool
end

MonotoneQuantiles(n::Integer, support) =
    isnothing(support) ? MonotoneQuantiles(n, 0f0, 0f0, false) : MonotoneQuantiles(n, Float32(support[1]), Float32(support[2]), true)

function (layer::MonotoneQuantiles)(x::AbstractMatrix)
    n = layer.n
    c₁ = x[1:n, :]
    c₂ = c₁ .+ softplus.(x[(n + 1):2n, :])
    c₃ = c₂ .+ softplus.(x[(2n + 1):3n, :])
    bound(c) = layer.bounded ? vcat(layer.a .+ (layer.b - layer.a) .* sigmoid.(c[1:1, :]), c[2:end, :]) : c
    return vcat(bound(c₁), bound(c₂), bound(c₃))
end

(layer::MonotoneQuantiles)(x::AbstractVector) = vec(layer(reshape(x, :, 1)))
