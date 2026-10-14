using Flux: softplus, sigmoid

## Output layer of the NBE: the 0.025, 0.5 and 0.975 quantiles of n parameters, which cannot cross
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

## MonotoneQuantiles for q quantile levels (a separate type so saved models still load)
struct MonotoneQuantilesQ
    n::Int
    q::Int
    a::Float32
    b::Float32
    bounded::Bool
end

MonotoneQuantilesQ(n::Integer, q::Integer, support) =
    isnothing(support) ? MonotoneQuantilesQ(n, q, 0f0, 0f0, false) : MonotoneQuantilesQ(n, q, Float32(support[1]), Float32(support[2]), true)

function (layer::MonotoneQuantilesQ)(x::AbstractMatrix)
    n, q = layer.n, layer.q
    z = reshape(x, n, q, :)
    c = cumsum(cat(z[:, 1:1, :], softplus.(z[:, 2:q, :]); dims = 2); dims = 2)
    if layer.bounded
        c = cat(layer.a .+ (layer.b - layer.a) .* sigmoid.(c[1:1, :, :]), c[2:end, :, :]; dims = 1)
    end
    return reshape(c, n * q, :)
end

(layer::MonotoneQuantilesQ)(x::AbstractVector) = vec(layer(reshape(x, :, 1)))
