# ============================================================
# Abstract interface
# ============================================================

abstract type AbstractNeuralNetwork end
abstract type AbstractBufferedNeuralNetwork <: AbstractNeuralNetwork end

inputdim(net::AbstractNeuralNetwork) =
    error("inputdim not implemented for $(typeof(net))")

outputdim(net::AbstractNeuralNetwork) =
    error("outputdim not implemented for $(typeof(net))")

numweights(net::AbstractNeuralNetwork) =
    error("numweights not implemented for $(typeof(net))")

basenet(net::AbstractBufferedNeuralNetwork) =
    error("basenet not implemented for $(typeof(net))")

workspace(net::AbstractBufferedNeuralNetwork) =
    error("workspace not implemented for $(typeof(net))")

# Generic forwarding for buffered networks
inputdim(net::AbstractBufferedNeuralNetwork) = inputdim(basenet(net))
outputdim(net::AbstractBufferedNeuralNetwork) = outputdim(basenet(net))
numweights(net::AbstractBufferedNeuralNetwork) = numweights(basenet(net))


# ============================================================
# Two-layer network
# ============================================================

struct TwoLayerNetwork{F} <: AbstractNeuralNetwork
    Din::Int
    H::Int
    Dout::Int
    f::F
end

TwoLayerNetwork(; in::Integer, H::Integer, out::Integer, f=tanh) =
    TwoLayerNetwork(Int(in), Int(H), Int(out), f)

inputdim(net::TwoLayerNetwork) = net.Din
outputdim(net::TwoLayerNetwork) = net.Dout

numweights(net::TwoLayerNetwork) =
    net.H * net.Din + net.H +
    net.Dout * net.H + net.Dout


# ============================================================
# Three-layer network
# ============================================================

struct ThreeLayerNetwork{F} <: AbstractNeuralNetwork
    Din::Int
    H1::Int
    H2::Int
    Dout::Int
    f::F
end

ThreeLayerNetwork(; in::Integer, H1::Integer, H2::Integer, out::Integer, f=tanh) =
    ThreeLayerNetwork(Int(in), Int(H1), Int(H2), Int(out), f)

inputdim(net::ThreeLayerNetwork) = net.Din
outputdim(net::ThreeLayerNetwork) = net.Dout

numweights(net::ThreeLayerNetwork) =
    net.H1 * net.Din + net.H1 +
    net.H2 * net.H1 + net.H2 +
    net.Dout * net.H2 + net.Dout


# ============================================================
# Workspaces
# ============================================================

struct TwoLayerWorkspace{T}
    Hbuf::Matrix{T}   # H × N
    Ybuf::Matrix{T}   # Dout × N
end

function TwoLayerWorkspace(net::TwoLayerNetwork, N::Integer, ::Type{T}=Float64) where {T}
    Nint = Int(N)
    TwoLayerWorkspace(
        Matrix{T}(undef, net.H, Nint),
        Matrix{T}(undef, net.Dout, Nint),
    )
end

struct ThreeLayerWorkspace{T}
    H1buf::Matrix{T}  # H1 × N
    H2buf::Matrix{T}  # H2 × N
    Ybuf::Matrix{T}   # Dout × N
end

function ThreeLayerWorkspace(net::ThreeLayerNetwork, N::Integer, ::Type{T}=Float64) where {T}
    Nint = Int(N)
    ThreeLayerWorkspace(
        Matrix{T}(undef, net.H1, Nint),
        Matrix{T}(undef, net.H2, Nint),
        Matrix{T}(undef, net.Dout, Nint),
    )
end


# ============================================================
# Buffered networks
# ============================================================

struct BufferedTwoLayerNetwork{T,F} <: AbstractBufferedNeuralNetwork
    net::TwoLayerNetwork{F}
    ws::TwoLayerWorkspace{T}
end

function BufferedTwoLayerNetwork(net::TwoLayerNetwork, N::Integer, ::Type{T}=Float64) where {T}
    BufferedTwoLayerNetwork{T, typeof(net.f)}(net, TwoLayerWorkspace(net, N, T))
end

function BufferedTwoLayerNetwork(;
    in::Integer,
    H::Integer,
    out::Integer,
    N::Integer,
    f=tanh,
    T::Type=Float64,
)
    net = TwoLayerNetwork(in=in, H=H, out=out, f=f)
    BufferedTwoLayerNetwork(net, N, T)
end

basenet(net::BufferedTwoLayerNetwork) = net.net
workspace(net::BufferedTwoLayerNetwork) = net.ws


struct BufferedThreeLayerNetwork{T,F} <: AbstractBufferedNeuralNetwork
    net::ThreeLayerNetwork{F}
    ws::ThreeLayerWorkspace{T}
end

function BufferedThreeLayerNetwork(net::ThreeLayerNetwork, N::Integer, ::Type{T}=Float64) where {T}
    BufferedThreeLayerNetwork{T, typeof(net.f)}(net, ThreeLayerWorkspace(net, N, T))
end

function BufferedThreeLayerNetwork(;
    in::Integer,
    H1::Integer,
    H2::Integer,
    out::Integer,
    N::Integer,
    f=tanh,
    T::Type=Float64,
)
    net = ThreeLayerNetwork(in=in, H1=H1, H2=H2, out=out, f=f)
    BufferedThreeLayerNetwork(net, N, T)
end

basenet(net::BufferedThreeLayerNetwork) = net.net
workspace(net::BufferedThreeLayerNetwork) = net.ws


# ============================================================
# Unpack helpers
# ============================================================

function unpackweights(net::TwoLayerNetwork, weights::AbstractVector)
    length(weights) == numweights(net) ||
        throw(DimensionMismatch("expected $(numweights(net)) weights, got $(length(weights))"))

    mark = 0

    W1 = @view reshape(weights[mark+1 : mark + net.H * net.Din], net.H, net.Din)
    mark += net.H * net.Din

    b1 = @view weights[mark+1 : mark + net.H]
    mark += net.H

    W2 = @view reshape(weights[mark+1 : mark + net.Dout * net.H], net.Dout, net.H)
    mark += net.Dout * net.H

    b2 = @view weights[mark+1 : mark + net.Dout]

    return W1, b1, W2, b2
end

function unpackweights(net::ThreeLayerNetwork, weights::AbstractVector)
    length(weights) == numweights(net) ||
        throw(DimensionMismatch("expected $(numweights(net)) weights, got $(length(weights))"))

    mark = 0

    W1 = @view reshape(weights[mark+1 : mark + net.H1 * net.Din], net.H1, net.Din)
    mark += net.H1 * net.Din

    b1 = @view weights[mark+1 : mark + net.H1]
    mark += net.H1

    W2 = @view reshape(weights[mark+1 : mark + net.H2 * net.H1], net.H2, net.H1)
    mark += net.H2 * net.H1

    b2 = @view weights[mark+1 : mark + net.H2]
    mark += net.H2

    W3 = @view reshape(weights[mark+1 : mark + net.Dout * net.H2], net.Dout, net.H2)
    mark += net.Dout * net.H2

    b3 = @view weights[mark+1 : mark + net.Dout]

    return W1, b1, W2, b2, W3, b3
end


# ============================================================
# Forward passes: two-layer
# ============================================================

function forward!(
    Y::AbstractMatrix,
    Hbuf::AbstractMatrix,
    net::TwoLayerNetwork,
    weights::AbstractVector,
    X::AbstractMatrix,
)
    N = size(X, 2)

    size(X, 1) == inputdim(net) ||
        throw(DimensionMismatch("X must have $(inputdim(net)) rows, got $(size(X,1))"))

    size(Hbuf) == (net.H, N) ||
        throw(DimensionMismatch("Hbuf must have size ($(net.H), $N)"))

    size(Y) == (outputdim(net), N) ||
        throw(DimensionMismatch("Y must have size ($(outputdim(net)), $N)"))

    W1, b1, W2, b2 = unpackweights(net, weights)

    mul!(Hbuf, W1, X)

    @inbounds for j in 1:N
        for i in 1:net.H
            Hbuf[i, j] += b1[i]
        end
    end

    @inbounds for j in 1:N
        for i in 1:net.H
            Hbuf[i, j] = net.f(Hbuf[i, j])
        end
    end

    mul!(Y, W2, Hbuf)

    @inbounds for j in 1:N
        for i in 1:outputdim(net)
            Y[i, j] += b2[i]
        end
    end

    return Y
end

function forward!(ws::TwoLayerWorkspace, net::TwoLayerNetwork, weights::AbstractVector, X::AbstractMatrix)
    forward!(ws.Ybuf, ws.Hbuf, net, weights, X)
end

function forward!(net::BufferedTwoLayerNetwork, weights::AbstractVector, X::AbstractMatrix)
    N = size(X, 2)
    size(net.ws.Hbuf, 2) == N ||
        throw(DimensionMismatch("buffered network was built for batch size $(size(net.ws.Hbuf, 2)), got input with $N columns"))
    forward!(net.ws, net.net, weights, X)
end


# ============================================================
# Forward passes: three-layer
# ============================================================

function forward!(
    Y::AbstractMatrix,
    H1buf::AbstractMatrix,
    H2buf::AbstractMatrix,
    net::ThreeLayerNetwork,
    weights::AbstractVector,
    X::AbstractMatrix,
)
    N = size(X, 2)

    size(X, 1) == inputdim(net) ||
        throw(DimensionMismatch("X must have $(inputdim(net)) rows, got $(size(X,1))"))

    size(H1buf) == (net.H1, N) ||
        throw(DimensionMismatch("H1buf must have size ($(net.H1), $N)"))

    size(H2buf) == (net.H2, N) ||
        throw(DimensionMismatch("H2buf must have size ($(net.H2), $N)"))

    size(Y) == (outputdim(net), N) ||
        throw(DimensionMismatch("Y must have size ($(outputdim(net)), $N)"))

    W1, b1, W2, b2, W3, b3 = unpackweights(net, weights)

    mul!(H1buf, W1, X)

    @inbounds for j in 1:N
        for i in 1:net.H1
            H1buf[i, j] += b1[i]
        end
    end

    @inbounds for j in 1:N
        for i in 1:net.H1
            H1buf[i, j] = net.f(H1buf[i, j])
        end
    end

    mul!(H2buf, W2, H1buf)

    @inbounds for j in 1:N
        for i in 1:net.H2
            H2buf[i, j] += b2[i]
        end
    end

    @inbounds for j in 1:N
        for i in 1:net.H2
            H2buf[i, j] = net.f(H2buf[i, j])
        end
    end

    mul!(Y, W3, H2buf)

    @inbounds for j in 1:N
        for i in 1:outputdim(net)
            Y[i, j] += b3[i]
        end
    end

    return Y
end

function forward!(ws::ThreeLayerWorkspace, net::ThreeLayerNetwork, weights::AbstractVector, X::AbstractMatrix)
    forward!(ws.Ybuf, ws.H1buf, ws.H2buf, net, weights, X)
end

function forward!(net::BufferedThreeLayerNetwork, weights::AbstractVector, X::AbstractMatrix)
    N = size(X, 2)
    size(net.ws.H1buf, 2) == N ||
        throw(DimensionMismatch("buffered network was built for batch size $(size(net.ws.H1buf, 2)), got input with $N columns"))
    forward!(net.ws, net.net, weights, X)
end


# ============================================================
# Call overloads
# ============================================================

function (net::AbstractBufferedNeuralNetwork)(weights::AbstractVector, X::AbstractMatrix)
    forward!(net, weights, X)
    return workspace(net).Ybuf
end

function (net::TwoLayerNetwork)(weights::AbstractVector, X::AbstractMatrix)
    T = promote_type(eltype(weights), eltype(X))
    bnet = BufferedTwoLayerNetwork(net, size(X, 2), T)
    bnet(weights, X)
end

function (net::ThreeLayerNetwork)(weights::AbstractVector, X::AbstractMatrix)
    T = promote_type(eltype(weights), eltype(X))
    bnet = BufferedThreeLayerNetwork(net, size(X, 2), T)
    bnet(weights, X)
end


# # ============================================================
# # Remake helpers
# # ============================================================

# function remake(net::BufferedTwoLayerNetwork, N::Integer, ::Type{T}=eltype(net.ws.Hbuf)) where {T}
#     BufferedTwoLayerNetwork(net.net, N, T)
# end

# function remake(net::BufferedThreeLayerNetwork, N::Integer, ::Type{T}=eltype(net.ws.H1buf)) where {T}
#     BufferedThreeLayerNetwork(net.net, N, T)
# end