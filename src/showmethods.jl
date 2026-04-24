# ============================================================
# Helpers
# ============================================================

_activation_name(f) = nameof(typeof(f))

# More readable fallback for anonymous functions / callable structs
function _activation_string(f)
    T = typeof(f)
    if isdefined(T, :name)
        return string(nameof(T))
    else
        return string(T)
    end
end


# ============================================================
# TwoLayerNetwork
# ============================================================

function Base.show(io::IO, net::TwoLayerNetwork)
    print(io,
        "TwoLayerNetwork(",
        "in=", net.Din,
        ", H=", net.H,
        ", out=", net.Dout,
        ", f=", _activation_string(net.f),
        ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", net::TwoLayerNetwork)
    println(io, "TwoLayerNetwork")
    println(io, "  input dimension:  ", net.Din)
    println(io, "  hidden dimension: ", net.H)
    println(io, "  output dimension: ", net.Dout)
    println(io, "  activation:       ", _activation_string(net.f))
    print(io,   "  number of weights: ", numweights(net))
end


# ============================================================
# ThreeLayerNetwork
# ============================================================

function Base.show(io::IO, net::ThreeLayerNetwork)
    print(io,
        "ThreeLayerNetwork(",
        "in=", net.Din,
        ", H1=", net.H1,
        ", H2=", net.H2,
        ", out=", net.Dout,
        ", f=", _activation_string(net.f),
        ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", net::ThreeLayerNetwork)
    println(io, "ThreeLayerNetwork")
    println(io, "  input dimension:   ", net.Din)
    println(io, "  hidden dimension 1:", " ", net.H1)
    println(io, "  hidden dimension 2:", " ", net.H2)
    println(io, "  output dimension:  ", net.Dout)
    println(io, "  activation:        ", _activation_string(net.f))
    print(io,   "  number of weights: ", numweights(net))
end


# ============================================================
# TwoLayerWorkspace
# ============================================================

function Base.show(io::IO, ws::TwoLayerWorkspace)
    print(io,
        "TwoLayerWorkspace(",
        "Hbuf=", size(ws.Hbuf),
        ", Ybuf=", size(ws.Ybuf),
        ", eltype=", eltype(ws.Hbuf),
        ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", ws::TwoLayerWorkspace)
    println(io, "TwoLayerWorkspace")
    println(io, "  hidden buffer size: ", size(ws.Hbuf))
    println(io, "  output buffer size: ", size(ws.Ybuf))
    print(io,   "  element type:       ", eltype(ws.Hbuf))
end


# ============================================================
# ThreeLayerWorkspace
# ============================================================

function Base.show(io::IO, ws::ThreeLayerWorkspace)
    print(io,
        "ThreeLayerWorkspace(",
        "H1buf=", size(ws.H1buf),
        ", H2buf=", size(ws.H2buf),
        ", Ybuf=", size(ws.Ybuf),
        ", eltype=", eltype(ws.H1buf),
        ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", ws::ThreeLayerWorkspace)
    println(io, "ThreeLayerWorkspace")
    println(io, "  hidden buffer 1 size: ", size(ws.H1buf))
    println(io, "  hidden buffer 2 size: ", size(ws.H2buf))
    println(io, "  output buffer size:   ", size(ws.Ybuf))
    print(io,   "  element type:         ", eltype(ws.H1buf))
end


# ============================================================
# BufferedTwoLayerNetwork
# ============================================================

function Base.show(io::IO, net::BufferedTwoLayerNetwork)
    N = size(net.ws.Ybuf, 2)
    print(io,
        "BufferedTwoLayerNetwork(",
        "in=", inputdim(net),
        ", H=", net.net.H,
        ", out=", outputdim(net),
        ", N=", N,
        ", T=", eltype(net.ws.Hbuf),
        ", f=", _activation_string(net.net.f),
        ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", net::BufferedTwoLayerNetwork)
    N = size(net.ws.Ybuf, 2)
    println(io, "BufferedTwoLayerNetwork")
    println(io, "  input dimension:   ", inputdim(net))
    println(io, "  hidden dimension:  ", net.net.H)
    println(io, "  output dimension:  ", outputdim(net))
    println(io, "  fixed batch size:  ", N)
    println(io, "  element type:      ", eltype(net.ws.Hbuf))
    println(io, "  activation:        ", _activation_string(net.net.f))
    print(io,   "  number of weights: ", numweights(net))
end


# ============================================================
# BufferedThreeLayerNetwork
# ============================================================

function Base.show(io::IO, net::BufferedThreeLayerNetwork)
    N = size(net.ws.Ybuf, 2)
    print(io,
        "BufferedThreeLayerNetwork(",
        "in=", inputdim(net),
        ", H1=", net.net.H1,
        ", H2=", net.net.H2,
        ", out=", outputdim(net),
        ", N=", N,
        ", T=", eltype(net.ws.H1buf),
        ", f=", _activation_string(net.net.f),
        ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", net::BufferedThreeLayerNetwork)
    N = size(net.ws.Ybuf, 2)
    println(io, "BufferedThreeLayerNetwork")
    println(io, "  input dimension:   ", inputdim(net))
    println(io, "  hidden dimension 1:", " ", net.net.H1)
    println(io, "  hidden dimension 2:", " ", net.net.H2)
    println(io, "  output dimension:  ", outputdim(net))
    println(io, "  fixed batch size:  ", N)
    println(io, "  element type:      ", eltype(net.ws.H1buf))
    println(io, "  activation:        ", _activation_string(net.net.f))
    print(io,   "  number of weights: ", numweights(net))
end