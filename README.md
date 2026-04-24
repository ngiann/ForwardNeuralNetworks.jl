# ForwardNeuralNetworks.jl

A small Julia toolkit for simple feed-forward neural networks with **flat parameter vectors** and **buffered execution objects** for performance-critical code.

The package provides:

- `TwoLayerNetwork`
- `ThreeLayerNetwork`
- `BufferedTwoLayerNetwork`
- `BufferedThreeLayerNetwork`

with a common interface through:

- `inputdim(net)`
- `outputdim(net)`
- `numweights(net)`

The intended use case is scientific and numerical code where network parameters are stored in a single vector and optimized externally.

---

## Design overview

There are two kinds of objects:

### 1. Unbuffered network objects

These describe only the architecture:

- input dimension
- hidden layer sizes
- output dimension
- activation function

They are convenient to construct and call, but allocate internal work buffers when evaluated.

Examples:

- `TwoLayerNetwork`
- `ThreeLayerNetwork`

### 2. Buffered network objects

These combine a network architecture with preallocated workspace arrays for a **fixed batch size**.

They are intended for repeated evaluation in performance-critical loops.

Examples:

- `BufferedTwoLayerNetwork`
- `BufferedThreeLayerNetwork`

---

## Input convention

Inputs are stored in a matrix `X` of size

```julia
Din × N
```

where:

- `Din` is the input dimension
- `N` is the number of data points

Each **column** of `X` is one input data point.

The output is a matrix of size

```julia
Dout × N
```

where each column is the output corresponding to the matching input column.

---

## Parameter convention

Each network is parameterized by a single flat vector `weights`.

Use

```julia
numweights(net)
```

to determine the required length.

### Two-layer network layout

For a network with dimensions:

- input: `Din`
- hidden: `H`
- output: `Dout`

the parameter vector is interpreted as:

1. `W1` of size `H × Din`
2. `b1` of length `H`
3. `W2` of size `Dout × H`
4. `b2` of length `Dout`

### Three-layer network layout

For a network with dimensions:

- input: `Din`
- hidden 1: `H1`
- hidden 2: `H2`
- output: `Dout`

the parameter vector is interpreted as:

1. `W1` of size `H1 × Din`
2. `b1` of length `H1`
3. `W2` of size `H2 × H1`
4. `b2` of length `H2`
5. `W3` of size `Dout × H2`
6. `b3` of length `Dout`

---

## Common interface

All neural network objects support:

```julia
inputdim(net)
outputdim(net)
numweights(net)
```

Examples:

```julia
net = TwoLayerNetwork(in=3, H=5, out=2)

inputdim(net)    # 3
outputdim(net)   # 2
numweights(net)  # 5*3 + 5 + 2*5 + 2 = 32
```

Buffered networks forward these queries to the stored base network.

---

## Creating networks

### Two-layer network

```julia
net = TwoLayerNetwork(in=3, H=5, out=2, f=tanh)
```

### Three-layer network

```julia
net = ThreeLayerNetwork(in=3, H1=6, H2=4, out=2, f=tanh)
```

The activation function `f` is stored as part of the network object.

---

## Evaluating unbuffered networks

Unbuffered networks can be called directly:

```julia
net = TwoLayerNetwork(in=3, H=5, out=2, f=tanh)
X = randn(3, 10)
w = randn(numweights(net))

Y = net(w, X)
```

Here:

- `X` has size `3 × 10`
- `Y` has size `2 × 10`

The same pattern works for `ThreeLayerNetwork`:

```julia
net = ThreeLayerNetwork(in=3, H1=6, H2=4, out=2, f=tanh)
X = randn(3, 10)
w = randn(numweights(net))

Y = net(w, X)
```

These calls are convenient, but they allocate temporary workspace internally.

---

## Buffered networks

Buffered networks are intended for repeated evaluations with a fixed batch size `N`.

### Construct from an existing network

```julia
net = TwoLayerNetwork(in=3, H=5, out=2, f=tanh)
bnet = BufferedTwoLayerNetwork(net, 10, Float64)
```

### Construct directly

```julia
bnet = BufferedTwoLayerNetwork(in=3, H=5, out=2, N=10, f=tanh, T=Float64)
```

Similarly for the three-layer case:

```julia
bnet = BufferedThreeLayerNetwork(in=3, H1=6, H2=4, out=2, N=10, f=tanh, T=Float64)
```

The `N` argument fixes the number of columns the buffered network is prepared to process.

---

## Evaluating buffered networks

Buffered networks can also be called directly:

```julia
X = randn(3, 10)
bnet = BufferedTwoLayerNetwork(in=3, H=5, out=2, N=10)
w = randn(numweights(bnet))

Y = bnet(w, X)
```

This reuses the internal work buffers stored inside `bnet`.

If `X` has the wrong number of columns, an error is thrown.

---

## In-place forward passes

For performance-critical code, use `forward!`.

### Two-layer case

```julia
forward!(bnet, w, X)
Y = workspace(bnet).Ybuf
```

### Three-layer case

```julia
forward!(bnet, w, X)
Y = workspace(bnet).Ybuf
```

The output is stored in the workspace buffer `Ybuf`.

This avoids allocating temporary hidden-layer arrays on each call.

---

## Accessing the underlying network and workspace

Buffered networks support:

```julia
basenet(bnet)
workspace(bnet)
```

Example:

```julia
bnet = BufferedTwoLayerNetwork(in=3, H=5, out=2, N=10)

net = basenet(bnet)
ws  = workspace(bnet)

inputdim(net)
size(ws.Ybuf)
```

---

## Rebuilding a buffered network for a different batch size

If you need a different number of columns, use `remake`:

```julia
bnet = BufferedTwoLayerNetwork(in=3, H=5, out=2, N=10)
bnet2 = remake(bnet, 20)
```

This creates a new buffered network with the same architecture and a new workspace sized for `N = 20`.

You can also change the workspace element type:

```julia
bnet3 = remake(bnet, 20, Float32)
```

---

## Example: two-layer network

```julia
using Random

rng = MersenneTwister(1)

net = TwoLayerNetwork(in=4, H=8, out=3, f=tanh)
X = randn(rng, 4, 100)
w = randn(rng, numweights(net))

Y = net(w, X)

@show size(Y)  # (3, 100)
```

Buffered version:

```julia
bnet = BufferedTwoLayerNetwork(net, 100, Float64)
forward!(bnet, w, X)
Y = workspace(bnet).Ybuf

@show size(Y)  # (3, 100)
```

---

## Example: three-layer network

```julia
using Random

rng = MersenneTwister(1)

net = ThreeLayerNetwork(in=4, H1=10, H2=6, out=2, f=tanh)
X = randn(rng, 4, 100)
w = randn(rng, numweights(net))

Y = net(w, X)

@show size(Y)  # (2, 100)
```

Buffered version:

```julia
bnet = BufferedThreeLayerNetwork(net, 100, Float64)
forward!(bnet, w, X)
Y = workspace(bnet).Ybuf

@show size(Y)  # (2, 100)
```

---

## Performance notes

### When to use unbuffered networks

Use the unbuffered types when:

- convenience matters more than absolute speed
- you evaluate the network only occasionally
- you want the simplest API

### When to use buffered networks

Use the buffered types when:

- you evaluate the same architecture many times
- batch size is fixed or changes only occasionally
- you want to avoid temporary array allocations in inner loops

Typical hot-loop pattern:

```julia
bnet = BufferedTwoLayerNetwork(in=3, H=5, out=2, N=128)
w = randn(numweights(bnet))
X = randn(3, 128)

for iter in 1:10_000
    forward!(bnet, w, X)
end
```

---

## Notes on types and performance

The package defines abstract supertypes:

- `AbstractNeuralNetwork`
- `AbstractBufferedNeuralNetwork`

This is useful for API organization and generic interface functions.

For performance-critical code, the important point is that hot loops should work with **concrete network objects**, for example:

- `TwoLayerNetwork{typeof(tanh)}`
- `BufferedTwoLayerNetwork{Float64, typeof(tanh)}`
- `BufferedThreeLayerNetwork{Float64, typeof(tanh)}`

Avoid storing networks in abstractly typed containers in hot code, such as:

```julia
Vector{AbstractNeuralNetwork}
```

unless you are willing to pay the corresponding performance cost.

---

## Summary

Use:

- `TwoLayerNetwork` or `ThreeLayerNetwork` for convenience
- `BufferedTwoLayerNetwork` or `BufferedThreeLayerNetwork` for repeated evaluation in hot loops

Core interface:

```julia
inputdim(net)
outputdim(net)
numweights(net)
forward!(net, weights, X)
```

Buffered-network utilities:

```julia
basenet(bnet)
workspace(bnet)
remake(bnet, N)
```

---

## License

Add your package license information here.