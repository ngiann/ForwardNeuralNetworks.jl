#########################################################
# Constructor
#########################################################

mutable struct layer
    w::Array{Float64, 2}
    b::Array{Float64, 1}
    f::Function
end


function layer(nou::Int, nin::Int, f = x -> tanh.(x) )
    layer(0.1*randn(nou, nin), 0.1*randn(nou), f)
end


##########################################################
# REPL
##########################################################

function Base.show(io::IO, a::layer)
    print(io, "layer with ",size(a.w,2)," inputs and ",size(a.w,1), " outputs.\nFunction is ",a.f)
end


#########################################################
# Evaluation
#########################################################

#----------------------------------------------------
function (a::layer)(x)
#----------------------------------------------------

    a.f(a.w * x .+ a.b)

end

#----------------------------------------------------
function (a::Array{layer,1})(x)
#----------------------------------------------------

    out = a[end](x)

    for l in a[end-1:-1:1]
        out = l(out)
    end

    return out

end


#----------------------------------------------------
function (a::layer)(new_weights, x)
#----------------------------------------------------

    MARK = 0
    w = reshape(new_weights[MARK+1:MARK+length(a.w)], size(a.w))

    MARK += length(a.w)
    b = reshape(new_weights[MARK+1:MARK+length(a.b)], size(a.b))

    MARK += length(a.b)

    # make sure we assigned all of the weights
    @assert(MARK == length(new_weights))

    a.f(w * x .+ b)

end


#----------------------------------------------------
function (a::Array{layer,1})(new_weights, x)
#----------------------------------------------------

    MARK = numweights(a)

    out  = a[end](new_weights[MARK-numweights(a[end])+1:MARK], x)

    MARK = MARK - numweights(a[end])

    for l in a[end-1:-1:1]
        out = l(new_weights[MARK-numweights(l)+1:MARK], out)
        MARK = MARK - numweights(l)
    end

    @assert(MARK == 0)
    return out

end




#########################################################
# Retrieve weights
#########################################################

function getparam(a::layer)
    [vec(a.w); vec(a.b)]
end

function getparam(a::Array{layer,1})
    reduce(vcat, map(getparam, a))
end


#########################################################
# Assign new weights
#########################################################


#----------------------------------------------------
function setparam!(a::Array{layer,1}, new_weights)
#----------------------------------------------------

    MARK::Int32 = 0

    for l in a

        for i in (MARK+1):(MARK + length(l.w))
            l.w[i - MARK] = new_weights[i]
        end

        MARK += length(l.w)

        for i in (MARK+1):(MARK + length(l.b))
            l.b[i - MARK] = new_weights[i]
        end

        MARK += length(l.b)

    end

    # make sure we assigned all of the weights
    @assert(MARK == length(new_weights) == numweights(a))

end


#----------------------------------------------------
function setparam!(a::layer, w, b)
#----------------------------------------------------

    copyto!(a.w, w)
    copyto!(a.b, b)

end


#----------------------------------------------------
function setparam(a::Array{layer,1}, new_weights)
#----------------------------------------------------

    MARK = 0

    new_array_of_layers = Array{layer,1}(undef, length(a))


    for (index, l) in enumerate(a)

        local w::Array{Float64,2} = reshape(new_weights[MARK+1:MARK + length(l.w)], size(l.w))
        MARK += length(l.w)

        local b::Array{Float64,1} = reshape(new_weights[MARK+1:MARK + length(l.b)], size(l.b))
        MARK += length(l.b)

        new_array_of_layers[index] = layer(w, b, l.f)

    end

    # make sure we assigned all of the weights
    @assert(MARK == length(new_weights) == numweights(new_array_of_layers) == numweights(a))

    return new_array_of_layers

end


#########################################################
# Chec for equality in weights
#########################################################



function ==(layer1::layer, layer2::layer)

    all(layer1.w .== layer2.w) && all(layer1.b .== layer2.b)

end


#########################################################
# Number of parameters in network
#########################################################

function numweights(a::layer)
    length(a.w) + length(a.b)
end

function numweights(a::Array{layer,1})
    mapreduce(numweights, +, a)
end

function numlayers(a::Array{layer,1})
    length(a)
end
