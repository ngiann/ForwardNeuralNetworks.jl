function test(N=30, seed = 1)

    rg = MersenneTwister(seed)
    
    x = randn(rg, 1, N)

    y = sin.(x) + 0.1*randn(rg, 1, N)

    # last layer <- first layer
    net = [makelayer(out=1, in=5, f=identity); makelayer(in = 1, out=5)]

    display(net)

    function objective(w)

        local pred = net(w, x)

        sum(abs2.(pred - y))

    end

    opt = Optim.Options(iterations = 1000, show_trace = true, show_every = 1)

    result = optimize(objective, randn(numweights(net)), LBFGS(), opt, autodiff=:forward)

    wopt = result.minimizer

    return x -> net(wopt, x)

end