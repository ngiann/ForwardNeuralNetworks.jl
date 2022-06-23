using ForwardNeuralNetworks
using Test

@testset "ForwardNeuralNetworks.jl" begin
    
    function evaluate_with_parameters()

        f = layer(2, 10, identity);

        g = layer(10, 3);

        pf = reduce(hcat, getparam(f))[:];

        pg = reduce(hcat, getparam(g))[:];

        net = [f ; g]

        x = randn(3)

        all(net(x) .== net([pf; pg], x))

    end


    function set_param_debug()

        net1 = [layer(10,3, identity) ; layer(10,2)]
        net2 = [layer(10,3, identity) ; layer(10,2)]

        new_weights = randn(numweights(net1))

        setparam!(net1, new_weights)

        # make sure that net2 has initially different weights to net1
        other_weights = randn(numweights(net2))

        setparam!(net2, other_weights)

        # now set the same weights as net1 using the non-mutating function
        net2 = setparam(net2, new_weights)

        # check if the two networks now hold the same weights
        all(net1 .== net2)

    end


    function test_setting_and_retrieving_weights_of_deeper_net()

        net = [layer(10,5, identity) ; layer(5, 7); layer(7, 7); layer(7,2)]

        # define some random weights
        weights = randn(numweights(net))

        # set the network
        setparam!(net, weights)

        # get a prediction on some arbitrary input
        x = randn(2)
        out1 = net(x)

        # retrieve weights and set again network weights
        weights = getparam(net)
        setparam!(net, weights)

        # check if prediction is still the same as it should be
        out2 = net(x)
        
        all(out1 .== out2)

    end


    @test set_param_debug()
    @test evaluate_with_parameters()
    @test test_setting_and_retrieving_weights_of_deeper_net()


end
