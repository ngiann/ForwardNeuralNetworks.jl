using Test
using Random
using LinearAlgebra

using ForwardNeuralNetworks

const ATOL = 1e-12
const RTOL = 1e-12

@testset "ForwardNeuralNetworks" begin

    rng = MersenneTwister(1234)

    @testset "TwoLayerNetwork interface and consistency" begin
        Din, H, Dout, N = 3, 5, 2, 7

        net = TwoLayerNetwork(in=Din, H=H, out=Dout, f=tanh)
        bnet_from_net = BufferedTwoLayerNetwork(net, N, Float64)
        bnet_direct   = BufferedTwoLayerNetwork(in=Din, H=H, out=Dout, N=N, f=tanh, T=Float64)

        @test inputdim(net) == Din
        @test outputdim(net) == Dout
        @test inputdim(bnet_from_net) == Din
        @test outputdim(bnet_from_net) == Dout
        @test numweights(net) == H*Din + H + Dout*H + Dout
        @test numweights(bnet_from_net) == numweights(net)
        @test numweights(bnet_direct) == numweights(net)

        X = randn(rng, Din, N)
        w = randn(rng, numweights(net))

        # allocating unbuffered call
        Y1 = net(w, X)

        # buffered call overload
        Y2 = bnet_from_net(w, X)

        # explicit forward! through buffered network
        Y3 = forward!(bnet_from_net, w, X)

        # direct workspace/net forward!
        Y4 = forward!(bnet_from_net.ws, bnet_from_net.net, w, X)

        # explicit buffer forward!
        Y5 = forward!(bnet_from_net.ws.Ybuf, bnet_from_net.ws.Hbuf, bnet_from_net.net, w, X)

        # direct-constructor buffered network
        Y6 = bnet_direct(w, X)

        @test size(Y1) == (Dout, N)
        @test size(Y2) == (Dout, N)
        @test size(Y3) == (Dout, N)
        @test size(Y4) == (Dout, N)
        @test size(Y5) == (Dout, N)
        @test size(Y6) == (Dout, N)

        @test Y1 ≈ Y2 atol=ATOL rtol=RTOL
        @test Y1 ≈ Y3 atol=ATOL rtol=RTOL
        @test Y1 ≈ Y4 atol=ATOL rtol=RTOL
        @test Y1 ≈ Y5 atol=ATOL rtol=RTOL
        @test Y1 ≈ Y6 atol=ATOL rtol=RTOL

        # repeated calls should remain consistent
        Y7 = bnet_from_net(w, X)
        Y8 = bnet_from_net(w, X)
        @test Y7 ≈ Y8 atol=ATOL rtol=RTOL

        # remake
        N2 = 11
        X2 = randn(rng, Din, N2)
        bnet2 = remake(bnet_from_net, N2, Float64)
        Y9 = bnet2(w, X2)
        Y10 = net(w, X2)

        @test size(Y9) == (Dout, N2)
        @test Y9 ≈ Y10 atol=ATOL rtol=RTOL
    end


    @testset "ThreeLayerNetwork interface and consistency" begin
        Din, H1, H2, Dout, N = 4, 6, 3, 2, 8

        net = ThreeLayerNetwork(in=Din, H1=H1, H2=H2, out=Dout, f=tanh)
        bnet_from_net = BufferedThreeLayerNetwork(net, N, Float64)
        bnet_direct   = BufferedThreeLayerNetwork(in=Din, H1=H1, H2=H2, out=Dout, N=N, f=tanh, T=Float64)

        @test inputdim(net) == Din
        @test outputdim(net) == Dout
        @test inputdim(bnet_from_net) == Din
        @test outputdim(bnet_from_net) == Dout
        @test numweights(net) == H1*Din + H1 + H2*H1 + H2 + Dout*H2 + Dout
        @test numweights(bnet_from_net) == numweights(net)
        @test numweights(bnet_direct) == numweights(net)

        X = randn(rng, Din, N)
        w = randn(rng, numweights(net))

        # allocating unbuffered call
        Y1 = net(w, X)

        # buffered call overload
        Y2 = bnet_from_net(w, X)

        # explicit forward! through buffered network
        Y3 = forward!(bnet_from_net, w, X)

        # direct workspace/net forward!
        Y4 = forward!(bnet_from_net.ws, bnet_from_net.net, w, X)

        # explicit buffer forward!
        Y5 = forward!(
            bnet_from_net.ws.Ybuf,
            bnet_from_net.ws.H1buf,
            bnet_from_net.ws.H2buf,
            bnet_from_net.net,
            w,
            X,
        )

        # direct-constructor buffered network
        Y6 = bnet_direct(w, X)

        @test size(Y1) == (Dout, N)
        @test size(Y2) == (Dout, N)
        @test size(Y3) == (Dout, N)
        @test size(Y4) == (Dout, N)
        @test size(Y5) == (Dout, N)
        @test size(Y6) == (Dout, N)

        @test Y1 ≈ Y2 atol=ATOL rtol=RTOL
        @test Y1 ≈ Y3 atol=ATOL rtol=RTOL
        @test Y1 ≈ Y4 atol=ATOL rtol=RTOL
        @test Y1 ≈ Y5 atol=ATOL rtol=RTOL
        @test Y1 ≈ Y6 atol=ATOL rtol=RTOL

        # repeated calls should remain consistent
        Y7 = bnet_from_net(w, X)
        Y8 = bnet_from_net(w, X)
        @test Y7 ≈ Y8 atol=ATOL rtol=RTOL

        # remake
        N2 = 12
        X2 = randn(rng, Din, N2)
        bnet2 = remake(bnet_from_net, N2, Float64)
        Y9 = bnet2(w, X2)
        Y10 = net(w, X2)

        @test size(Y9) == (Dout, N2)
        @test Y9 ≈ Y10 atol=ATOL rtol=RTOL
    end


    @testset "Alternative activation functions" begin
        Din, H, Dout, N = 2, 4, 3, 5

        relu(x) = max(x, zero(x))

        net = TwoLayerNetwork(in=Din, H=H, out=Dout, f=relu)
        bnet = BufferedTwoLayerNetwork(net, N, Float64)

        X = randn(rng, Din, N)
        w = randn(rng, numweights(net))

        @test net(w, X) ≈ bnet(w, X) atol=ATOL rtol=RTOL
    end


    @testset "Error handling: TwoLayerNetwork" begin
        Din, H, Dout, N = 3, 5, 2, 7

        net = TwoLayerNetwork(in=Din, H=H, out=Dout)
        bnet = BufferedTwoLayerNetwork(net, N, Float64)

        X = randn(rng, Din, N)
        w = randn(rng, numweights(net))

        # wrong number of weights
        @test_throws DimensionMismatch net(w[1:end-1], X)
        @test_throws DimensionMismatch bnet(w[1:end-1], X)

        # wrong input dimension
        Xbad = randn(rng, Din + 1, N)
        @test_throws DimensionMismatch net(w, Xbad)
        @test_throws DimensionMismatch bnet(w, Xbad)

        # wrong batch size for buffered net
        XbadN = randn(rng, Din, N + 1)
        @test_throws DimensionMismatch bnet(w, XbadN)

        # wrong explicit workspace sizes
        Hbuf_bad = zeros(H + 1, N)
        Ybuf_bad = zeros(Dout, N)
        @test_throws DimensionMismatch forward!(Ybuf_bad, Hbuf_bad, net, w, X)
    end


    @testset "Error handling: ThreeLayerNetwork" begin
        Din, H1, H2, Dout, N = 3, 4, 5, 2, 6

        net = ThreeLayerNetwork(in=Din, H1=H1, H2=H2, out=Dout)
        bnet = BufferedThreeLayerNetwork(net, N, Float64)

        X = randn(rng, Din, N)
        w = randn(rng, numweights(net))

        # wrong number of weights
        @test_throws DimensionMismatch net(w[1:end-1], X)
        @test_throws DimensionMismatch bnet(w[1:end-1], X)

        # wrong input dimension
        Xbad = randn(rng, Din + 1, N)
        @test_throws DimensionMismatch net(w, Xbad)
        @test_throws DimensionMismatch bnet(w, Xbad)

        # wrong batch size for buffered net
        XbadN = randn(rng, Din, N + 1)
        @test_throws DimensionMismatch bnet(w, XbadN)

        # wrong explicit workspace sizes
        H1buf_bad = zeros(H1 + 1, N)
        H2buf_bad = zeros(H2, N)
        Ybuf_bad  = zeros(Dout, N)
        @test_throws DimensionMismatch forward!(Ybuf_bad, H1buf_bad, H2buf_bad, net, w, X)
    end


    @testset "Workspace sizes are correct" begin
        net2 = TwoLayerNetwork(in=3, H=5, out=2)
        ws2 = ForwardNeuralNetworks.TwoLayerWorkspace(net2, 9, Float64)
        @test size(ws2.Hbuf) == (5, 9)
        @test size(ws2.Ybuf) == (2, 9)

        net3 = ThreeLayerNetwork(in=4, H1=6, H2=3, out=2)
        ws3 = ForwardNeuralNetworks.ThreeLayerWorkspace(net3, 10, Float64)
        @test size(ws3.H1buf) == (6, 10)
        @test size(ws3.H2buf) == (3, 10)
        @test size(ws3.Ybuf) == (2, 10)
    end
end