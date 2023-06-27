# ForwardNeuralNetworks

```
# working with a single layer

l = makelayer(in=2,out=5)
w = randn(numweights(l));
l(w,randn(2,10))


# putting layers together

l1 = makelayer(in=2,out=5)
w1 = randn(numweights(l1))

l2 = makelayer(in=5,out=3)
w2 = randn(numweights(l2))

x = randn(2,10)
output = l2(w2, l1(w1,x))
net = [l2;l1]
w = [w2;w1]
output2 = net(w, x)

# output2 must be identical to output!
```