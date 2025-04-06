# PyTorch Broadcasting

Upon going through the fastai course, something caught my attention that I have never heard of before. I have previously studied a bit of Machine Learning, so I have experience using PyTorch and have an understanding of some important concepts, such as, Neural Networks, activation functions like ReLU, and even Stochastic Gradient Descent (SGD). However, while reading fastai's jupyter notebook called `04_mnist_basics.ipynb`, I was surprised to learn about PyTorch's broadcasting ability. It is quite simple to understand, but seems to be an especially important mechanic to make training deep learning models faster and less code verbose.

To start, let's consider the following simple example of PyTorch broadcasting, which uses two tensors and multiplies them together. 

```python
tensor([1, 2, 3]) * tensor(2)
```

After running the above code the output will be `tensor([2, 4, 6])`. So what has happened? Well, it's clear that PyTorch has multiplied the scalar `2` value between each number in the `[1, 2, 3]` tensor, even though we never explicitly told PyTorch to do so. Under the hood, PyTorch has implicitly 'stretched' the `2` tensor into the same shape as the `[1, 2, 3]` tensor, so it's more analogous to the following code, with element-wise multiplication.

```python
tensor([1, 2, 3]) * tensor([2, 2, 2])
```

Now you may be thinking, if the `2` is copied multiple times to form a new tensor, won't this cause an issue of extreme memory usage when using a very larger tensor, such as a high quality image? And you would be correct. However, there was a reason I put the quotation marks around 'stretched', in the above example. And that is because PyTorch is smarter than this, it doesn't actually create a copy of the `2` value at all, in this instance, it has just multiplied the same `2` value across each number.

Thus the implicit application of broadcasting by PyTorch cuts down severely on the amount of memory that would be taken up by this sort of computation normally. And considering that PyTorch can compute these calculations in parallel too, adds to this features effectiveness for deep learning.

If you would like to know more about PyTorch broadcasting, then I would recommend reading the [PyTorch documentation](https://pytorch.org/docs/stable/notes/broadcasting.html). And also checkout the [NumPy documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html) for it too, which has better visuals and might make it easier to understand. 


{% include signature.html %}