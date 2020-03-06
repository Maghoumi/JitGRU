
JitGRU: GRU with PyTorch's TorchScript
===

A simple implementation of GRUs using PyTorch's JIT (TorchScript). The API follows that of `torch.nn.GRU`. Should run reasonably fast.

But... why?
---
At the time of writing, PyTorch does not support second order derivatives for GRUs with CUDA (see [this issue](https://github.com/pytorch/pytorch/issues/5261)). As a result, any loss function that depends on computing the second derivatives of GRUs doesn't work on out of the box. I needed double `backward()` calls for a project, so here it is!

How to use
---
I've implemented equivalents of `torch.nn.GRUCell` and `torch.nn.GRU`. Look at the test cases that I've included in the implementation. Those should help you get started.

Contribution
---
I'm actively using this implementation, so contributions are greatly welcome as they help my work too. If you think you can improve this project, or implement something more efficiently, then feel free to submit pull requests!

License
---
This project is licensed under the MIT License.
