
JitGRU: GRU with PyTorch's TorchScript
===

A simple implementation of GRUs using PyTorch's JIT (TorchScript). The API follows that of `torch.nn.GRU`. Should run reasonably fast.

But... why?
---
At the time of writing, PyTorch does not support second order derivatives for GRUs with CUDA (see [this issue](https://github.com/pytorch/pytorch/issues/5261)). As a result, any loss function that depends on computing the second derivatives of GRUs doesn't work on out of the box. I needed double `backward()` calls for a project, so here it is!

How to use
---
The main implementation is available in [jit_gru.py](https://github.com/Maghoumi/JitGRU/blob/master/jit_gru.py).
I've implemented equivalents of `torch.nn.GRUCell` and `torch.nn.GRU` in that file. Look at the test cases that I've included in the implementation. Those should help you get started.

Bi-Directional GRUs
---
Support for bi-directional GRUs with variable input lengths was recently added (credits go to [@elixir-code](https://github.com/elixir-code)). This implementation is available separately in [jit_bigru.py](https://github.com/Maghoumi/JitGRU/blob/master/jit_bigru.py). See the included test cases in that file for example usage.

Demo Project
---
Checkout [DeepNAG](https://github.com/Maghoumi/DeepNAG), which contains a GAN-based sequence generation model, as well as a non-adversarial sequence generator.
The GAN-based sequence generator in the aforementioned repository is trained with the [improved Wasserstein GAN](https://github.com/caogang/wgan-gp) loss function, and relies on the code from this repository.

<p align="center">
  <img width="400" src="https://github.com/Maghoumi/DeepNAG/raw/master/images/kick.gif"/>
  <img width="400" src="https://github.com/Maghoumi/DeepNAG/raw/master/images/uppercut.gif"/>
</p>


Support/Citing
---
If you find our work useful, please consider starring this repository and citing our work:

```
@phdthesis{maghoumi2020dissertation,
  title={{Deep Recurrent Networks for Gesture Recognition and Synthesis}},
  author={Mehran Maghoumi},
  year={2020},
  school={University of Central Florida Orlando, Florida}
}

@misc{maghoumi2020deepnag,
      title={{DeepNAG: Deep Non-Adversarial Gesture Generation}}, 
      author={Mehran Maghoumi and Eugene M. Taranta II and Joseph J. LaViola Jr},
      year={2020},
      eprint={2011.09149},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Contribution
---
I'm actively using this implementation, so contributions are greatly welcome as they help my work too. If you think you can improve this project, or implement something more efficiently, then feel free to submit pull requests!

License
---
This project is licensed under the MIT License.
