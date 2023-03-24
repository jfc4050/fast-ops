# Evolved Sign Momentum (Lion)

**See [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/pdf/2302.06675.pdf)**

Fused Lion implementation 😾

## Benchmarks
First working version is about 3x faster than the
[vanilla implementation taken from Google AutoML](https://github.com/google/automl/blob/master/lion/lion_pytorch.py).
Gap would probably be smaller if we let `torch.jit` fuse some of it
(needs some reworking/rearranging to enable this).
I'll get around to profiling/optimizing later.
```
[------------------ lion step -------------------]
                               |   ours  |   ref
1 threads: ---------------------------------------
      8.2e+03, torch.float16   |   25.1  |    84.5
      8.2e+03, torch.bfloat16  |   25.0  |    84.8
      6.7e+07, torch.float16   |  651.9  |  1878.6
      6.7e+07, torch.bfloat16  |  710.1  |  1889.7

Times are in microseconds (us).
```
