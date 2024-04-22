# tvb_kernels

Provides some kernels to speed up tricky bits of TVB simulations.

## scope

- time delayed connectivity
- fused dfun-heun

## approach

- all kernels written in two forms
  - 1 instance of computation
  - N instances of computation (batch)
- C++ templates to generate variations

## optionally

- vjp implementations for gradients
- 

## mirrors

- https://github.com/maedoc/tvb_kernels
- https://gitlab.ebrains.eu/woodman/tvb_kernels


## setup environment

### pip

```
pip install -e '.[dev]'
```

### spack

### docker

todo

