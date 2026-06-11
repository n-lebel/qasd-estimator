# Security estimator for the QA-SD problem

Python tool to select parameters for the quasi-abelian syndrome decoding problem and its stationary variant.
[paper](https://eprint.iacr.org/2024/429.pdf) for details. Adapted by the authors of [[LXXY+26]](https://eprint.iacr.org/2026/196.pdf) to cover the stationary variant and include the newer attack from [[BDHV25]](https://eprint.iacr.org/2025/892.pdf).

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to run the estimator

The estimator is exposed as the pure-Python `estimator` module — no SageMath
required.

The prime field is specified by its **bit size** via `q_bits`. The estimator
uses the Mersenne-shaped representative `q = 2^q_bits − 1` internally (so
`d = 2` always satisfies the `d | q − 1` requirement); the lower-order bits of
`q` do not affect the security estimate.

```python
>>> import estimator
>>> estimator.expect_cost(c=5, t=14, s=16, q_bits=107, verbose=True)
[...]
Computing the probability that the folded error has weight 14
Prange --> ...
Lee-Brickell --> ...
Stern --> ...
Optimized_Stern --> ...
Folded code has length ...
[...]
```

To get a suitable number of errors to achieve a specific security level,
call `find_t`:

```python
>>> estimator.find_t(c=5, s=15, q_bits=107, security_parameter=128, verbose=True)
[...]
For s=15, c=5, q_bits=107, d=2, we need t=... for a security estimated to ... bits
```

## Levels of aggressivity in parameter selection

Historically, the analysis of attacks against the decoding problem at the
GV bound (essentially ISD algorithms) has focused on optimising the
exponent of the complexity. In particular, polynomial factors have often
been ignored when designing parameters in code-based cryptography. We
consider this overly conservative and count them by default. To ignore
polynomial factors, set the module-level `conservative = False` in
`estimator.py`. In our instances of the decoding problem, the error is
split into blocks with the same number of erroneous coordinates. No
known algorithm exploits this in our code-rate range, but to be even
more conservative, set `csplit = True` to give a provable lower bound
for putative algorithms that do.

## How to add a new ISD algorithm

Implement the complexity in `estimator.py` following the same template
as the existing algorithms, then add it to the `ISDs` dictionary.

## Web UI

A minimal [Streamlit](https://streamlit.io) UI is provided in `app.py`.
It exposes `expect_cost` and `find_t` through a browser form.

```sh
streamlit run app.py
```

## Citation

```
@inproceedings{BBCCDS24,
  author       = {Maxime Bombar and
                  Dung Bui and
                  Geoffroy Couteau and
                  Alain Couvreur and
                  Clément Ducros and
                  Sacha Servan-Schreiber},
  title        = {FOLEAGE: $\mathbb{F}_{4}$OLE-Based Multi-Party
                  Computation for Boolean Circuits},
  note         = {\url{https://eprint.iacr.org/2024/429}},
  editor       = {Kai-Min Chung and
                  Yu Sasaki},
  booktitle    = {Advances in Cryptology - {ASIACRYPT} 2024 - 30th
                  International Conference on the Theory and
                  Application of Cryptology and Information Security,
                  Kolkata, India, December 9-13, 2024 %
                  },
  publisher    = {Springer},
  year         = {2024},
}
```
