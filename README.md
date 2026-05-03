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

```python
>>> import estimator
>>> estimator.expect_cost(c=5, t=14, s=16, q=4, verbose=True)
[...]
Computing the probability that the folded error has weight 14
Prange --> 181.08545120625692
Lee-Brickell --> 169.5462959218013
Stern --> 175.78460971203532
Optimized_Stern --> 164.2579898159922
Folded code has length 3645
[...]
```

To get a suitable number of errors to achieve a specific security level,
call `find_t`:

```python
>>> estimator.find_t(c=5, s=15, q=4, security_parameter=128, verbose=True)
[...]
For s=15, c=5, we need t=12 for a security estimated to 128.83 bits
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
