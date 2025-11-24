# Mathematical Background

## Introduction to Sketching

**Sketching** is a dimensionality reduction technique that uses random projections to compress large matrices while preserving their essential structure. The key idea is to multiply a large matrix by a much smaller random matrix, creating a "sketch" that captures the important features of the original data.

## The Streaming Low-Rank Approximation Problem

Given a matrix ``A \in \mathbb{R}^{m \times n}`` that arrives as a stream of columns ``\{a_1, a_2, \ldots, a_n\}``, we want to compute a rank-``r`` approximation:

```math
A \approx U \Sigma V^T
```

where:
- ``U \in \mathbb{R}^{m \times r}`` contains left singular vectors
- ``\Sigma \in \mathbb{R}^{r \times r}`` is diagonal with singular values
- ``V \in \mathbb{R}^{n \times r}`` contains right singular vectors

The challenge is to compute this approximation **incrementally** as columns arrive, without storing the entire matrix ``A``.

## Sketch-and-Solve Framework

The SketchySVD algorithm [TYUC2019](@cite) maintains four sketches:

1. **Range sketch** ``Y \in \mathbb{R}^{m \times k}``: Captures the column space of ``A``
2. **Corange sketch** ``X \in \mathbb{R}^{k \times n}``: Captures the row space of ``A``
3. **Core sketch** ``Z \in \mathbb{R}^{s \times s}``: Provides tighter approximation
4. **Error sketch** ``E \in \mathbb{R}^{q \times n}``: Estimates approximation error (optional)

These sketches are updated incrementally as each column ``a_j`` arrives:

```math
\begin{aligned}
X_{:,j} &\leftarrow X_{:,j} + \Xi a_j \\
Y &\leftarrow Y + a_j \cdot (\Omega^T)_{j,:} \\
Z &\leftarrow Z + (\Phi a_j) \cdot (\Psi^T)_{j,:} \\
E_{:,j} &\leftarrow E_{:,j} + \Theta a_j
\end{aligned}
```

where ``\Xi, \Omega, \Phi, \Psi`` are random test matrices (dimension reduction maps), and ``\Theta`` is a Gaussian test matrix for error estimation.

## Dimension Selection

The sketching dimensions are chosen based on the target rank ``r``:

**Without storage budget** (Equation 5.3 in [TYUC2019](@cite)):
```math
\begin{aligned}
k &= 4r + \alpha \\
s &= 2k + \alpha \\
q &= m
\end{aligned}
```

**With storage budget** ``T`` (Equation 5.5-5.6 in [TYUC2019](@cite)):
```math
\begin{aligned}
k &= \left\lfloor \frac{\sqrt{(m+n+4\alpha)^2 + 16(T-\alpha^2)} - (m+n+4\alpha)}{8} \right\rfloor \\
s &= \left\lfloor \sqrt{T - k(m+n)} \right\rfloor
\end{aligned}
```

where ``\alpha = 1`` for real matrices and ``\alpha = 0`` for complex matrices.

The constraint ``k \leq s \leq \min\{m,n\}`` must be satisfied.

## Forgetting Factors

For time-varying data, SketchySVD supports **exponential forgetting** through parameters ``\eta`` (forgetting factor) and ``\nu`` (scaling factor):

```math
\begin{aligned}
X_{:,j} &\leftarrow \eta X_{:,j} + \nu \Xi a_j \\
Y &\leftarrow \eta Y + \nu a_j \cdot (\Omega^T)_{j,:} \\
Z &\leftarrow \eta Z + \nu (\Phi a_j) \cdot (\Psi^T)_{j,:}
\end{aligned}
```

Typically:
- ``\eta = 1, \nu = 1``: Standard incremental update (stationary data)
- ``\eta < 1, \nu = 1``: Exponential forgetting (non-stationary data)
- ``\eta = 0, \nu = 1``: Sliding window

## Finalization Step

After all columns have been processed, the SVD is recovered from the sketches:

1. **Orthonormalize sketches**:
   ```math
   Q_Y = \text{orth}(Y), \quad Q_X = \text{orth}(X^T)
   ```

2. **Form core matrix**:
   ```math
   C = (\Phi Q_Y)^{-1} Z (Q_X^T \Psi)^{-1}
   ```

3. **Compute SVD of core**:
   ```math
   C = \tilde{U} \tilde{\Sigma} \tilde{V}^T
   ```

4. **Recover final approximation**:
   ```math
   U = Q_Y \tilde{U}_{:,1:r}, \quad \Sigma = \tilde{\Sigma}_{1:r,1:r}, \quad V = Q_X \tilde{V}_{:,1:r}
   ```

## Error Estimation

When ``E`` is maintained, the Frobenius norm error can be estimated as:

```math
\|A - U\Sigma V^T\|_F \approx \frac{\|E - \Theta \hat{A}\|_2}{\sqrt{\beta q}}
```

where ``\hat{A} = Q_Y C Q_X^T`` is the reconstruction and ``\beta = 1`` for real matrices, ``\beta = 2`` for complex matrices.

## Spectral Decay Analysis

When computing the **scree plot**, SketchySVD estimates the relative energy captured at each rank:

```math
\text{scree}(r) = \left(\frac{\sum_{i=r+1}^s \sigma_i(\hat{A}) + \epsilon}{\|A\|_F}\right)^2
```

where ``\epsilon`` is the estimated error and ``\sigma_i(\hat{A})`` are singular values of the approximation.

## Theoretical Guarantees

Under appropriate conditions [TYUC2019](@cite), the SketchySVD approximation satisfies:

```math
\mathbb{E}\left[\|A - U_r\Sigma_r V_r^T\|_F^2\right] \leq \tau_{r+1}^2(A) + 2 \left[ \frac{s-\alpha}{s-k-\alpha} \cdot \min_{\rho < k-\alpha} \frac{k+\rho-\alpha}{k-\rho-\alpha} \cdot \tau_{\rho+1}^2(A) \right]^{1/2}
```

where 

```math
\tau_{r+1}^2(A) = \sum_{j \geq r} \sigma_j^2(A)
```

and this error depends on the sketching dimensions ``k, s`` and decreases as these increase.

The algorithm has:
- **Memory complexity**: ``O(k(m+n) + s^2 + qn)``
- **Time complexity**: ``O(kmn + smn)`` for dense test matrices, less for structured ones
- **Update complexity**: ``O(km + sn)`` per column
