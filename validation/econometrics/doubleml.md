# DoubleML Validation

## Method: Double/Debiased Machine Learning

### Implementation
- **File**: `crates/p2a-core/src/econometrics/doubleml.rs`
- **Function**: `run_double_ml`
- **MCP Tool**: `treatment_double_ml`

### Models Implemented

1. **Partially Linear Regression (PLR)**:
   - Model: Y = theta*D + g(X) + zeta, D = m(X) + V
   - Orthogonal score: psi = (Y - l(X) - theta*(D - m(X)))*(D - m(X))
   - Suitable for continuous treatment

2. **Interactive Regression Model (IRM)**:
   - Model: Y(d) = g_d(X) + U_d for d in {0,1}
   - Orthogonal score (AIPW): psi = g_1(X) - g_0(X) + D*(Y-g_1(X))/m(X) - (1-D)*(Y-g_0(X))/(1-m(X)) - theta
   - Suitable for binary treatment with heterogeneous effects

### Key Features

- **Cross-fitting**: K-fold sample splitting (default K=5) to avoid overfitting bias
- **Orthogonal scores**: Neyman-orthogonal moment conditions for root-n consistency
- **Influence function variance**: Asymptotically valid standard errors
- **Nuisance model diagnostics**: R-squared and RMSE for outcome and treatment models

### Reference Implementations

- **Python**: `DoubleML` package (Bach et al., 2022)
  - https://docs.doubleml.org/stable/
  - Version: 0.7.x
- **R**: `DoubleML` package
  - https://docs.doubleml.org/r/stable/

### Test Cases

#### Test 1: PLR Model with Known Treatment Effect

**Data Generating Process**:
```
X1, X2 ~ Uniform(0, 1)
D = 0.5*X1 + 0.3*X2 + noise  (continuous treatment)
Y = 0.5*D + 0.3*X1 + 0.2*X2 + noise  (theta = 0.5)
n = 500
```

**Rust Results** (seed = 42):
- theta_hat: ~0.45-0.55 (within 2 SE of true 0.5)
- Standard Error: ~0.03-0.10
- 95% CI contains 0.5

**Validation Criteria**:
- theta_hat within 0.2 of true value (0.3 to 0.7)
- SE positive and reasonable (< 0.5)
- CI contains true value

#### Test 2: IRM Model with Binary Treatment

**Data Generating Process**:
```
X1, X2 ~ Uniform(0, 1)
P(D=1|X) = expit(0.5 + X1 - 0.3*X2)
Y(0) = 0.3*X1 + 0.2*X2 + noise
Y(1) = 0.5 + 0.3*X1 + 0.2*X2 + noise  (ATE = 0.5)
n = 500
```

**Rust Results** (seed = 42):
- theta_hat (ATE): ~0.45-0.55
- SE: ~0.05-0.15
- 95% CI contains 0.5

### Python Reproduction Code

```python
import numpy as np
from doubleml import DoubleMLPLR, DoubleMLIRM
from doubleml import DoubleMLData
from sklearn.linear_model import LinearRegression, LogisticRegression

np.random.seed(42)
n = 500

# Generate data
X = np.random.uniform(size=(n, 2))
D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.uniform(-0.1, 0.1, n)
Y = 0.5 * D + 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.uniform(-0.1, 0.1, n)

# Create DoubleML data object
dml_data = DoubleMLData.from_arrays(X, Y, D)

# PLR with linear models (matching our implementation)
ml_l = LinearRegression()
ml_m = LinearRegression()
dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=5)
dml_plr.fit()

print(f"theta: {dml_plr.coef[0]:.4f}")
print(f"SE: {dml_plr.se[0]:.4f}")
print(f"95% CI: [{dml_plr.confint().iloc[0, 0]:.4f}, {dml_plr.confint().iloc[0, 1]:.4f}]")
```

### Tolerance Criteria

| Statistic | Tolerance |
|-----------|-----------|
| Coefficient (theta) | Within 2 SE of true value |
| Standard Error | Positive, finite |
| t-statistic | |theta|/SE |
| p-value | 2*(1 - Phi(|t|)) |
| 95% CI | Contains true value with high probability |

### Notes

1. **Nuisance models**: Current implementation uses OLS for nuisance estimation. Future versions may add Ridge, Lasso, or other ML methods.

2. **Cross-fitting**: Results depend on random fold assignments. Use `seed` parameter for reproducibility.

3. **IRM propensity trimming**: Propensity scores are clipped to [trim, 1-trim] to avoid extreme weights.

4. **Comparison with Python DoubleML**: Small differences expected due to:
   - Random seed implementation differences
   - Numerical precision in matrix operations
   - Score aggregation method (we use simple averaging)

### References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1-C68.

- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). "DoubleML - An Object-Oriented Implementation of Double Machine Learning in Python." *Journal of Machine Learning Research*, 23(53), 1-6.

### Status

- [x] Core implementation (PLR model)
- [x] IRM model for binary treatment
- [x] K-fold cross-fitting
- [x] Influence function variance estimation
- [x] Unit tests
- [x] MCP tool
- [ ] Ridge/Lasso nuisance models (future)
- [ ] Comparison with Python DoubleML on same data
