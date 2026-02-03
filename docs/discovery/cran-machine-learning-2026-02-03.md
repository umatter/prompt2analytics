# METHOD DISCOVERY REPORT
**Source:** CRAN Machine Learning Task View
**URL:** https://cran.r-project.org/web/views/MachineLearning.html
**Date:** 2026-02-03

---

## SUMMARY

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total methods scanned** | 147 | 100% |
| **Already implemented** | 38 | 26% |
| **Partially implemented** | 12 | 8% |
| **Not implemented** | 82 | 56% |
| **Skipped (out of scope)** | 15 | 10% |

---

## CATEGORY BREAKDOWN

| Category | Total | Implemented | Gaps | Priority |
|----------|-------|-------------|------|----------|
| **Regularization/Penalized** | 18 | 2 | 16 | HIGH |
| **Boosting** | 8 | 0 | 8 | HIGH |
| **Ensemble Methods** | 14 | 4 | 10 | HIGH |
| **Tree Methods** | 13 | 2 | 11 | MEDIUM |
| **SVM/Kernel** | 5 | 1 | 4 | MEDIUM |
| **Neural Networks** | 9 | 0 | 9 | LOW |
| **Bayesian Methods** | 8 | 2 | 6 | MEDIUM |
| **Explainability (XAI)** | 12 | 0 | 12 | MEDIUM |
| **Causal ML** | 4 | 2 | 2 | HIGH |
| **Association Rules** | 5 | 0 | 5 | LOW |
| **Model Selection** | 11 | 3 | 8 | MEDIUM |
| **Fuzzy/Rough Sets** | 3 | 0 | 3 | SKIP |
| **Meta-Frameworks** | 7 | 0 | 7 | SKIP |
| **Evolutionary** | 2 | 0 | 2 | SKIP |
| **Other/Misc** | 28 | 22 | 6 | - |

---

## EXISTING IMPLEMENTATIONS IN p2a-core

### Fully Implemented ✅

| R Package/Method | p2a-core Function | Module |
|------------------|-------------------|--------|
| `stats::kmeans` | `kmeans` | `ml/clustering.rs` |
| `cluster::pam` | `kmedoids` | `ml/advanced_clustering_mod.rs` |
| `cluster::agnes` | `agnes` | `ml/advanced_clustering_mod.rs` |
| `cluster::diana` | `diana` | `ml/advanced_clustering_mod.rs` |
| `cluster::clara` | `clara` | `ml/advanced_clustering_mod.rs` |
| `cluster::fanny` | `fanny` | `ml/advanced_clustering_mod.rs` |
| `dbscan::dbscan` | `dbscan` | `ml/clustering.rs` |
| `dbscan::hdbscan` | `hdbscan` | `ml/advanced_clustering_mod.rs` |
| `dbscan::optics` | `optics` | `ml/advanced_clustering_mod.rs` |
| `stats::hclust` | `hierarchical` | `ml/clustering.rs` |
| `fastcluster` | `fastcluster` | `ml/advanced_clustering_mod.rs` |
| `kernlab::specc` | `spectral_clustering` | `ml/advanced_clustering_mod.rs` |
| `apcluster` | `affinity_propagation` | `ml/advanced_clustering_mod.rs` |
| `mclust::Mclust` | `gaussian_mixture` | `ml/advanced_clustering_mod.rs` |
| `e1071::cmeans` | `fuzzy_cmeans` | `ml/advanced_clustering_mod.rs` |
| `cluster::silhouette` | `silhouette` | `ml/cluster_validation.rs` |
| `cluster::clusGap` | `gap_statistic` | `ml/cluster_validation.rs` |
| `fpc::calinhara` | `calinski_harabasz` | `ml/cluster_validation.rs` |
| `clusterCrit::db` | `davies_bouldin` | `ml/cluster_validation.rs` |
| `clValid::dunn` | `dunn_index` | `ml/cluster_validation.rs` |
| `mclust::adjustedRandIndex` | `rand_index` | `ml/cluster_validation.rs` |
| `aricode::NMI` | `nmi` | `ml/cluster_validation.rs` |
| `stats::prcomp` | `pca` | `ml/reduction.rs` |
| `Rtsne` | `tsne` | `ml/reduction.rs` |
| `stats::cmdscale` | `cmdscale` | `ml/reduction.rs` |
| `randomForest` | `random_forest` | `ml/trees.rs` |
| `e1071::svm` (linear) | `linear_svm` | `ml/svm.rs` |
| `grf::causal_forest` | `causal_forest` | `ml/causal_forest.rs` |
| `bartCause/bcf` | `bart_causal` | `ml/bart_causal.rs` |
| `DoubleML` | (via `run_doubleml`) | `econometrics/doubleml.rs` |
| `ClusterR::MiniBatchKmeans` | `mini_batch_kmeans` | `ml/advanced_clustering_mod.rs` |
| `tclust::tkmeans` | `trimmed_kmeans` | `ml/advanced_clustering_mod.rs` |
| `skmeans` | `skmeans` | `ml/advanced_clustering_mod.rs` |
| `dynamicTreeCut` | `dynamic_tree_cut` | `ml/advanced_clustering_mod.rs` |
| `mixtools::normalmixEM` | `normal_mix_em` | `ml/advanced_clustering_mod.rs` |
| `pvclust` | `pvclust` | `ml/advanced_clustering_mod.rs` |
| `flexmix` | `flexmix` | `ml/advanced_clustering_mod.rs` |
| `clustMixType::kproto` | `kprototypes` | `ml/advanced_clustering_mod.rs` |

### Partially Implemented 🔶

| R Package/Method | Status | Notes |
|------------------|--------|-------|
| `glmnet` (elastic net) | 🔶 | OLS/GLS exist, but not penalized GLM |
| `lars` (lasso) | 🔶 | Related to GLS but path algo missing |
| `nnet` (neural net) | 🔶 | PPR exists as related method |
| `e1071::svm` (kernel) | 🔶 | Only linear SVM, not kernel |
| `rpart` (CART) | 🔶 | RF uses trees internally |
| `ROCR` | 🔶 | Binary classification exists but not ROC |
| `caret` (tuning) | 🔶 | Cross-validation via bootstrap |
| `gbm/xgboost` | 🔶 | Random forest, but not boosting |

---

## TOP 25 METHODS TO IMPLEMENT (by priority score)

| Rank | Score | Method | Category | Complexity | R Package |
|------|-------|--------|----------|------------|-----------|
| 1 | 95 | **glmnet** | Regularization | Medium | glmnet |
| 2 | 93 | **xgboost** | Boosting | Complex | xgboost |
| 3 | 91 | **gbm** | Boosting | Medium | gbm |
| 4 | 90 | **lightgbm** | Boosting | Complex | lightgbm |
| 5 | 88 | **elastic_net** | Regularization | Medium | elasticnet |
| 6 | 87 | **lasso** | Regularization | Medium | lars/glmnet |
| 7 | 86 | **ridge** | Regularization | Simple | glmnet |
| 8 | 85 | **rpart** | Tree Methods | Medium | rpart |
| 9 | 84 | **kernel_svm** | SVM/Kernel | Medium | kernlab |
| 10 | 83 | **adaboost** | Boosting | Medium | adabag |
| 11 | 82 | **mboost** | Boosting | Complex | mboost |
| 12 | 81 | **C5.0** | Tree Methods | Medium | C50 |
| 13 | 80 | **cubist** | Tree Methods | Medium | Cubist |
| 14 | 79 | **quantile_rf** | Ensemble | Medium | quantregForest |
| 15 | 78 | **earth_mars** | Regularization | Medium | earth |
| 16 | 77 | **grf_full** | Causal ML | Medium | grf |
| 17 | 76 | **bart** | Bayesian | Complex | BART |
| 18 | 75 | **ctree** | Tree Methods | Medium | partykit |
| 19 | 74 | **shap_values** | XAI | Medium | fastshap |
| 20 | 73 | **partial_dependence** | XAI | Simple | pdp |
| 21 | 72 | **ice_curves** | XAI | Simple | ICEbox |
| 22 | 71 | **lime** | XAI | Medium | lime |
| 23 | 70 | **variable_importance** | Feature Selection | Simple | varSelRF |
| 24 | 69 | **boruta** | Feature Selection | Medium | Boruta |
| 25 | 68 | **apriori** | Association Rules | Medium | arules |

---

## QUICK WINS (High priority + Simple complexity)

These can be implemented quickly with high impact:

### 1. Ridge Regression
- **Score:** 86
- **Complexity:** Simple
- **Description:** L2-penalized linear regression with closed-form solution
- **R Package:** `glmnet`, `MASS::lm.ridge`
- **Doc:** https://glmnet.stanford.edu/articles/glmnet.html
- **Implementation Notes:**
  - Closed-form: β = (X'X + λI)⁻¹X'y
  - Already have matrix operations in `linalg/matrix_ops.rs`
  - Cross-validation for λ selection
- **Estimated effort:** 2-3 hours
- **Command:** `/implement_metrics https://glmnet.stanford.edu/reference/glmnet.html`

### 2. Partial Dependence Plots
- **Score:** 73
- **Complexity:** Simple
- **Description:** Marginal effect of features on predictions
- **R Package:** `pdp`, `gbm`
- **Doc:** https://christophm.github.io/interpretable-ml-book/pdp.html
- **Implementation Notes:**
  - Grid-based averaging of predictions
  - Works with any predict function
  - Already have random_forest with predict
- **Estimated effort:** 2-3 hours
- **Command:** `/implement_metrics https://www.rdocumentation.org/packages/pdp/versions/0.8.1/topics/partial`

### 3. Variable Importance (Permutation)
- **Score:** 70
- **Complexity:** Simple
- **Description:** Feature importance via permutation
- **R Package:** `varSelRF`, `randomForest`
- **Doc:** https://christophm.github.io/interpretable-ml-book/feature-importance.html
- **Implementation Notes:**
  - Permute feature, measure accuracy drop
  - Already have random_forest
- **Estimated effort:** 1-2 hours

### 4. ICE Curves
- **Score:** 72
- **Complexity:** Simple
- **Description:** Individual Conditional Expectation curves
- **R Package:** `ICEbox`, `pdp`
- **Doc:** https://christophm.github.io/interpretable-ml-book/ice.html
- **Implementation Notes:**
  - Extension of PDP for individual observations
  - Reuses PDP infrastructure
- **Estimated effort:** 1-2 hours

### 5. ROC/AUC Metrics
- **Score:** 68
- **Complexity:** Simple
- **Description:** ROC curves and AUC for binary classification
- **R Package:** `ROCR`, `pROC`
- **Doc:** https://www.rdocumentation.org/packages/ROCR/versions/1.0-11/topics/performance
- **Implementation Notes:**
  - Threshold-based TPR/FPR calculation
  - Trapezoidal AUC integration
- **Estimated effort:** 2-3 hours

---

## DETAILED GAP ANALYSIS (Top 10)

### 1. GLMNET (Elastic Net / Lasso / Ridge)

**Description:** Regularized generalized linear models with elastic net penalty (alpha * L1 + (1-alpha) * L2).

**Category:** Regularization
**Priority Score:** 95/100
**Complexity:** Medium
**Documentation:** https://glmnet.stanford.edu/

**Why Important:**
- Most widely used regularization method in ML
- Handles high-dimensional data (p >> n)
- Automatic feature selection (lasso)
- Essential for modern econometrics

**Existing Related Components:**
- `run_ols` in `regression/ols.rs` - base regression
- `safe_inverse` in `linalg/matrix_ops.rs` - matrix ops
- `run_step` in `regression/step.rs` - model selection

**Implementation Notes:**
- Core algorithm: Coordinate descent
- Need pathwise solution for λ sequence
- Cross-validation for hyperparameter selection
- Support: gaussian, binomial, poisson families

**Estimated effort:** 8-12 hours

---

### 2. XGBoost (Extreme Gradient Boosting)

**Description:** Scalable gradient boosting with regularization, handling missing values, and parallel computation.

**Category:** Boosting
**Priority Score:** 93/100
**Complexity:** Complex
**Documentation:** https://xgboost.readthedocs.io/

**Why Important:**
- State-of-the-art for tabular data
- Kaggle competition winner
- Handles missing values natively
- Highly optimized

**Existing Related Components:**
- `random_forest` in `ml/trees.rs` - tree building
- Decision stump infrastructure for splits

**Implementation Notes:**
- Newton-Raphson approximation to loss
- Regularized objective: loss + Ω(tree)
- Histogram-based split finding for speed
- Consider: integrate via FFI or pure Rust

**Estimated effort:** 20-30 hours (pure Rust) or 5 hours (FFI wrapper)

---

### 3. GBM (Gradient Boosting Machine)

**Description:** Classic gradient boosting for regression and classification with tree base learners.

**Category:** Boosting
**Priority Score:** 91/100
**Complexity:** Medium
**Documentation:** https://cran.r-project.org/web/packages/gbm/gbm.pdf

**Why Important:**
- Foundation for modern boosting methods
- Interpretable with partial dependence
- Good baseline before XGBoost

**Existing Related Components:**
- `random_forest` - tree infrastructure
- Residual fitting from OLS

**Implementation Notes:**
- Sequential tree fitting to pseudo-residuals
- Shrinkage (learning rate) parameter
- Interaction depth control
- Stochastic gradient boosting (subsampling)

**Estimated effort:** 10-15 hours

---

### 4. CART Trees (rpart)

**Description:** Classification and Regression Trees with pruning.

**Category:** Tree Methods
**Priority Score:** 85/100
**Complexity:** Medium
**Documentation:** https://cran.r-project.org/web/packages/rpart/rpart.pdf

**Why Important:**
- Foundation for all tree-based methods
- Highly interpretable
- Needed for proper tree visualization

**Existing Related Components:**
- `random_forest` uses internal CART
- Split finding infrastructure

**Implementation Notes:**
- Expose single tree building
- Cost-complexity pruning (cp parameter)
- Surrogate splits for missing data
- Visualization/print methods

**Estimated effort:** 6-8 hours (expose existing + pruning)

---

### 5. Kernel SVM

**Description:** Support Vector Machines with RBF, polynomial, and sigmoid kernels.

**Category:** SVM/Kernel
**Priority Score:** 84/100
**Complexity:** Medium
**Documentation:** https://www.rdocumentation.org/packages/kernlab/topics/ksvm

**Why Important:**
- Classic ML method
- Effective for small-medium datasets
- Strong theoretical foundations

**Existing Related Components:**
- `linear_svm` in `ml/svm.rs` - SMO algorithm
- Kernel evaluation patterns

**Implementation Notes:**
- Extend SMO for kernel matrix
- Add RBF: K(x,y) = exp(-γ||x-y||²)
- Polynomial: K(x,y) = (scale*x·y + coef0)^degree
- Kernel caching for efficiency

**Estimated effort:** 6-8 hours

---

### 6. AdaBoost

**Description:** Adaptive Boosting for classification with exponential loss.

**Category:** Boosting
**Priority Score:** 83/100
**Complexity:** Medium
**Documentation:** https://cran.r-project.org/web/packages/adabag/adabag.pdf

**Why Important:**
- Classic boosting algorithm
- Foundation for understanding boosting
- Simple but effective

**Existing Related Components:**
- `random_forest` - tree building
- DecisionStump infrastructure

**Implementation Notes:**
- Weight update: w_i *= exp(-α_t * y_i * h_t(x_i))
- Classifier weight: α_t = 0.5 * log((1-ε)/ε)
- SAMME variant for multiclass

**Estimated effort:** 5-7 hours

---

### 7. EARTH/MARS (Multivariate Adaptive Regression Splines)

**Description:** Piecewise linear regression with automatic knot selection.

**Category:** Regularization
**Priority Score:** 78/100
**Complexity:** Medium
**Documentation:** https://cran.r-project.org/web/packages/earth/earth.pdf

**Why Important:**
- Non-linear but interpretable
- Automatic interaction detection
- Handles categorical variables

**Existing Related Components:**
- `smooth_spline` in `regression/smooth_spline.rs`
- GCV for model selection

**Implementation Notes:**
- Forward pass: add basis functions
- Backward pass: prune via GCV
- Hinge functions: max(0, x-c), max(0, c-x)
- Interaction terms

**Estimated effort:** 8-10 hours

---

### 8. C5.0 Trees

**Description:** Successor to C4.5 with boosting, rule-based models, and cost-sensitive learning.

**Category:** Tree Methods
**Priority Score:** 81/100
**Complexity:** Medium
**Documentation:** https://cran.r-project.org/web/packages/C50/C50.pdf

**Why Important:**
- Better than CART for classification
- Rule extraction capability
- Handles costs and boosting

**Implementation Notes:**
- Information gain ratio for splits
- Rule-based mode extracts rules from trees
- Boosting variant (AdaBoost style)

**Estimated effort:** 10-12 hours

---

### 9. SHAP Values

**Description:** SHapley Additive exPlanations for model interpretation.

**Category:** XAI
**Priority Score:** 74/100
**Complexity:** Medium
**Documentation:** https://christophm.github.io/interpretable-ml-book/shap.html

**Why Important:**
- Theoretically grounded explanations
- Unifies feature importance methods
- Local + global interpretation

**Existing Related Components:**
- Prediction functions for all models
- Coalitional value computation

**Implementation Notes:**
- KernelSHAP for model-agnostic
- TreeSHAP optimization for tree models
- Need: feature sampling, coalition generation

**Estimated effort:** 12-15 hours

---

### 10. Boruta Feature Selection

**Description:** Wrapper feature selection using random forest shadow variables.

**Category:** Feature Selection
**Priority Score:** 69/100
**Complexity:** Medium
**Documentation:** https://cran.r-project.org/web/packages/Boruta/Boruta.pdf

**Why Important:**
- Robust feature selection
- Statistical testing of importance
- Works well with RF

**Existing Related Components:**
- `random_forest` - importance scores
- Bootstrap/permutation infrastructure

**Implementation Notes:**
- Create shadow features (permuted copies)
- Compare real vs shadow importance
- Statistical test for significance

**Estimated effort:** 4-6 hours

---

## METHODS SKIPPED (Out of Scope)

| Method | Reason |
|--------|--------|
| TensorFlow/PyTorch interfaces | External deep learning frameworks |
| H2O platform | External platform |
| mlr3/caret/tidymodels | Meta-frameworks, not algorithms |
| Fuzzy/Rough set methods | Niche application |
| Genetic algorithms | Optimization, not ML |
| RSNNS neural networks | Deep learning |

---

## IMPLEMENTATION ROADMAP

### Phase 1: Regularization Foundation (Week 1)
1. Ridge regression (simple, closed-form)
2. Lasso via coordinate descent
3. Elastic net (combines above)
4. glmnet-style path fitting

### Phase 2: Boosting Methods (Week 2)
1. GBM (gradient boosting)
2. AdaBoost
3. XGBoost (or FFI wrapper)

### Phase 3: Explainability (Week 3)
1. Partial dependence plots
2. ICE curves
3. Permutation importance
4. SHAP values (basic KernelSHAP)

### Phase 4: Tree Extensions (Week 4)
1. CART with pruning (expose internals)
2. C5.0 or ctree
3. Kernel SVM extension

---

## NEXT STEPS

What would you like to do?

1. **Implement top priority method: glmnet**
   → `/implement_metrics https://glmnet.stanford.edu/reference/glmnet.html`

2. **Implement Quick Win: Ridge Regression**
   → `/implement_metrics https://glmnet.stanford.edu/reference/glmnet.html` (alpha=0)

3. **Implement all Quick Wins** (5 methods)
   → Sequential implementation of simple methods

4. **Export full list to CSV for review**
   → Creates `docs/discovery/cran-ml-methods.csv`

5. **Focus on specific category**
   → Re-run analysis for Boosting only
