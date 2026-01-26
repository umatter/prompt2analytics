# Citation Guide for p2a-core

This document provides guidance on which references to cite in academic papers
describing or using p2a-core, versus which references are for code documentation only.

## Citation Strategy Overview

| Tier | When to Cite | Example |
|------|--------------|---------|
| **Tier 1** | Always cite in any paper | R Core Team, Polars |
| **Tier 2** | Cite when discussing that method category | White (1980) for robust SEs |
| **Tier 3** | Cite only if discussing implementation details | Durbin-Levinson algorithm |
| **Tier 4** | Code documentation only, never in paper | Historical origins |

---

## Tier 1: Core Software References (Always Cite)

These should appear in every paper about p2a-core:

### R Language and Stats Package
```bibtex
@Manual{R2024,
  title = {R: A Language and Environment for Statistical Computing},
  author = {{R Core Team}},
  organization = {R Foundation for Statistical Computing},
  address = {Vienna, Austria},
  year = {2024},
  url = {https://www.R-project.org/},
}
```

**Note**: The R stats package is part of base R and cited via the R Core Team reference.
Mention in text: "...replicating functionality from R's stats package (R Core Team, 2024)..."

### Polars (Data Backend)
```bibtex
@software{polars2024,
  author = {Ritchie Vink and contributors},
  title = {Polars: Fast DataFrame Library},
  year = {2024},
  url = {https://pola.rs/},
}
```

### Rust Numerical Libraries (Optional)
Only cite if discussing implementation details:
```bibtex
@software{ndarray2024,
  author = {bluss and ndarray contributors},
  title = {ndarray: An N-dimensional array for general elements and for numerics},
  year = {2024},
  url = {https://github.com/rust-ndarray/ndarray},
}
```

---

## Tier 2: Seminal Papers by Method Category

Cite **one** seminal paper per method category that your paper discusses or benchmarks.

### Regression & Standard Errors

**Robust Standard Errors** → Cite if using HC0-HC3 or discussing heteroskedasticity:
```bibtex
@article{White1980,
  author = {White, Halbert},
  title = {A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity},
  journal = {Econometrica},
  volume = {48},
  number = {4},
  pages = {817--838},
  year = {1980},
  doi = {10.2307/1912934},
}
```

**Clustered Standard Errors** → Cite if using clustered SEs:
```bibtex
@article{CameronMiller2015,
  author = {Cameron, A. Colin and Miller, Douglas L.},
  title = {A Practitioner's Guide to Cluster-Robust Inference},
  journal = {Journal of Human Resources},
  volume = {50},
  number = {2},
  pages = {317--372},
  year = {2015},
  doi = {10.3368/jhr.50.2.317},
}
```

### Panel Data Econometrics

**Fixed/Random Effects** → Cite one of:
```bibtex
@article{Mundlak1978,
  author = {Mundlak, Yair},
  title = {On the Pooling of Time Series and Cross Section Data},
  journal = {Econometrica},
  volume = {46},
  number = {1},
  pages = {69--85},
  year = {1978},
  doi = {10.2307/1913646},
}

@book{Baltagi2013,
  author = {Baltagi, Badi H.},
  title = {Econometric Analysis of Panel Data},
  edition = {5th},
  publisher = {Wiley},
  year = {2013},
  isbn = {978-1118672327},
}
```

**Hausman Test** → Cite if discussing FE vs RE selection:
```bibtex
@article{Hausman1978,
  author = {Hausman, Jerry A.},
  title = {Specification Tests in Econometrics},
  journal = {Econometrica},
  volume = {46},
  number = {6},
  pages = {1251--1271},
  year = {1978},
  doi = {10.2307/1913827},
}
```

### Instrumental Variables

**2SLS/IV** → Cite for any IV discussion:
```bibtex
@book{AngristPischke2009,
  author = {Angrist, Joshua D. and Pischke, Jörn-Steffen},
  title = {Mostly Harmless Econometrics: An Empiricist's Companion},
  publisher = {Princeton University Press},
  year = {2009},
  isbn = {978-0691120355},
}
```

**Weak Instruments** → Cite if discussing first-stage F-statistics:
```bibtex
@incollection{StockYogo2005,
  author = {Stock, James H. and Yogo, Motohiro},
  title = {Testing for Weak Instruments in Linear IV Regression},
  booktitle = {Identification and Inference for Econometric Models},
  editor = {Andrews, Donald W.K. and Stock, James H.},
  publisher = {Cambridge University Press},
  pages = {80--108},
  year = {2005},
  doi = {10.1017/CBO9780511614491.006},
}
```

### Causal Inference

**Difference-in-Differences** → Cite for DiD:
```bibtex
@article{CardKrueger1994,
  author = {Card, David and Krueger, Alan B.},
  title = {Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania},
  journal = {American Economic Review},
  volume = {84},
  number = {4},
  pages = {772--793},
  year = {1994},
}
```

Or for modern DiD developments:
```bibtex
@article{Roth2023,
  author = {Roth, Jonathan and Sant'Anna, Pedro H.C. and Bilinski, Alyssa and Poe, John},
  title = {What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature},
  journal = {Journal of Econometrics},
  volume = {235},
  number = {2},
  pages = {2218--2244},
  year = {2023},
  doi = {10.1016/j.jeconom.2023.03.008},
}
```

### Discrete Choice Models

**Logit/Probit** → Cite for discrete choice:
```bibtex
@incollection{McFadden1974,
  author = {McFadden, Daniel},
  title = {Conditional Logit Analysis of Qualitative Choice Behavior},
  booktitle = {Frontiers in Econometrics},
  editor = {Zarembka, Paul},
  publisher = {Academic Press},
  pages = {105--142},
  year = {1974},
}
```

### Time Series

**ARIMA** → Cite for any ARIMA/Box-Jenkins discussion:
```bibtex
@book{BoxJenkins2015,
  author = {Box, George E.P. and Jenkins, Gwilym M. and Reinsel, Gregory C. and Ljung, Greta M.},
  title = {Time Series Analysis: Forecasting and Control},
  edition = {5th},
  publisher = {Wiley},
  year = {2015},
  isbn = {978-1118675021},
}
```

**VAR Models** → Cite for multivariate time series:
```bibtex
@article{Sims1980,
  author = {Sims, Christopher A.},
  title = {Macroeconomics and Reality},
  journal = {Econometrica},
  volume = {48},
  number = {1},
  pages = {1--48},
  year = {1980},
  doi = {10.2307/1912017},
}
```

**Cointegration/VECM** → Cite for error correction models:
```bibtex
@article{EngleGranger1987,
  author = {Engle, Robert F. and Granger, Clive W.J.},
  title = {Co-Integration and Error Correction: Representation, Estimation, and Testing},
  journal = {Econometrica},
  volume = {55},
  number = {2},
  pages = {251--276},
  year = {1987},
  doi = {10.2307/1913236},
}
```

### Hypothesis Testing

**General textbook** → For broad coverage of statistical tests:
```bibtex
@book{Wooldridge2010,
  author = {Wooldridge, Jeffrey M.},
  title = {Econometric Analysis of Cross Section and Panel Data},
  edition = {2nd},
  publisher = {MIT Press},
  year = {2010},
  isbn = {978-0262232586},
}
```

### Regression Diagnostics

**Heteroskedasticity test** → Cite if discussing BP test:
```bibtex
@article{BreuschPagan1979,
  author = {Breusch, Trevor S. and Pagan, Adrian R.},
  title = {A Simple Test for Heteroscedasticity and Random Coefficient Variation},
  journal = {Econometrica},
  volume = {47},
  number = {5},
  pages = {1287--1294},
  year = {1979},
  doi = {10.2307/1911963},
}
```

**Normality test** → Cite if discussing JB test:
```bibtex
@article{JarqueBera1980,
  author = {Jarque, Carlos M. and Bera, Anil K.},
  title = {Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals},
  journal = {Economics Letters},
  volume = {6},
  number = {3},
  pages = {255--259},
  year = {1980},
  doi = {10.1016/0165-1765(80)90024-5},
}
```

### Smoothing & Nonparametric Methods

**Projection Pursuit Regression** → Cite if discussing PPR:
```bibtex
@article{FriedmanStuetzle1981,
  author = {Friedman, Jerome H. and Stuetzle, Werner},
  title = {Projection Pursuit Regression},
  journal = {Journal of the American Statistical Association},
  volume = {76},
  number = {376},
  pages = {817--823},
  year = {1981},
  doi = {10.1080/01621459.1981.10477729},
}
```

**LOESS/LOWESS** → Cite if discussing local regression:
```bibtex
@article{Cleveland1979,
  author = {Cleveland, William S.},
  title = {Robust Locally Weighted Regression and Smoothing Scatterplots},
  journal = {Journal of the American Statistical Association},
  volume = {74},
  number = {368},
  pages = {829--836},
  year = {1979},
  doi = {10.1080/01621459.1979.10481038},
}
```

---

## Tier 3: Implementation-Specific References

Only cite these if your paper discusses **how** we implemented the algorithm:

### Algorithms

**Durbin-Levinson** (for ACF to AR conversion):
```bibtex
@article{Durbin1960,
  author = {Durbin, James},
  title = {The Fitting of Time-Series Models},
  journal = {Revue de l'Institut International de Statistique},
  volume = {28},
  number = {3},
  pages = {233--244},
  year = {1960},
  doi = {10.2307/1401322},
}
```

**SuperSmoother algorithm**:
```bibtex
@techreport{Friedman1984,
  author = {Friedman, Jerome H.},
  title = {A Variable Span Smoother},
  institution = {Laboratory for Computational Statistics, Stanford University},
  number = {Technical Report No. 5},
  year = {1984},
}
```

**Barrier methods** (for constrained optimization):
```bibtex
@book{FiaccoMcCormick1968,
  author = {Fiacco, Anthony V. and McCormick, Garth P.},
  title = {Nonlinear Programming: Sequential Unconstrained Minimization Techniques},
  publisher = {Wiley},
  year = {1968},
  note = {Reprinted by SIAM, 1990},
}
```

---

## Tier 4: Code Documentation Only (Do Not Cite in Papers)

These references provide historical context or multiple perspectives in the code
documentation but should **not** be cited in papers:

### Historical Origins
- Gauss (1821) - Original least squares derivation
- Bliss (1934) - Original probit formulation
- Berkson (1944) - Introduction of logit
- Wright (1928) - First IV application
- Fisher (1925) - Original ANOVA

### Duplicate Coverage (use seminal paper instead)
- Multiple textbooks covering the same method
- Review papers when original exists
- Software documentation (cite R Core Team instead)

### Supplementary Textbooks
These are valuable for users learning the methods but don't need paper citations:
- Brockwell & Davis (1991) - Time series textbook
- Hamilton (1994) - Time series textbook
- Train (2009) - Discrete choice textbook
- Hastie, Tibshirani & Friedman (2009) - ESL textbook

---

## Quick Reference: Minimum Citation Set

For a **general paper** introducing p2a-core, cite at minimum:

1. **R Core Team (2024)** - The R language and stats package
2. **Polars** - Data backend
3. **Wooldridge (2010)** - General econometrics reference
4. **Box & Jenkins (2015)** - Time series methods

For a **methods comparison paper**, add the seminal paper for each method benchmarked.

For an **implementation paper**, add Tier 3 references for algorithms discussed.

---

## Example Paper Citation Section

> p2a-core is a pure Rust implementation of statistical and econometric methods,
> designed to replicate the functionality of R's stats package (R Core Team, 2024).
> The library implements ordinary least squares with heteroskedasticity-robust
> standard errors (White, 1980), panel data estimators including fixed and random
> effects (Baltagi, 2013), instrumental variables estimation (Angrist & Pischke, 2009),
> and ARIMA time series models (Box et al., 2015). Data handling uses the Polars
> DataFrame library (Vink et al., 2024).

---

## Updating This Document

When adding new methods to p2a-core:

1. Add full references to the relevant source file (all tiers)
2. Add Tier 2 seminal reference to this document if it's a new method category
3. Add Tier 3 reference only if using a specific published algorithm

Last updated: 2026-01-22
