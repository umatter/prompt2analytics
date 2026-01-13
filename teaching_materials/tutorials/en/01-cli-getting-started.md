# Getting Started with the p2a Command-Line Interface

## Learning Objectives

After completing this tutorial, you will be able to:
- [ ] Install and run the p2a CLI
- [ ] Load datasets from various file formats
- [ ] Perform basic data exploration
- [ ] Run simple regression analyses
- [ ] Export results for reports

## Prerequisites

- p2a CLI installed (see Installation section)
- Terminal/command prompt access
- Sample datasets from `teaching_data/`

## Installation

<!-- TODO: Add installation instructions -->

## Section 1: Your First Commands

### Loading a Dataset

<!-- TODO: Add step-by-step instructions -->

```bash
p2a data load path/to/data.csv --name mydata
```

### Viewing Data

<!-- TODO: Add examples -->

```bash
p2a data describe mydata
p2a data head mydata
```

## Section 2: Basic Statistics

<!-- TODO: Add correlation, summary statistics examples -->

## Section 3: Running Regressions

<!-- TODO: Add OLS regression examples -->

```bash
p2a reg ols mydata -y dependent_var -x independent_var1 independent_var2
```

## Section 4: Creating Visualizations

<!-- TODO: Add visualization examples -->

## Section 5: Session Management

<!-- TODO: Add session recording and script export -->

## Practice Exercises

1. <!-- TODO: Exercise 1 -->
2. <!-- TODO: Exercise 2 -->
3. <!-- TODO: Exercise 3 -->

## Summary

<!-- TODO: Add key points summary -->

## Next Steps

- Continue with [Chat Interface Guide](02-chat-interface-guide.md) for the web-based approach
- Try the [Data Analysis Workflow](03-data-analysis-workflow.md) for a complete example
