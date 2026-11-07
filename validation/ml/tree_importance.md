# Impurity importance of the tree ensembles

## Definition

Each accepted split adds its SSE decrease, on the node's rows with their bootstrap multiplicity, to the feature it
splits on. Case file: `validation/r/cases/ml_tree_importance.R`. Registry: `validation/methods/ml_cart_cp.json`.
