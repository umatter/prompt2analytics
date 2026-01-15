/**
 * MCP Tool to CLI Command Mapping
 *
 * Maps MCP tool names and their arguments to p2a CLI command syntax.
 */

export interface CliCommand {
  category: string
  subcommand: string
  args: string[]
}

type ArgMapper = (args: Record<string, unknown>) => string[]

interface ToolMapping {
  category: string
  subcommand: string
  argMapper: ArgMapper
}

// Helper to safely get string value
function getString(args: Record<string, unknown>, key: string): string | undefined {
  const val = args[key]
  return typeof val === 'string' ? val : undefined
}

// Helper to safely get string array
function getStringArray(args: Record<string, unknown>, key: string): string[] {
  const val = args[key]
  if (Array.isArray(val)) {
    return val.filter((v): v is string => typeof v === 'string')
  }
  return []
}

// Helper to safely get number
function getNumber(args: Record<string, unknown>, key: string): number | undefined {
  const val = args[key]
  return typeof val === 'number' ? val : undefined
}

// Helper to safely get boolean
function getBool(args: Record<string, unknown>, key: string): boolean | undefined {
  const val = args[key]
  return typeof val === 'boolean' ? val : undefined
}

// Quote path if it contains spaces
function quotePath(path: string): string {
  return path.includes(' ') ? `"${path}"` : path
}

/**
 * Tool mappings from MCP tool name to CLI command structure
 */
const toolMappings: Record<string, ToolMapping> = {
  // === Data Loading ===
  load_dataset: {
    category: 'data',
    subcommand: 'load',
    argMapper: (args) => {
      const result: string[] = []
      const path = getString(args, 'path')
      if (path) result.push(quotePath(path))
      const name = getString(args, 'name')
      if (name) result.push('--name', name)
      return result
    },
  },

  // === Regression ===
  regression_ols: {
    category: 'reg',
    subcommand: 'ols',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      // MCP hardcodes these, but we include them explicitly
      result.push('--intercept', 'true')
      result.push('--robust', 'hc1')
      return result
    },
  },

  regression_clustered: {
    category: 'reg',
    subcommand: 'clustered',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      const cluster1 = getString(args, 'cluster1')
      if (cluster1) result.push('--cluster', cluster1)
      const cluster2 = getString(args, 'cluster2')
      if (cluster2) result.push('--cluster2', cluster2)
      return result
    },
  },

  regression_diagnostics: {
    category: 'reg',
    subcommand: 'diagnostics',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      return result
    },
  },

  // === Panel Data ===
  panel_fixed_effects: {
    category: 'panel',
    subcommand: 'fe',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      const entity = getString(args, 'entity_var')
      if (entity) result.push('--entity', entity)
      return result
    },
  },

  panel_random_effects: {
    category: 'panel',
    subcommand: 're',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      const entity = getString(args, 'entity_var')
      if (entity) result.push('--entity', entity)
      return result
    },
  },

  hausman_test: {
    category: 'panel',
    subcommand: 'hausman',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      const entity = getString(args, 'entity_var')
      if (entity) result.push('--entity', entity)
      return result
    },
  },

  panel_hdfe: {
    category: 'panel',
    subcommand: 'hdfe',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      const fe = getStringArray(args, 'fe')
      if (fe.length > 0) result.push('--fe', ...fe)
      const seType = getString(args, 'se_type')
      if (seType) result.push('--robust', seType)
      return result
    },
  },

  panel_feglm: {
    category: 'panel',
    subcommand: 'feglm',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      const fe = getStringArray(args, 'fe')
      if (fe.length > 0) result.push('--fe', ...fe)
      const family = getString(args, 'family')
      if (family) result.push('--family', family)
      return result
    },
  },

  // === Survival Analysis ===
  survival_km: {
    category: 'survival',
    subcommand: 'km',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const time = getString(args, 'time_col')
      if (time) result.push('--time', time)
      const event = getString(args, 'event_col')
      if (event) result.push('--event', event)
      const group = getString(args, 'group_col')
      if (group) result.push('--group', group)
      return result
    },
  },

  survival_cox: {
    category: 'survival',
    subcommand: 'cox',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const time = getString(args, 'time_col')
      if (time) result.push('--time', time)
      const event = getString(args, 'event_col')
      if (event) result.push('--event', event)
      const x = getStringArray(args, 'covariates')
      if (x.length > 0) result.push('-x', ...x)
      return result
    },
  },

  survival_aft: {
    category: 'survival',
    subcommand: 'aft',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const time = getString(args, 'time_col')
      if (time) result.push('--time', time)
      const event = getString(args, 'event_col')
      if (event) result.push('--event', event)
      const x = getStringArray(args, 'covariates')
      if (x.length > 0) result.push('-x', ...x)
      const dist = getString(args, 'distribution')
      if (dist) result.push('--dist', dist)
      return result
    },
  },

  survival_competing_risks: {
    category: 'survival',
    subcommand: 'competing-risks',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const time = getString(args, 'time_col')
      if (time) result.push('--time', time)
      const event = getString(args, 'event_col')
      if (event) result.push('--event', event)
      const eventType = getNumber(args, 'event_type')
      if (eventType !== undefined) result.push('--event-type', eventType.toString())
      return result
    },
  },

  survival_log_rank: {
    category: 'survival',
    subcommand: 'log-rank',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const time = getString(args, 'time_col')
      if (time) result.push('--time', time)
      const event = getString(args, 'event_col')
      if (event) result.push('--event', event)
      const group = getString(args, 'group_col')
      if (group) result.push('--group', group)
      return result
    },
  },

  // === Causal Inference ===
  iv_2sls: {
    category: 'causal',
    subcommand: 'iv',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const exog = getStringArray(args, 'x_exog')
      if (exog.length > 0) result.push('--exog', ...exog)
      const endog = getStringArray(args, 'x_endog')
      if (endog.length > 0) result.push('--endog', ...endog)
      const instruments = getStringArray(args, 'instruments')
      if (instruments.length > 0) result.push('--instruments', ...instruments)
      return result
    },
  },

  iv_first_stage: {
    category: 'causal',
    subcommand: 'iv-first-stage',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const endog = getString(args, 'endogenous')
      if (endog) result.push('--endog', endog)
      const exog = getStringArray(args, 'exogenous')
      if (exog.length > 0) result.push('--exog', ...exog)
      const instruments = getStringArray(args, 'instruments')
      if (instruments.length > 0) result.push('--instruments', ...instruments)
      return result
    },
  },

  diff_in_diff: {
    category: 'causal',
    subcommand: 'did',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'dep_var')
      if (y) result.push('-y', y)
      const treat = getString(args, 'treatment_var')
      if (treat) result.push('--treat', treat)
      const post = getString(args, 'post_var')
      if (post) result.push('--post', post)
      return result
    },
  },

  rd_estimate: {
    category: 'causal',
    subcommand: 'rd',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const running = getString(args, 'running_var')
      if (running) result.push('--running', running)
      const cutoff = getNumber(args, 'cutoff')
      if (cutoff !== undefined) result.push('--cutoff', cutoff.toString())
      return result
    },
  },

  fuzzy_rd: {
    category: 'causal',
    subcommand: 'fuzzy-rd',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const running = getString(args, 'running_var')
      if (running) result.push('--running', running)
      const treatment = getString(args, 'treatment_var')
      if (treatment) result.push('--treatment', treatment)
      const cutoff = getNumber(args, 'cutoff')
      if (cutoff !== undefined) result.push('--cutoff', cutoff.toString())
      return result
    },
  },

  ipw_estimate: {
    category: 'causal',
    subcommand: 'ipw',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const t = getString(args, 'treatment')
      if (t) result.push('-t', t)
      const x = getStringArray(args, 'covariates')
      if (x.length > 0) result.push('-x', ...x)
      return result
    },
  },

  doubly_robust: {
    category: 'causal',
    subcommand: 'doubly-robust',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const t = getString(args, 'treatment')
      if (t) result.push('-t', t)
      const x = getStringArray(args, 'covariates')
      if (x.length > 0) result.push('-x', ...x)
      const method = getString(args, 'method')
      if (method) result.push('--method', method)
      return result
    },
  },

  mediation_analysis: {
    category: 'causal',
    subcommand: 'mediation',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const t = getString(args, 'treatment')
      if (t) result.push('-t', t)
      const m = getString(args, 'mediator')
      if (m) result.push('-m', m)
      const x = getStringArray(args, 'covariates')
      if (x.length > 0) result.push('-x', ...x)
      return result
    },
  },

  // === Discrete Choice ===
  logit: {
    category: 'discrete',
    subcommand: 'logit',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      return result
    },
  },

  probit: {
    category: 'discrete',
    subcommand: 'probit',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      return result
    },
  },

  // === Time Series ===
  ts_var: {
    category: 'ts',
    subcommand: 'var',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const lags = getNumber(args, 'lags')
      if (lags !== undefined) result.push('--lags', lags.toString())
      return result
    },
  },

  ts_arima_fit: {
    category: 'ts',
    subcommand: 'arima',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const p = getNumber(args, 'p')
      if (p !== undefined) result.push('-p', p.toString())
      const d = getNumber(args, 'd')
      if (d !== undefined) result.push('-d', d.toString())
      const q = getNumber(args, 'q')
      if (q !== undefined) result.push('-q', q.toString())
      return result
    },
  },

  ts_arima_forecast: {
    category: 'ts',
    subcommand: 'arima',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const p = getNumber(args, 'p')
      if (p !== undefined) result.push('-p', p.toString())
      const d = getNumber(args, 'd')
      if (d !== undefined) result.push('-d', d.toString())
      const q = getNumber(args, 'q')
      if (q !== undefined) result.push('-q', q.toString())
      const horizon = getNumber(args, 'horizon')
      if (horizon !== undefined) result.push('--horizon', horizon.toString())
      return result
    },
  },

  ts_mstl: {
    category: 'ts',
    subcommand: 'mstl',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const periods = getNumber(args, 'period')
      if (periods !== undefined) result.push('--period', periods.toString())
      return result
    },
  },

  ts_varma: {
    category: 'ts',
    subcommand: 'varma',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const p = getNumber(args, 'p')
      if (p !== undefined) result.push('-p', p.toString())
      const q = getNumber(args, 'q')
      if (q !== undefined) result.push('-q', q.toString())
      return result
    },
  },

  ts_vecm: {
    category: 'ts',
    subcommand: 'vecm',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const lags = getNumber(args, 'lags')
      if (lags !== undefined) result.push('--lags', lags.toString())
      const rank = getNumber(args, 'rank')
      if (rank !== undefined) result.push('--rank', rank.toString())
      return result
    },
  },

  ts_irf: {
    category: 'ts',
    subcommand: 'irf',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const lags = getNumber(args, 'lags')
      if (lags !== undefined) result.push('--lags', lags.toString())
      const steps = getNumber(args, 'steps')
      if (steps !== undefined) result.push('--steps', steps.toString())
      return result
    },
  },

  ts_changepoint: {
    category: 'ts',
    subcommand: 'changepoint',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const penalty = getNumber(args, 'penalty')
      if (penalty !== undefined) result.push('--penalty', penalty.toString())
      const changeType = getString(args, 'change_type')
      if (changeType) result.push('--change-type', changeType)
      return result
    },
  },

  // === Machine Learning ===
  ml_kmeans: {
    category: 'ml',
    subcommand: 'kmeans',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const k = getNumber(args, 'k')
      if (k !== undefined) result.push('-k', k.toString())
      return result
    },
  },

  ml_pca: {
    category: 'ml',
    subcommand: 'pca',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const n = getNumber(args, 'n_components')
      if (n !== undefined) result.push('-n', n.toString())
      return result
    },
  },

  ml_dbscan: {
    category: 'ml',
    subcommand: 'dbscan',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const eps = getNumber(args, 'eps')
      if (eps !== undefined) result.push('--eps', eps.toString())
      const minPts = getNumber(args, 'min_samples')
      if (minPts !== undefined) result.push('--min-pts', minPts.toString())
      return result
    },
  },

  ml_hierarchical: {
    category: 'ml',
    subcommand: 'hierarchical',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const k = getNumber(args, 'n_clusters')
      if (k !== undefined) result.push('-k', k.toString())
      const linkage = getString(args, 'linkage')
      if (linkage) result.push('--linkage', linkage)
      return result
    },
  },

  ml_random_forest: {
    category: 'ml',
    subcommand: 'random-forest',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'features')
      if (cols.length > 0) result.push('--cols', ...cols)
      const y = getString(args, 'target')
      if (y) result.push('-y', y)
      const nTrees = getNumber(args, 'n_trees')
      if (nTrees !== undefined) result.push('--n-trees', nTrees.toString())
      const maxDepth = getNumber(args, 'max_depth')
      if (maxDepth !== undefined) result.push('--max-depth', maxDepth.toString())
      return result
    },
  },

  ml_tsne: {
    category: 'ml',
    subcommand: 'tsne',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const nComponents = getNumber(args, 'n_components')
      if (nComponents !== undefined) result.push('-n', nComponents.toString())
      const perplexity = getNumber(args, 'perplexity')
      if (perplexity !== undefined) result.push('--perplexity', perplexity.toString())
      const seed = getNumber(args, 'seed')
      if (seed !== undefined) result.push('--seed', seed.toString())
      return result
    },
  },

  // === Visualization ===
  viz_histogram: {
    category: 'viz',
    subcommand: 'histogram',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const bins = getNumber(args, 'bins')
      if (bins !== undefined) result.push('--bins', bins.toString())
      // Add placeholder output path
      result.push('-f', './output_histogram.png')
      return result
    },
  },

  viz_scatter: {
    category: 'viz',
    subcommand: 'scatter',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const x = getString(args, 'x_column')
      if (x) result.push('--x', x)
      const y = getString(args, 'y_column')
      if (y) result.push('--y', y)
      result.push('-f', './output_scatter.png')
      return result
    },
  },

  viz_line: {
    category: 'viz',
    subcommand: 'line',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const x = getString(args, 'x_column')
      if (x) result.push('--x', x)
      const y = getStringArray(args, 'y_columns')
      if (y.length > 0) result.push('--y', ...y)
      result.push('-f', './output_line.png')
      return result
    },
  },

  viz_boxplot: {
    category: 'viz',
    subcommand: 'box',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const valueCol = getString(args, 'value_column')
      if (valueCol) result.push('-y', valueCol)
      const groupCol = getString(args, 'group_column')
      if (groupCol) result.push('-g', groupCol)
      result.push('-f', './output_boxplot.png')
      return result
    },
  },

  viz_heatmap: {
    category: 'viz',
    subcommand: 'heatmap',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      result.push('-f', './output_heatmap.png')
      return result
    },
  },

  viz_coefficient: {
    category: 'viz',
    subcommand: 'coefplot',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      const confLevel = getNumber(args, 'conf_level')
      if (confLevel !== undefined) result.push('--conf-level', confLevel.toString())
      result.push('-f', './output_coefplot.png')
      return result
    },
  },

  viz_residuals: {
    category: 'viz',
    subcommand: 'residuals',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const y = getString(args, 'y')
      if (y) result.push('-y', y)
      const x = getStringArray(args, 'x')
      if (x.length > 0) result.push('-x', ...x)
      result.push('-f', './output_residuals.png')
      return result
    },
  },

  viz_dendrogram: {
    category: 'viz',
    subcommand: 'dendrogram',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const linkage = getString(args, 'linkage')
      if (linkage) result.push('--linkage', linkage)
      result.push('-f', './output_dendrogram.png')
      return result
    },
  },

  viz_event_study: {
    category: 'viz',
    subcommand: 'event-study',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const timeCol = getString(args, 'time_col')
      if (timeCol) result.push('--time-col', timeCol)
      const estimateCol = getString(args, 'estimate_col')
      if (estimateCol) result.push('--estimate-col', estimateCol)
      const ciLowerCol = getString(args, 'ci_lower_col')
      if (ciLowerCol) result.push('--ci-lower-col', ciLowerCol)
      const ciUpperCol = getString(args, 'ci_upper_col')
      if (ciUpperCol) result.push('--ci-upper-col', ciUpperCol)
      result.push('-f', './output_event_study.png')
      return result
    },
  },

  viz_irf: {
    category: 'viz',
    subcommand: 'irf',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const horizonCol = getString(args, 'horizon_col')
      if (horizonCol) result.push('--horizon-col', horizonCol)
      const responseCol = getString(args, 'response_col')
      if (responseCol) result.push('--response-col', responseCol)
      const ciLowerCol = getString(args, 'ci_lower_col')
      if (ciLowerCol) result.push('--ci-lower-col', ciLowerCol)
      const ciUpperCol = getString(args, 'ci_upper_col')
      if (ciUpperCol) result.push('--ci-upper-col', ciUpperCol)
      const shockLabel = getString(args, 'shock_label')
      if (shockLabel) result.push('--shock-label', shockLabel)
      const responseLabel = getString(args, 'response_label')
      if (responseLabel) result.push('--response-label', responseLabel)
      result.push('-f', './output_irf.png')
      return result
    },
  },

  // === Data Munging ===
  munge_filter: {
    category: 'munge',
    subcommand: 'filter',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      const op = getString(args, 'operator')
      const val = args['value']
      if (col && op && val !== undefined) {
        result.push('--col', col)
        result.push('--op', op)
        result.push('--value', String(val))
      }
      const outputName = getString(args, 'output_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_select: {
    category: 'munge',
    subcommand: 'select',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const outputName = getString(args, 'output_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_rename: {
    category: 'munge',
    subcommand: 'rename',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      // Renaming is complex, add as placeholder
      result.push('# Rename mappings need manual adjustment')
      return result
    },
  },

  munge_sort: {
    category: 'munge',
    subcommand: 'sort',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--by', ...cols)
      const desc = getBool(args, 'descending')
      if (desc) result.push('--desc')
      return result
    },
  },

  munge_drop_columns: {
    category: 'munge',
    subcommand: 'drop',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_mutate: {
    category: 'munge',
    subcommand: 'mutate',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const newCol = getString(args, 'new_column')
      if (newCol) result.push('--new-col', newCol)
      const exprType = getString(args, 'expr_type')
      if (exprType) result.push('--expr-type', exprType)
      const left = getString(args, 'left')
      if (left) result.push('--left', left)
      const operator = getString(args, 'operator')
      if (operator) result.push('--op', operator)
      const right = getString(args, 'right')
      if (right) result.push('--right', right)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_sample: {
    category: 'munge',
    subcommand: 'sample',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const n = getNumber(args, 'n')
      if (n !== undefined) result.push('-n', n.toString())
      const replace = getBool(args, 'replace')
      if (replace) result.push('--replace')
      const seed = getNumber(args, 'seed')
      if (seed !== undefined) result.push('--seed', seed.toString())
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_join: {
    category: 'munge',
    subcommand: 'join',
    argMapper: (args) => {
      const result: string[] = []
      const left = getString(args, 'left')
      if (left) result.push('--left', left)
      const right = getString(args, 'right')
      if (right) result.push('--right', right)
      const leftOn = getStringArray(args, 'left_on')
      if (leftOn.length > 0) result.push('--left-on', ...leftOn)
      const rightOn = getStringArray(args, 'right_on')
      if (rightOn.length > 0) result.push('--right-on', ...rightOn)
      const how = getString(args, 'how')
      if (how) result.push('--how', how)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_concat: {
    category: 'munge',
    subcommand: 'concat',
    argMapper: (args) => {
      const result: string[] = []
      const datasets = getStringArray(args, 'datasets')
      if (datasets.length > 0) result.push('--datasets', ...datasets)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_group_by: {
    category: 'munge',
    subcommand: 'groupby',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const by = getStringArray(args, 'by')
      if (by.length > 0) result.push('--by', ...by)
      // aggs is [[col, func], ...] - serialize as JSON
      const aggs = args['aggs']
      if (Array.isArray(aggs)) {
        result.push('--aggs', JSON.stringify(aggs))
      }
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_value_counts: {
    category: 'munge',
    subcommand: 'value-counts',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const normalize = getBool(args, 'normalize')
      if (normalize) result.push('--normalize')
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_pivot: {
    category: 'munge',
    subcommand: 'pivot',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const index = getStringArray(args, 'index')
      if (index.length > 0) result.push('--index', ...index)
      const on = getString(args, 'on')
      if (on) result.push('--on', on)
      const values = getString(args, 'values')
      if (values) result.push('--values', values)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_melt: {
    category: 'munge',
    subcommand: 'melt',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const idVars = getStringArray(args, 'id_vars')
      if (idVars.length > 0) result.push('--id-vars', ...idVars)
      const valueVars = getStringArray(args, 'value_vars')
      if (valueVars.length > 0) result.push('--value-vars', ...valueVars)
      const varName = getString(args, 'variable_name')
      if (varName) result.push('--var-name', varName)
      const valName = getString(args, 'value_name')
      if (valName) result.push('--val-name', valName)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_drop_na: {
    category: 'munge',
    subcommand: 'dropna',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const how = getString(args, 'how')
      if (how) result.push('--how', how)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_fill_na: {
    category: 'munge',
    subcommand: 'fillna',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const strategy = getString(args, 'strategy')
      if (strategy) result.push('--strategy', strategy)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_deduplicate: {
    category: 'munge',
    subcommand: 'dedup',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const keep = getString(args, 'keep')
      if (keep) result.push('--keep', keep)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_lag_lead: {
    category: 'munge',
    subcommand: 'lag',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const periods = getNumber(args, 'periods')
      if (periods !== undefined) result.push('--periods', periods.toString())
      const direction = getString(args, 'direction')
      if (direction) result.push('--direction', direction)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_diff: {
    category: 'munge',
    subcommand: 'diff',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const periods = getNumber(args, 'periods')
      if (periods !== undefined) result.push('--periods', periods.toString())
      const diffType = getString(args, 'diff_type')
      if (diffType) result.push('--type', diffType)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_standardize: {
    category: 'munge',
    subcommand: 'standardize',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const cols = getStringArray(args, 'columns')
      if (cols.length > 0) result.push('--cols', ...cols)
      const method = getString(args, 'method')
      if (method) result.push('--method', method)
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_bin: {
    category: 'munge',
    subcommand: 'bin',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const strategy = getString(args, 'strategy')
      if (strategy) result.push('--strategy', strategy)
      // bins can be number or array
      const bins = args['bins']
      if (Array.isArray(bins)) {
        result.push('--bins', JSON.stringify(bins))
      } else if (typeof bins === 'number') {
        result.push('--bins', bins.toString())
      }
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },

  munge_one_hot_encode: {
    category: 'munge',
    subcommand: 'onehot',
    argMapper: (args) => {
      const result: string[] = []
      const dataset = getString(args, 'dataset')
      if (dataset) result.push(dataset)
      const col = getString(args, 'column')
      if (col) result.push('--col', col)
      const dropFirst = getBool(args, 'drop_first')
      if (dropFirst) result.push('--drop-first')
      const outputName = getString(args, 'result_name')
      if (outputName) result.push('--name', outputName)
      return result
    },
  },
}

/**
 * Tools to exclude from script export (inspection-only tools)
 */
export const excludedTools = new Set([
  'list_datasets',
  'describe_dataset',
  'head_dataset',
  'compute_correlation',
  'data_quality_profile',
  'preview_cleaning',
  'verify_cleaning',
  'cleaning_session_start',
  'cleaning_session_status',
  'list_cleaning_sessions',
  'cleaning_session_checkpoints',
  'suggest_cleaning',
  'batch_process',
  'compare_datasets',
  'generate_report',
])

/**
 * Check if a tool should be included in script export
 */
export function shouldExportTool(toolName: string): boolean {
  return !excludedTools.has(toolName) && toolMappings[toolName] !== undefined
}

/**
 * Convert an MCP tool call to a CLI command
 */
export function toolToCliCommand(
  toolName: string,
  args: Record<string, unknown>
): CliCommand | null {
  const mapping = toolMappings[toolName]
  if (!mapping) {
    return null
  }

  return {
    category: mapping.category,
    subcommand: mapping.subcommand,
    args: mapping.argMapper(args),
  }
}

/**
 * Format a CLI command as a string
 */
export function formatCliCommand(cmd: CliCommand, sessionVar: string = '$SESSION_FILE'): string {
  const parts = ['p2a', '--session', `"${sessionVar}"`, cmd.category, cmd.subcommand, ...cmd.args]
  return parts.join(' ')
}

/**
 * Get a list of all supported MCP tools that can be exported
 */
export function getSupportedTools(): string[] {
  return Object.keys(toolMappings)
}
