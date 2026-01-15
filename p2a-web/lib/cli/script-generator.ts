/**
 * Bash Script Generator
 *
 * Generates reproducible bash scripts from recorded MCP tool calls.
 */

import type { CommandRecord } from '@/lib/store/command-history-store'
import { toolToCliCommand, formatCliCommand } from './tool-mapping'

export interface ScriptOptions {
  includeComments?: boolean
  sessionTitle?: string
}

/**
 * Generate a bash script from a list of command records
 */
export function generateBashScript(
  commands: CommandRecord[],
  options: ScriptOptions = {}
): string {
  const { includeComments = true, sessionTitle } = options
  const lines: string[] = []

  // Shebang and header
  lines.push('#!/bin/bash')
  lines.push('# p2a analytics script')
  if (sessionTitle) {
    lines.push(`# Session: ${sessionTitle}`)
  }
  lines.push(`# Generated: ${new Date().toISOString()}`)
  lines.push(`# Commands: ${commands.length}`)
  lines.push('')

  // Bash strict mode
  lines.push('set -euo pipefail')
  lines.push('')

  // Session file for the CLI
  lines.push('# Create temporary session file for dataset persistence')
  lines.push('SESSION_FILE=".p2a_session_$$.json"')
  lines.push('')

  // Separate load commands from analysis commands
  const loadCommands = commands.filter((cmd) => cmd.toolName === 'load_dataset')
  const analysisCommands = commands.filter((cmd) => cmd.toolName !== 'load_dataset')

  // Generate load commands first
  if (loadCommands.length > 0) {
    lines.push('# === Load Datasets ===')
    lines.push('')
    for (const cmd of loadCommands) {
      const cliCmd = toolToCliCommand(cmd.toolName, cmd.arguments)
      if (cliCmd) {
        if (includeComments) {
          const datasetName = cmd.arguments.name || extractFilename(cmd.arguments.path as string)
          lines.push(`# Load dataset: ${datasetName}`)
        }
        lines.push(formatCliCommand(cliCmd))
        lines.push('')
      }
    }
  }

  // Generate analysis commands
  if (analysisCommands.length > 0) {
    lines.push('# === Analysis ===')
    lines.push('')
    for (const cmd of analysisCommands) {
      const cliCmd = toolToCliCommand(cmd.toolName, cmd.arguments)
      if (cliCmd) {
        if (includeComments) {
          lines.push(`# ${formatToolDescription(cmd.toolName)}`)
        }
        const cmdStr = formatCliCommand(cliCmd)
        // Skip commands that are just comments (incomplete mappings)
        if (!cmdStr.includes('# ')) {
          lines.push(cmdStr)
        } else {
          lines.push(`# (Manual adjustment needed for: ${cmd.toolName})`)
          lines.push(`# ${cmdStr}`)
        }
        lines.push('')
      } else {
        // Tool not mapped - add as comment
        if (includeComments) {
          lines.push(`# Unsupported tool: ${cmd.toolName}`)
          lines.push(`# Arguments: ${JSON.stringify(cmd.arguments)}`)
          lines.push('')
        }
      }
    }
  }

  // Cleanup
  lines.push('# === Cleanup ===')
  lines.push('rm -f "$SESSION_FILE"')
  lines.push('')
  lines.push('echo "Script completed successfully"')

  return lines.join('\n')
}

/**
 * Extract filename from path
 */
function extractFilename(path: string | undefined): string {
  if (!path) return 'unknown'
  const parts = path.split(/[/\\]/)
  return parts[parts.length - 1] || 'unknown'
}

/**
 * Format tool name as human-readable description
 */
function formatToolDescription(toolName: string): string {
  const descriptions: Record<string, string> = {
    // Regression
    regression_ols: 'OLS Regression',
    regression_clustered: 'Clustered Standard Errors Regression',
    regression_diagnostics: 'Regression Diagnostics',

    // Panel
    panel_fixed_effects: 'Fixed Effects Panel Regression',
    panel_random_effects: 'Random Effects Panel Regression',
    hausman_test: 'Hausman Specification Test',
    panel_hdfe: 'High-Dimensional Fixed Effects',

    // Causal
    iv_2sls: '2SLS Instrumental Variables',
    iv_first_stage: 'IV First Stage Diagnostics',
    diff_in_diff: 'Difference-in-Differences',
    rd_estimate: 'Regression Discontinuity',

    // Discrete
    logit: 'Logit Regression',
    probit: 'Probit Regression',

    // Time Series
    ts_var: 'Vector Autoregression (VAR)',
    ts_arima_fit: 'ARIMA Model',
    ts_arima_forecast: 'ARIMA Forecast',
    ts_mstl: 'MSTL Decomposition',

    // ML
    ml_kmeans: 'K-Means Clustering',
    ml_pca: 'Principal Component Analysis',
    ml_dbscan: 'DBSCAN Clustering',
    ml_hierarchical: 'Hierarchical Clustering',
    ml_random_forest: 'Random Forest',

    // Viz
    viz_histogram: 'Histogram',
    viz_scatter: 'Scatter Plot',
    viz_line: 'Line Chart',
    viz_boxplot: 'Box Plot',
    viz_heatmap: 'Correlation Heatmap',

    // Munge
    munge_filter: 'Filter Data',
    munge_select: 'Select Columns',
    munge_rename: 'Rename Columns',
    munge_sort: 'Sort Data',
  }

  return descriptions[toolName] || toolName.replace(/_/g, ' ')
}

/**
 * Validate that a script has the required structure
 */
export function validateScript(script: string): { valid: boolean; issues: string[] } {
  const issues: string[] = []

  if (!script.startsWith('#!/bin/bash')) {
    issues.push('Missing shebang')
  }

  if (!script.includes('set -euo pipefail')) {
    issues.push('Missing strict mode')
  }

  if (!script.includes('SESSION_FILE=')) {
    issues.push('Missing session file definition')
  }

  // Check for at least one p2a command
  const p2aCommands = script.match(/p2a --session/g) || []
  if (p2aCommands.length === 0) {
    issues.push('No p2a commands found')
  }

  return {
    valid: issues.length === 0,
    issues,
  }
}
