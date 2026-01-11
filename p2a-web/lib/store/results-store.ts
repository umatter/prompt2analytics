import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'

export interface AnalysisResult {
  id: string
  type: 'regression' | 'summary' | 'chart' | 'table' | 'other'
  title: string
  timestamp: Date
  content: string
  toolName?: string
  imageData?: string // Base64 encoded image
}

interface ResultsState {
  results: AnalysisResult[]
  expandedResultId: string | null

  // Actions
  addResult: (result: Omit<AnalysisResult, 'id' | 'timestamp'>) => void
  removeResult: (id: string) => void
  clearResults: () => void
  toggleExpand: (id: string) => void
}

let resultIdCounter = 0

export const useResultsStore = create<ResultsState>()(
  immer((set) => ({
    results: [],
    expandedResultId: null,

    addResult: (result) => {
      const newResult: AnalysisResult = {
        ...result,
        id: `result-${++resultIdCounter}`,
        timestamp: new Date(),
      }
      set((state) => {
        state.results.unshift(newResult) // Add to beginning
        state.expandedResultId = newResult.id // Auto-expand new result
      })
    },

    removeResult: (id) => {
      set((state) => {
        state.results = state.results.filter((r) => r.id !== id)
        if (state.expandedResultId === id) {
          state.expandedResultId = null
        }
      })
    },

    clearResults: () => {
      set((state) => {
        state.results = []
        state.expandedResultId = null
      })
    },

    toggleExpand: (id) => {
      set((state) => {
        state.expandedResultId = state.expandedResultId === id ? null : id
      })
    },
  }))
)

// Helper function to determine result type from tool name
export function getResultType(toolName: string): AnalysisResult['type'] {
  if (toolName.includes('regression') || toolName.includes('ols') || toolName.includes('2sls')) {
    return 'regression'
  }
  if (toolName.includes('describe') || toolName.includes('summary')) {
    return 'summary'
  }
  if (
    toolName.includes('histogram') ||
    toolName.includes('scatter') ||
    toolName.includes('plot') ||
    toolName.includes('chart')
  ) {
    return 'chart'
  }
  if (toolName.includes('head') || toolName.includes('preview')) {
    return 'table'
  }
  return 'other'
}

// Helper to format tool name as title
export function formatToolTitle(toolName: string): string {
  return toolName
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (l) => l.toUpperCase())
}
