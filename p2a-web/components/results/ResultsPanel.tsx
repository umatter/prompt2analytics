'use client'

import { useResultsStore, type AnalysisResult } from '@/lib/store/results-store'

export function ResultsPanel() {
  const { results, expandedResultId, removeResult, clearResults, toggleExpand } = useResultsStore()

  const getTypeIcon = (type: AnalysisResult['type']) => {
    switch (type) {
      case 'regression':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
            />
          </svg>
        )
      case 'summary':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        )
      case 'chart':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"
            />
          </svg>
        )
      case 'table':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
        )
      default:
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        )
    }
  }

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Results</h2>
        {results.length > 0 && (
          <button
            onClick={clearResults}
            className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            Clear all
          </button>
        )}
      </div>

      {results.length === 0 ? (
        <div className="text-center py-8">
          <svg
            className="w-12 h-12 mx-auto mb-3 text-gray-300 dark:text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p className="text-sm text-gray-500">No results yet</p>
          <p className="text-xs text-gray-400 mt-1">
            Results from your analyses will appear here
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {results.map((result) => (
            <div
              key={result.id}
              className="border rounded-lg overflow-hidden bg-white dark:bg-gray-800"
            >
              <div className="flex items-center justify-between">
                <button
                  onClick={() => toggleExpand(result.id)}
                  className="flex-1 px-3 py-2 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-gray-500">{getTypeIcon(result.type)}</span>
                    <span className="text-sm font-medium">{result.title}</span>
                  </div>
                  <svg
                    className={`w-4 h-4 text-gray-400 transition-transform ${
                      expandedResultId === result.id ? 'rotate-180' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>
                <button
                  onClick={() => removeResult(result.id)}
                  className="px-2 py-2 text-gray-400 hover:text-red-500 transition-colors"
                  title="Remove result"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>

              {expandedResultId === result.id && (
                <div className="px-3 py-2 border-t bg-gray-50 dark:bg-gray-900">
                  {/* Image display for charts */}
                  {result.imageData && (
                    <div className="mb-2">
                      <img
                        src={`data:image/png;base64,${result.imageData}`}
                        alt={result.title}
                        className="w-full rounded border"
                      />
                    </div>
                  )}

                  {/* Text content */}
                  <pre className="text-xs font-mono whitespace-pre-wrap overflow-x-auto max-h-96">
                    {result.content}
                  </pre>

                  <div className="flex items-center justify-between text-xs text-gray-400 mt-2 pt-2 border-t">
                    <span>{result.timestamp.toLocaleString()}</span>
                    {result.toolName && (
                      <span className="font-mono text-gray-500">{result.toolName}</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
