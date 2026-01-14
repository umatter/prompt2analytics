'use client'

import { useEffect, useState, useCallback } from 'react'
import { useDatasetsStore } from '@/lib/store/datasets-store'
import { FileBrowser } from './FileBrowser'

export function DataPanel() {
  const {
    datasets,
    selectedDataset,
    preview,
    isLoading,
    error,
    loadDatasets,
    selectDataset,
    loadDatasetFromPath,
    deleteDataset,
    clearError,
  } = useDatasetsStore()

  const [filePath, setFilePath] = useState('')
  const [showFileBrowser, setShowFileBrowser] = useState(false)

  // Load datasets on mount
  useEffect(() => {
    loadDatasets()
  }, [loadDatasets])

  const handleLoadDataset = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      if (filePath.trim()) {
        await loadDatasetFromPath(filePath.trim())
        setFilePath('')
      }
    },
    [filePath, loadDatasetFromPath]
  )

  const handleFileSelect = useCallback(
    async (path: string) => {
      setFilePath(path)
      await loadDatasetFromPath(path)
    },
    [loadDatasetFromPath]
  )

  return (
    <div className="p-4">
      <h2 className="text-lg font-semibold mb-4">Datasets</h2>

      {/* Error display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            <button
              onClick={clearError}
              className="text-red-400 hover:text-red-600"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* File Browser Button */}
      <button
        onClick={() => setShowFileBrowser(true)}
        disabled={isLoading}
        className="w-full mb-4 flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors disabled:opacity-50"
      >
        <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
        </svg>
        <span className="text-sm text-gray-600 dark:text-gray-400">Browse for data files...</span>
      </button>

      {/* Or enter path manually */}
      <form onSubmit={handleLoadDataset} className="mb-4">
        <label htmlFor="file-path" className="block text-xs text-gray-400 mb-1">
          Or enter path manually
        </label>
        <div className="flex gap-2">
          <input
            id="file-path"
            type="text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="/path/to/data.csv"
            disabled={isLoading}
            className="flex-1 min-w-0 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !filePath.trim()}
            className="flex-shrink-0 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
            ) : (
              'Load'
            )}
          </button>
        </div>
      </form>

      {/* File Browser Modal */}
      <FileBrowser
        isOpen={showFileBrowser}
        onClose={() => setShowFileBrowser(false)}
        onSelect={handleFileSelect}
      />

      {/* Dataset List */}
      {datasets.length === 0 ? (
        <div className="text-center py-8 text-gray-500 text-sm">
          No datasets loaded yet.
          <br />
          Upload a file to get started.
        </div>
      ) : (
        <div className="space-y-2">
          {datasets.map((dataset) => (
            <div
              key={dataset.name}
              className={`group flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
                selectedDataset === dataset.name
                  ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <button
                onClick={() => selectDataset(dataset.name)}
                className="flex-1 text-left flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <div>
                  <div className="font-medium text-sm">{dataset.name}</div>
                  <div className="text-xs text-gray-500">
                    {dataset.row_count} rows, {dataset.column_count} cols
                  </div>
                </div>
              </button>
              <button
                onClick={() => deleteDataset(dataset.name)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-opacity"
                title="Delete dataset"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Dataset Preview (when selected) */}
      {selectedDataset && preview && (
        <div className="mt-4 border-t pt-4">
          <h3 className="text-sm font-medium mb-2">Preview: {selectedDataset}</h3>
          <div className="text-xs text-gray-500 mb-2">
            {preview.totalRows} rows, {preview.columns.length} columns
          </div>
          <div className="overflow-x-auto">
            <div className="flex flex-wrap gap-1">
              {preview.columns.slice(0, 10).map((col) => (
                <span
                  key={col}
                  className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono"
                >
                  {col}
                </span>
              ))}
              {preview.columns.length > 10 && (
                <span className="px-2 py-0.5 text-xs text-gray-400">
                  +{preview.columns.length - 10} more
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
