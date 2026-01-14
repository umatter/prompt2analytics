'use client'

import { useState, useEffect, useCallback } from 'react'

interface FileEntry {
  name: string
  path: string
  is_dir: boolean
  size?: number
}

interface FileBrowserProps {
  isOpen: boolean
  onClose: () => void
  onSelect: (path: string) => void
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'

export function FileBrowser({ isOpen, onClose, onSelect }: FileBrowserProps) {
  const [currentPath, setCurrentPath] = useState<string>('')
  const [parentPath, setParentPath] = useState<string | null>(null)
  const [entries, setEntries] = useState<FileEntry[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadDirectory = useCallback(async (path?: string) => {
    setIsLoading(true)
    setError(null)

    try {
      const url = path
        ? `${API_BASE}/api/files?path=${encodeURIComponent(path)}`
        : `${API_BASE}/api/files`

      const response = await fetch(url)
      const data = await response.json()

      if (data.success && data.data) {
        setCurrentPath(data.data.path)
        setParentPath(data.data.parent)
        setEntries(data.data.entries)
      } else {
        setError(data.error || 'Failed to load directory')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Load home directory when opened
  useEffect(() => {
    if (isOpen) {
      loadDirectory()
    }
  }, [isOpen, loadDirectory])

  const handleEntryClick = (entry: FileEntry) => {
    if (entry.is_dir) {
      loadDirectory(entry.path)
    } else {
      onSelect(entry.path)
      onClose()
    }
  }

  const handleGoUp = () => {
    if (parentPath) {
      loadDirectory(parentPath)
    }
  }

  const formatSize = (bytes?: number) => {
    if (bytes === undefined) return ''
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold">Select a Data File</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Path bar */}
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={handleGoUp}
            disabled={!parentPath || isLoading}
            className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded disabled:opacity-50"
            title="Go up"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          </button>
          <div className="flex-1 text-sm font-mono text-gray-600 dark:text-gray-400 truncate">
            {currentPath}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
            </div>
          ) : error ? (
            <div className="p-4 text-center text-red-500">{error}</div>
          ) : entries.length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              No data files found in this directory
            </div>
          ) : (
            <div className="divide-y divide-gray-100 dark:divide-gray-800">
              {entries.map((entry) => (
                <button
                  key={entry.path}
                  onClick={() => handleEntryClick(entry)}
                  className="w-full flex items-center gap-3 px-4 py-2 hover:bg-gray-50 dark:hover:bg-gray-800 text-left"
                >
                  {entry.is_dir ? (
                    <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M10 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V8a2 2 0 00-2-2h-8l-2-2z" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  )}
                  <span className="flex-1 truncate">{entry.name}</span>
                  {!entry.is_dir && (
                    <span className="text-xs text-gray-400">{formatSize(entry.size)}</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500">
          Supported formats: CSV, Parquet, JSON, Excel, Stata, SAS
        </div>
      </div>
    </div>
  )
}
