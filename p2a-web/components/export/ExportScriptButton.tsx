'use client'

import { useState, useCallback } from 'react'
import { useCommandHistoryStore } from '@/lib/store/command-history-store'
import { generateBashScript } from '@/lib/cli/script-generator'

export function ExportScriptButton() {
  const [isOpen, setIsOpen] = useState(false)
  const [script, setScript] = useState('')
  const [copied, setCopied] = useState(false)
  const { commands, getExportableCommands, clearCommands } = useCommandHistoryStore()

  const exportableCommands = getExportableCommands()
  const hasCommands = exportableCommands.length > 0

  const handleExport = useCallback(() => {
    const cmds = getExportableCommands()
    if (cmds.length === 0) return

    const generatedScript = generateBashScript(cmds, {
      includeComments: true,
      sessionTitle: `Analysis Session - ${new Date().toLocaleDateString()}`,
    })
    setScript(generatedScript)
    setIsOpen(true)
    setCopied(false)
  }, [getExportableCommands])

  const handleClose = useCallback(() => {
    setIsOpen(false)
    setScript('')
    setCopied(false)
  }, [])

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(script)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }, [script])

  const handleDownload = useCallback(() => {
    const blob = new Blob([script], { type: 'text/x-sh' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `p2a-analysis-${Date.now()}.sh`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [script])

  const handleClear = useCallback(() => {
    if (confirm('Clear all recorded commands? This cannot be undone.')) {
      clearCommands()
      handleClose()
    }
  }, [clearCommands, handleClose])

  return (
    <>
      {/* Export Button */}
      <button
        onClick={handleExport}
        disabled={!hasCommands}
        className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg transition-colors ${
          hasCommands
            ? 'bg-green-600 hover:bg-green-700 text-white'
            : 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
        }`}
        title={hasCommands ? `Export ${exportableCommands.length} command(s) to bash script` : 'No commands to export'}
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Export Script
        {hasCommands && (
          <span className="bg-green-500 text-white text-xs px-1.5 py-0.5 rounded-full">
            {exportableCommands.length}
          </span>
        )}
      </button>

      {/* Modal */}
      {isOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <div>
                <h2 className="text-lg font-semibold">Export Bash Script</h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {exportableCommands.length} command(s) ready to export
                </p>
              </div>
              <button
                onClick={handleClose}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Script Preview */}
            <div className="flex-1 overflow-auto p-6">
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm font-mono overflow-x-auto whitespace-pre">
                {script}
              </pre>
            </div>

            {/* Footer Actions */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 rounded-b-xl">
              <button
                onClick={handleClear}
                className="px-4 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
              >
                Clear History
              </button>
              <div className="flex items-center gap-3">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-2 px-4 py-2 text-sm bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-lg transition-colors"
                >
                  {copied ? (
                    <>
                      <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy
                    </>
                  )}
                </button>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-2 px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download .sh
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
