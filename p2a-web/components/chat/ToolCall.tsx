'use client'

import { useState } from 'react'
import type { ToolCall as ToolCallType } from '@/lib/types/api'

interface ToolCallProps {
  toolCall: ToolCallType
  result?: {
    content: string
    isError: boolean
  }
}

export function ToolCall({ toolCall, result }: ToolCallProps) {
  const [expanded, setExpanded] = useState(false)

  const statusColor = result
    ? result.isError
      ? 'border-red-300 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
      : 'border-green-300 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
    : 'border-yellow-300 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20'

  const statusIcon = result ? (
    result.isError ? (
      <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ) : (
      <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    )
  ) : (
    <svg className="w-4 h-4 text-yellow-500 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  )

  return (
    <div className={`rounded-lg border ${statusColor} overflow-hidden`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2 flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          {statusIcon}
          <span className="font-mono text-sm">{toolCall.name}</span>
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="px-3 py-2 border-t border-inherit bg-white/50 dark:bg-black/20">
          {/* Arguments */}
          <div className="mb-2">
            <div className="text-xs font-semibold text-gray-500 mb-1">Arguments</div>
            <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap bg-gray-100 dark:bg-gray-800 p-2 rounded">
              {JSON.stringify(toolCall.arguments, null, 2)}
            </pre>
          </div>

          {/* Result */}
          {result && (
            <div>
              <div className="text-xs font-semibold text-gray-500 mb-1">Result</div>
              <pre
                className={`text-xs font-mono overflow-x-auto whitespace-pre-wrap p-2 rounded ${
                  result.isError ? 'bg-red-100 dark:bg-red-900/30' : 'bg-gray-100 dark:bg-gray-800'
                }`}
              >
                {result.content}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
