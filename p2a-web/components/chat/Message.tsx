'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'
import { type ChatMessage } from '@/lib/store/chat-store'
import { StreamingIndicator } from './StreamingIndicator'
import { ToolCall } from './ToolCall'

interface MessageProps {
  message: ChatMessage
}

// Fix tables that might have lost their newlines
// This handles cases where markdown tables are on a single line
function fixTableNewlines(content: string): string {
  // If content already has proper newlines for tables, return as-is
  if (content.includes('\n')) {
    return content
  }

  // Look for markdown table patterns - the separator row with dashes
  // Pattern like: |---|---| or |-----|-----|
  if (!content.includes('|--') && !content.includes('| --')) {
    return content
  }

  console.log('[fixTableNewlines] Detected table without newlines, attempting fix')
  console.log('[fixTableNewlines] Original:', content.substring(0, 300))

  // Strategy: Split on the pattern "| |" which indicates row boundaries
  // But we need to be careful - "| |" within a row (empty cell) vs between rows

  // The key insight: row boundaries are "|<space>|" followed by either:
  // 1. Dashes (separator row): | |---
  // 2. Content starting with space/letter: | | Variable

  // First, find and mark the separator row
  // The separator looks like: |---|---|---| or |-----|-----|
  const separatorMatch = content.match(/\|[-\s:|]+\|[-\s:|]+/)

  if (!separatorMatch) {
    return content
  }

  // Split the content around the separator
  const sepIndex = content.indexOf(separatorMatch[0])
  const sepEndIndex = sepIndex + separatorMatch[0].length

  // Find where the separator row actually starts (find the | before dashes)
  let sepStart = sepIndex
  // Go back to find the start of the separator row
  for (let i = sepIndex - 1; i >= 0; i--) {
    if (content[i] === '|') {
      // Check if this is the end of previous row or start of separator
      const afterPipe = content.substring(i + 1, sepIndex).trim()
      if (afterPipe === '' || afterPipe.match(/^[-:\s|]+$/)) {
        sepStart = i
        break
      }
    }
  }

  // Now find the full separator row end
  let sepEnd = sepEndIndex
  for (let i = sepEndIndex; i < content.length; i++) {
    if (content[i] === '|') {
      const beforePipe = content.substring(sepEndIndex, i).trim()
      if (beforePipe.match(/^[-:\s|]*$/)) {
        sepEnd = i + 1
      } else {
        break
      }
    }
  }

  const header = content.substring(0, sepStart).trim()
  const separator = content.substring(sepStart, sepEnd).trim()
  const body = content.substring(sepEnd).trim()

  console.log('[fixTableNewlines] Header:', header)
  console.log('[fixTableNewlines] Separator:', separator)
  console.log('[fixTableNewlines] Body preview:', body.substring(0, 100))

  // Now split the body into rows
  // Each row ends with | and the next starts with |
  // So we look for "| |" pattern (but not "||")
  const bodyRows = body.split(/\|\s*(?=\|)/).filter(r => r.trim())

  // Reconstruct with proper newlines
  let result = header + '\n' + separator + '\n'

  for (const row of bodyRows) {
    const trimmedRow = row.trim()
    if (trimmedRow && trimmedRow !== '|') {
      // Ensure row starts and ends with |
      const normalizedRow = trimmedRow.startsWith('|') ? trimmedRow : '|' + trimmedRow
      result += normalizedRow + '\n'
    }
  }

  console.log('[fixTableNewlines] Fixed result preview:', result.substring(0, 300))

  return result
}

// Custom components for better markdown rendering
const markdownComponents: Components = {
  // Wrap tables in a scrollable container
  table: ({ children }) => (
    <div className="overflow-x-auto my-4">
      <table className="min-w-full border-collapse text-sm">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-gray-200 dark:bg-gray-700">{children}</thead>
  ),
  th: ({ children }) => (
    <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left font-semibold">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-gray-300 dark:border-gray-600 px-3 py-2">
      {children}
    </td>
  ),
  tr: ({ children }) => (
    <tr className="even:bg-gray-100 dark:even:bg-gray-800/50 hover:bg-gray-200 dark:hover:bg-gray-700/50">
      {children}
    </tr>
  ),
  // Code blocks with better styling
  code: ({ className, children }) => {
    const isInline = !className
    if (isInline) {
      return (
        <code className="bg-gray-200 dark:bg-gray-700 px-1.5 py-0.5 rounded text-sm">
          {children}
        </code>
      )
    }
    return (
      <code className={className}>
        {children}
      </code>
    )
  },
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === 'user'
  const isStreaming = message.isStreaming
  const hasContent = message.content && message.content.trim().length > 0
  const hasImages = message.images && message.images.length > 0

  // Debug: log content to see if newlines are preserved
  if (!isUser && hasContent && message.content.includes('|')) {
    console.log('[Message] Content with table:', JSON.stringify(message.content))
    console.log('[Message] Has newlines:', message.content.includes('\n'))
  }

  // Pre-process content to fix tables that might be on a single line
  // This is a workaround for cases where newlines get lost
  const processedContent = hasContent ? fixTableNewlines(message.content) : ''

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2 ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <>
            {/* Show streaming indicator when no content yet */}
            {isStreaming && !hasContent && (
              <div className="py-2">
                <StreamingIndicator />
              </div>
            )}

            {/* Markdown content */}
            {hasContent && (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={markdownComponents}
                >
                  {processedContent}
                </ReactMarkdown>
              </div>
            )}

            {/* Streaming cursor when content is being streamed */}
            {isStreaming && hasContent && (
              <span className="inline-block w-2 h-4 ml-1 bg-blue-500 animate-pulse" />
            )}

            {/* Display images from viz tools */}
            {hasImages && (
              <div className="mt-4 space-y-4">
                {message.images!.map((img, i) => (
                  <div key={i} className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                    <img
                      src={`data:${img.mime_type};base64,${img.data}`}
                      alt={`Visualization ${i + 1}`}
                      className="max-w-full h-auto"
                    />
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* Tool calls display */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mt-3 space-y-2">
            {message.toolCalls.map((tc, i) => {
              const result = message.toolResults?.[i]
              return (
                <ToolCall
                  key={tc.id || i}
                  toolCall={tc}
                  result={result ? {
                    content: result.content.map(c =>
                      c.type === 'text' ? c.text : JSON.stringify(c)
                    ).join('\n'),
                    isError: result.isError
                  } : undefined}
                />
              )
            })}
          </div>
        )}

        {/* Timestamp */}
        <div className={`text-xs mt-2 ${isUser ? 'text-blue-200' : 'text-gray-400'}`}>
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}
