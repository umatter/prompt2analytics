'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { type ChatMessage } from '@/lib/store/chat-store'
import { StreamingIndicator } from './StreamingIndicator'
import { ToolCall } from './ToolCall'

interface MessageProps {
  message: ChatMessage
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === 'user'
  const isStreaming = message.isStreaming
  const hasContent = message.content && message.content.trim().length > 0

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
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              </div>
            )}

            {/* Streaming cursor when content is being streamed */}
            {isStreaming && hasContent && (
              <span className="inline-block w-2 h-4 ml-1 bg-blue-500 animate-pulse" />
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
