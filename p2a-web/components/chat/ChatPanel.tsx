'use client'

import { useRef, useEffect } from 'react'
import { useChatStore, type ChatMessage } from '@/lib/store/chat-store'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { ExportScriptButton } from '../export/ExportScriptButton'

export function ChatPanel() {
  const { messages, isProcessing, status, error, clearMessages } = useChatStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive or status changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, status])

  return (
    <div className="flex flex-col h-full">
      {/* Header with actions */}
      <div className="flex items-center justify-between px-4 py-2 border-b bg-gray-50 dark:bg-gray-900/50">
        <h2 className="text-sm font-medium text-gray-600 dark:text-gray-400">Chat</h2>
        <div className="flex items-center gap-2">
          <ExportScriptButton />
          {messages.length > 0 && (
            <button
              onClick={() => {
                if (confirm('Clear all messages?')) {
                  clearMessages()
                }
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Clear chat messages"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <WelcomeMessage />
        ) : (
          <MessageList messages={messages} />
        )}

        {isProcessing && (
          <div className="flex items-center gap-2 text-gray-500 mt-4">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            <span className="font-medium">{status || 'Processing...'}</span>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t p-4 bg-white dark:bg-gray-900">
        <ChatInput />
      </div>
    </div>
  )
}

function WelcomeMessage() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
        <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </div>
      <h2 className="text-xl font-semibold mb-2">Welcome to prompt2analytics</h2>
      <p className="text-gray-600 dark:text-gray-400 max-w-md mb-6">
        I can help you analyze data using natural language. Load a dataset and ask me to run regressions, create visualizations, or explore your data.
      </p>
      <div className="grid grid-cols-1 gap-2 text-sm text-left max-w-md w-full">
        <SuggestionCard text="Load the sales.csv dataset and describe it" />
        <SuggestionCard text="Run an OLS regression of price on sqft and bedrooms" />
        <SuggestionCard text="Create a histogram of the income column" />
      </div>
    </div>
  )
}

function SuggestionCard({ text }: { text: string }) {
  const { setInput, sendMessage } = useChatStore()

  const handleClick = () => {
    setInput(text)
  }

  return (
    <button
      onClick={handleClick}
      className="p-3 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg text-left border transition-colors"
    >
      {text}
    </button>
  )
}
