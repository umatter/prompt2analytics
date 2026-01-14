'use client'

import { useRef, useEffect, KeyboardEvent, ChangeEvent } from 'react'
import { useChatStore } from '@/lib/store/chat-store'

export function ChatInput() {
  const {
    input,
    setInput,
    sendMessage,
    cancelRequest,
    isProcessing,
    navigateHistoryUp,
    navigateHistoryDown,
    resetHistoryIndex,
  } = useChatStore()
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }, [input])

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter to send, Shift+Enter for newline
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
      return
    }

    // Arrow up to navigate history (only when cursor is at start or input is empty)
    if (e.key === 'ArrowUp') {
      const textarea = textareaRef.current
      const cursorAtStart = textarea?.selectionStart === 0 && textarea?.selectionEnd === 0
      const isEmpty = !input.trim()

      if (cursorAtStart || isEmpty) {
        e.preventDefault()
        navigateHistoryUp()
      }
    }

    // Arrow down to navigate history (only when cursor is at end or input is empty)
    if (e.key === 'ArrowDown') {
      const textarea = textareaRef.current
      const cursorAtEnd = textarea?.selectionStart === input.length
      const isEmpty = !input.trim()

      if (cursorAtEnd || isEmpty) {
        e.preventDefault()
        navigateHistoryDown()
      }
    }
  }

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    // Reset history index when user types
    resetHistoryIndex()
  }

  const handleSend = () => {
    if (input.trim() && !isProcessing) {
      sendMessage()
    }
  }

  return (
    <div className="flex items-end gap-2">
      <div className="flex-1 relative">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask me to analyze your data..."
          disabled={isProcessing}
          rows={1}
          className="w-full resize-none rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-3 pr-12 text-gray-900 dark:text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </div>
      {isProcessing ? (
        <button
          onClick={cancelRequest}
          className="flex-shrink-0 w-10 h-10 rounded-lg bg-red-600 text-white hover:bg-red-700 flex items-center justify-center transition-colors"
          title="Stop request"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="1" />
          </svg>
        </button>
      ) : (
        <button
          onClick={handleSend}
          disabled={!input.trim()}
          className="flex-shrink-0 w-10 h-10 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
          title="Send message"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            />
          </svg>
        </button>
      )}
    </div>
  )
}
