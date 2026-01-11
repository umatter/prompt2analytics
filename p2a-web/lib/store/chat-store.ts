import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { api } from '@/lib/api/client'
import type { Message, ToolCall, ContentItem } from '@/lib/types/api'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  toolCalls?: ToolCall[]
  toolResults?: Array<{
    name: string
    content: ContentItem[]
    isError: boolean
  }>
  isStreaming?: boolean
}

interface ChatState {
  messages: ChatMessage[]
  input: string
  isProcessing: boolean
  error: string | null

  // Actions
  setInput: (input: string) => void
  setError: (error: string | null) => void
  sendMessage: () => Promise<void>
  clearMessages: () => void

  // Streaming actions
  addUserMessage: (content: string) => string
  addStreamingMessage: (id: string) => void
  updateStreamingContent: (id: string, content: string) => void
  finalizeStreamingMessage: (id: string) => void
}

let messageIdCounter = 0

export const useChatStore = create<ChatState>()(
  immer((set, get) => ({
    messages: [],
    input: '',
    isProcessing: false,
    error: null,

    setInput: (input: string) => {
      set((state) => {
        state.input = input
      })
    },

    setError: (error: string | null) => {
      set((state) => {
        state.error = error
        state.isProcessing = false
      })
    },

    addUserMessage: (content: string) => {
      const id = `msg-${++messageIdCounter}`
      set((state) => {
        state.messages.push({
          id,
          role: 'user',
          content,
          timestamp: new Date(),
        })
        state.isProcessing = true
        state.error = null
      })
      return id
    },

    addStreamingMessage: (id: string) => {
      set((state) => {
        state.messages.push({
          id,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        })
      })
    },

    updateStreamingContent: (id: string, content: string) => {
      set((state) => {
        const message = state.messages.find((m) => m.id === id)
        if (message) {
          message.content += content
        }
      })
    },

    finalizeStreamingMessage: (id: string) => {
      set((state) => {
        const message = state.messages.find((m) => m.id === id)
        if (message) {
          message.isStreaming = false
        }
        state.isProcessing = false
      })
    },

    sendMessage: async () => {
      const { input, isProcessing, messages } = get()

      if (!input.trim() || isProcessing) return

      const userMessage: ChatMessage = {
        id: `msg-${++messageIdCounter}`,
        role: 'user',
        content: input.trim(),
        timestamp: new Date(),
      }

      // Add user message and create placeholder for assistant
      set((state) => {
        state.messages.push(userMessage)
        state.input = ''
        state.isProcessing = true
        state.error = null
      })

      // Build history for the API
      const history: Message[] = messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }))

      try {
        const response = await api.chat(userMessage.content, history)

        if (response.success && response.data) {
          const assistantMessage: ChatMessage = {
            id: `msg-${++messageIdCounter}`,
            role: 'assistant',
            content: response.data.message.content,
            timestamp: new Date(),
            toolCalls: response.data.message.tool_calls,
          }

          set((state) => {
            state.messages.push(assistantMessage)
            state.isProcessing = false
          })
        } else {
          set((state) => {
            state.error = response.error || 'Failed to get response'
            state.isProcessing = false
          })
        }
      } catch (err) {
        set((state) => {
          state.error = err instanceof Error ? err.message : 'Network error'
          state.isProcessing = false
        })
      }
    },

    clearMessages: () => {
      set((state) => {
        state.messages = []
        state.error = null
      })
    },
  }))
)
