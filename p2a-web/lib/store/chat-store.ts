import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { api } from '@/lib/api/client'
import { useSettingsStore } from './settings-store'
import { useSessionStore } from './session-store'
import { useCommandHistoryStore } from './command-history-store'
import type { Message, ToolCall, ContentItem } from '@/lib/types/api'

export interface ImageData {
  data: string
  mime_type: string
}

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
  images?: ImageData[]  // Images from viz tools
  isStreaming?: boolean
}

interface ChatState {
  messages: ChatMessage[]
  input: string
  isProcessing: boolean
  status: string | null  // Current status message (e.g., "Connecting to OpenAI...", "Calling viz_histogram...")
  error: string | null
  abortController: AbortController | null

  // Prompt history for arrow-up navigation
  promptHistory: string[]
  historyIndex: number  // -1 means not navigating history, 0+ means position in history

  // Actions
  setInput: (input: string) => void
  setError: (error: string | null) => void
  setStatus: (status: string | null) => void
  sendMessage: () => Promise<void>
  cancelRequest: () => void
  clearMessages: () => void

  // History navigation
  navigateHistoryUp: () => void
  navigateHistoryDown: () => void
  resetHistoryIndex: () => void

  // Streaming actions
  addUserMessage: (content: string) => string
  addStreamingMessage: (id: string) => void
  updateStreamingContent: (id: string, content: string) => void
  finalizeStreamingMessage: (id: string) => void
}

let messageIdCounter = 0
const REQUEST_TIMEOUT_MS = 60000 // 60 second timeout

export const useChatStore = create<ChatState>()(
  immer((set, get) => ({
    messages: [],
    input: '',
    isProcessing: false,
    status: null,
    error: null,
    abortController: null,
    promptHistory: [],
    historyIndex: -1,

    setInput: (input: string) => {
      set((state) => {
        state.input = input
      })
    },

    setError: (error: string | null) => {
      set((state) => {
        state.error = error
        state.isProcessing = false
        state.status = null
        state.abortController = null
      })
    },

    setStatus: (status: string | null) => {
      set((state) => {
        state.status = status
      })
    },

    cancelRequest: () => {
      const { abortController } = get()
      if (abortController) {
        abortController.abort()
      }
      set((state) => {
        state.isProcessing = false
        state.status = null
        state.abortController = null
        state.error = 'Request cancelled'
      })
    },

    navigateHistoryUp: () => {
      const { promptHistory, historyIndex } = get()
      if (promptHistory.length === 0) return

      const newIndex = historyIndex === -1
        ? promptHistory.length - 1  // Start from most recent
        : Math.max(0, historyIndex - 1)  // Go to older

      set((state) => {
        state.historyIndex = newIndex
        state.input = promptHistory[newIndex]
      })
    },

    navigateHistoryDown: () => {
      const { promptHistory, historyIndex } = get()
      if (historyIndex === -1) return  // Not navigating history

      const newIndex = historyIndex + 1

      set((state) => {
        if (newIndex >= promptHistory.length) {
          // Past the end, clear input and reset index
          state.historyIndex = -1
          state.input = ''
        } else {
          state.historyIndex = newIndex
          state.input = promptHistory[newIndex]
        }
      })
    },

    resetHistoryIndex: () => {
      set((state) => {
        state.historyIndex = -1
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

      // Create abort controller for cancellation
      const abortController = new AbortController()

      // Add user message and set processing state
      set((state) => {
        state.messages.push(userMessage)
        // Add to prompt history (avoid duplicates of the last entry)
        const lastPrompt = state.promptHistory[state.promptHistory.length - 1]
        if (lastPrompt !== userMessage.content) {
          state.promptHistory.push(userMessage.content)
        }
        state.historyIndex = -1  // Reset history navigation
        state.input = ''
        state.isProcessing = true
        state.status = 'Starting analysis...'
        state.error = null
        state.abortController = abortController as unknown as AbortController
      })

      // Build history for the API
      const history: Message[] = messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }))

      // Set up timeout (increased to 120 seconds for LLM responses)
      const REQUEST_TIMEOUT = 120000
      const timeoutId = setTimeout(() => {
        abortController.abort()
        set((state) => {
          state.error = 'Request timed out. Please check your API key in Settings.'
          state.isProcessing = false
          state.status = null
          state.abortController = null
        })
      }, REQUEST_TIMEOUT)

      try {
        // Get provider config and session ID
        const providerConfig = useSettingsStore.getState().getProviderConfig()
        const sessionId = api.getSessionId()

        if (!sessionId) {
          clearTimeout(timeoutId)
          set((state) => {
            state.error = 'No session initialized'
            state.isProcessing = false
            state.status = null
            state.abortController = null
          })
          return
        }

        // Use SSE streaming endpoint
        const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
        const interpretResults = useSettingsStore.getState().interpretResults
        console.log('[Chat] Sending message with interpret:', interpretResults)
        const response = await fetch(`${API_BASE}/api/llm/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            message: userMessage.content,
            provider: providerConfig,
            history,
            interpret: interpretResults,
          }),
          signal: abortController.signal,
        })

        if (!response.ok) {
          clearTimeout(timeoutId)
          const errorText = await response.text()
          set((state) => {
            state.error = `API error: ${errorText}`
            state.isProcessing = false
            state.status = null
            state.abortController = null
          })
          return
        }

        // Process SSE stream
        const reader = response.body?.getReader()
        const decoder = new TextDecoder()

        if (!reader) {
          throw new Error('No response body')
        }

        let buffer = ''
        let pendingImages: ImageData[] = []  // Accumulate images from tool results
        let streamingMessageId: string | null = null  // Track the streaming message

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)
              if (!data) continue

              try {
                const event = JSON.parse(data)

                switch (event.type) {
                  case 'status':
                    set((state) => {
                      state.status = event.message
                    })
                    break
                  case 'tool_start':
                    set((state) => {
                      state.status = `Calling ${event.tool}...`
                    })
                    break
                  case 'tool_end':
                    set((state) => {
                      state.status = `${event.tool} completed (${event.elapsed_ms}ms)`
                    })
                    break
                  case 'tool_result':
                    // Capture images from viz tool results
                    if (event.images && event.images.length > 0) {
                      pendingImages.push(...event.images)
                    }
                    break
                  case 'content':
                    // Handle streaming content chunks
                    if (!streamingMessageId) {
                      // Create a new streaming message on first content chunk
                      streamingMessageId = `msg-${++messageIdCounter}`
                      set((state) => {
                        state.messages.push({
                          id: streamingMessageId!,
                          role: 'assistant',
                          content: event.text || '',
                          timestamp: new Date(),
                          isStreaming: true,
                        })
                        state.status = null  // Clear status when content starts
                      })
                    } else {
                      // Append to existing streaming message
                      set((state) => {
                        const message = state.messages.find((m) => m.id === streamingMessageId)
                        if (message) {
                          message.content += event.text || ''
                        }
                      })
                    }
                    break
                  case 'done':
                    clearTimeout(timeoutId)
                    // Record tool calls for script export
                    if (event.message.tool_calls && event.message.tool_calls.length > 0) {
                      const commandHistoryStore = useCommandHistoryStore.getState()
                      for (const toolCall of event.message.tool_calls) {
                        commandHistoryStore.addCommand({
                          timestamp: new Date(),
                          toolName: toolCall.name,
                          arguments: toolCall.arguments || {},
                          success: true,
                        })
                      }
                    }
                    if (streamingMessageId) {
                      // Finalize the streaming message with additional data
                      set((state) => {
                        const message = state.messages.find((m) => m.id === streamingMessageId)
                        if (message) {
                          // Update content if the final message has different content
                          // (this handles cases where streaming didn't capture everything)
                          if (event.message.content && message.content !== event.message.content) {
                            message.content = event.message.content
                          }
                          message.isStreaming = false
                          message.toolCalls = event.message.tool_calls
                          message.images = pendingImages.length > 0 ? pendingImages : undefined
                        }
                        state.isProcessing = false
                        state.status = null
                        state.abortController = null
                      })
                    } else {
                      // No streaming happened, add the complete message
                      const assistantMessage: ChatMessage = {
                        id: `msg-${++messageIdCounter}`,
                        role: 'assistant',
                        content: event.message.content,
                        timestamp: new Date(),
                        toolCalls: event.message.tool_calls,
                        images: pendingImages.length > 0 ? pendingImages : undefined,
                      }
                      set((state) => {
                        state.messages.push(assistantMessage)
                        state.isProcessing = false
                        state.status = null
                        state.abortController = null
                      })
                    }
                    return
                  case 'error':
                    clearTimeout(timeoutId)
                    // Handle session not found
                    if (event.error?.toLowerCase().includes('session not found')) {
                      useSessionStore.getState().clearSession()
                      await useSessionStore.getState().initSession()
                      set((state) => {
                        state.error = 'Session expired. Please try again.'
                        state.isProcessing = false
                        state.status = null
                        state.abortController = null
                      })
                    } else {
                      set((state) => {
                        state.error = event.error || 'Unknown error'
                        state.isProcessing = false
                        state.status = null
                        state.abortController = null
                      })
                    }
                    return
                }
              } catch {
                // Ignore parse errors
              }
            }
          }
        }

        clearTimeout(timeoutId)
      } catch (err) {
        clearTimeout(timeoutId)
        // Don't show error if it was cancelled
        if (err instanceof Error && err.name === 'AbortError') {
          set((state) => {
            state.isProcessing = false
            state.status = null
            state.abortController = null
          })
          return
        }
        set((state) => {
          state.error = err instanceof Error ? err.message : 'Network error'
          state.isProcessing = false
          state.status = null
          state.abortController = null
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
