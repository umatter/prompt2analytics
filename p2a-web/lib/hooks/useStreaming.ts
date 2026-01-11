'use client'

import { useEffect, useRef, useCallback, useState } from 'react'
import { getWebSocketClient, type WsMessage, type WebSocketClient } from '@/lib/api/websocket'
import { useChatStore, type ChatMessage } from '@/lib/store/chat-store'

interface StreamingState {
  isStreaming: boolean
  currentContent: string
  toolCalls: Array<{ name: string; args: Record<string, unknown> }>
  toolResults: Array<{ name: string; success: boolean; content: string }>
}

export function useStreaming() {
  const clientRef = useRef<WebSocketClient | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [streamingState, setStreamingState] = useState<StreamingState>({
    isStreaming: false,
    currentContent: '',
    toolCalls: [],
    toolResults: [],
  })

  const { addStreamingMessage, updateStreamingContent, finalizeStreamingMessage, setError } =
    useChatStore()

  // Connect to WebSocket on mount
  useEffect(() => {
    const client = getWebSocketClient()
    clientRef.current = client

    if (!client.isConnected) {
      client
        .connect()
        .then(() => {
          setIsConnected(true)
        })
        .catch((err) => {
          console.error('Failed to connect to WebSocket:', err)
          setError('Failed to connect to streaming server')
        })
    } else {
      setIsConnected(true)
    }

    return () => {
      // Don't disconnect on unmount - keep connection alive
    }
  }, [setError])

  const sendStreamingMessage = useCallback(
    async (content: string) => {
      const client = clientRef.current
      if (!client || !client.isConnected) {
        setError('Not connected to server')
        return
      }

      // Create a streaming message placeholder
      const messageId = `msg-stream-${Date.now()}`
      addStreamingMessage(messageId)

      setStreamingState({
        isStreaming: true,
        currentContent: '',
        toolCalls: [],
        toolResults: [],
      })

      // Generate request ID
      const requestId = `req-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`

      // Set up message handler
      client.onMessage(requestId, (message: WsMessage) => {
        switch (message.type) {
          case 'text':
            if (message.content) {
              setStreamingState((prev) => ({
                ...prev,
                currentContent: prev.currentContent + message.content,
              }))
              updateStreamingContent(messageId, message.content)
            }
            break

          case 'tool_call':
            if (message.name && message.args) {
              setStreamingState((prev) => ({
                ...prev,
                toolCalls: [...prev.toolCalls, { name: message.name!, args: message.args! }],
              }))
            }
            break

          case 'tool_result':
            if (message.name) {
              setStreamingState((prev) => ({
                ...prev,
                toolResults: [
                  ...prev.toolResults,
                  {
                    name: message.name!,
                    success: message.success ?? true,
                    content: message.content ?? '',
                  },
                ],
              }))
            }
            break

          case 'done':
            setStreamingState((prev) => ({
              ...prev,
              isStreaming: false,
            }))
            finalizeStreamingMessage(messageId)
            break

          case 'error':
            setStreamingState({
              isStreaming: false,
              currentContent: '',
              toolCalls: [],
              toolResults: [],
            })
            setError(message.message || 'Streaming error occurred')
            finalizeStreamingMessage(messageId)
            break
        }
      })

      // Send the message
      client.send({
        type: 'chat',
        id: requestId,
        message: content,
        stream: true,
      })
    },
    [addStreamingMessage, updateStreamingContent, finalizeStreamingMessage, setError]
  )

  return {
    isConnected,
    isStreaming: streamingState.isStreaming,
    currentContent: streamingState.currentContent,
    toolCalls: streamingState.toolCalls,
    toolResults: streamingState.toolResults,
    sendStreamingMessage,
  }
}
