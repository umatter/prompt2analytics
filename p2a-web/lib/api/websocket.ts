import { useSessionStore } from '@/lib/store/session-store'

export type WsMessageType = 'text' | 'tool_call' | 'tool_result' | 'done' | 'error'

export interface WsMessage {
  type: WsMessageType
  id: string
  content?: string
  name?: string
  args?: Record<string, unknown>
  success?: boolean
  message?: string
}

export interface WsRequest {
  type: 'chat' | 'tool'
  id: string
  message?: string
  stream?: boolean
  name?: string
  args?: Record<string, unknown>
}

type MessageHandler = (message: WsMessage) => void

export class WebSocketClient {
  private ws: WebSocket | null = null
  private messageHandlers: Map<string, MessageHandler> = new Map()
  private globalHandlers: Set<MessageHandler> = new Set()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private url: string

  constructor(url: string) {
    this.url = url
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const sessionId = useSessionStore.getState().sessionId
        const wsUrl = sessionId ? `${this.url}?session_id=${sessionId}` : this.url

        this.ws = new WebSocket(wsUrl)

        this.ws.onopen = () => {
          console.log('WebSocket connected')
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: WsMessage = JSON.parse(event.data)

            // Call specific handler for this message ID
            const handler = this.messageHandlers.get(message.id)
            if (handler) {
              handler(message)

              // Clean up handler when done or error
              if (message.type === 'done' || message.type === 'error') {
                this.messageHandlers.delete(message.id)
              }
            }

            // Call global handlers
            this.globalHandlers.forEach((h) => h(message))
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err)
          }
        }

        this.ws.onerror = (event) => {
          console.error('WebSocket error:', event)
          reject(new Error('WebSocket connection failed'))
        }

        this.ws.onclose = () => {
          console.log('WebSocket disconnected')
          this.attemptReconnect()
        }
      } catch (err) {
        reject(err)
      }
    })
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
      console.log(`Attempting to reconnect in ${delay}ms...`)

      setTimeout(() => {
        this.connect().catch((err) => {
          console.error('Reconnection failed:', err)
        })
      }, delay)
    } else {
      console.error('Max reconnection attempts reached')
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.messageHandlers.clear()
  }

  send(request: WsRequest, handler?: MessageHandler): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not connected')
    }

    if (handler) {
      this.messageHandlers.set(request.id, handler)
    }

    this.ws.send(JSON.stringify(request))
  }

  sendChat(message: string, stream: boolean = true): string {
    const id = `req-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`

    this.send({
      type: 'chat',
      id,
      message,
      stream,
    })

    return id
  }

  onMessage(id: string, handler: MessageHandler) {
    this.messageHandlers.set(id, handler)
  }

  addGlobalHandler(handler: MessageHandler) {
    this.globalHandlers.add(handler)
    return () => this.globalHandlers.delete(handler)
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }
}

// Singleton instance
let wsClient: WebSocketClient | null = null

export function getWebSocketClient(): WebSocketClient {
  if (!wsClient) {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
    const wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws'
    wsClient = new WebSocketClient(wsUrl)
  }
  return wsClient
}
