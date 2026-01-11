import type {
  ApiResponse,
  Session,
  CreateSessionResponse,
  ToolDefinition,
  ToolResult,
  LlmChatRequest,
  LlmChatResponse,
  LlmModelsResponse,
  ProviderConfig,
  Message,
  ListDatasetsResponse,
  DescribeDatasetResponse,
} from '@/lib/types/api'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'

class ApiClient {
  private sessionId: string | null = null

  setSessionId(id: string) {
    this.sessionId = id
  }

  getSessionId(): string | null {
    return this.sessionId
  }

  private async fetch<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<ApiResponse<T>> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    })
    return response.json()
  }

  // Health check
  async health(): Promise<{ status: string; version: string; active_sessions: number }> {
    const response = await fetch(`${API_BASE}/health`)
    return response.json()
  }

  // Session management
  async createSession(userId?: string): Promise<ApiResponse<CreateSessionResponse>> {
    return this.fetch('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId }),
    })
  }

  async getSession(id: string): Promise<ApiResponse<Session>> {
    return this.fetch(`/api/sessions/${id}`)
  }

  async listSessions(): Promise<ApiResponse<Session[]>> {
    return this.fetch('/api/sessions')
  }

  async deleteSession(id: string): Promise<ApiResponse<void>> {
    return this.fetch(`/api/sessions/${id}`, { method: 'DELETE' })
  }

  // Tool discovery
  async listTools(): Promise<ApiResponse<ToolDefinition[]>> {
    return this.fetch('/api/tools')
  }

  // Tool execution
  async callTool(
    name: string,
    args: Record<string, unknown>
  ): Promise<ApiResponse<ToolResult>> {
    if (!this.sessionId) {
      return { success: false, error: 'No session initialized' }
    }
    return this.fetch(`/api/tools/${name}`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: this.sessionId,
        arguments: args,
      }),
    })
  }

  // LLM chat
  async chat(
    message: string,
    history: Message[] = [],
    provider?: ProviderConfig
  ): Promise<ApiResponse<LlmChatResponse>> {
    if (!this.sessionId) {
      return { success: false, error: 'No session initialized' }
    }
    const request: LlmChatRequest = {
      session_id: this.sessionId,
      message,
      history,
      provider,
    }
    return this.fetch('/api/llm/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  // LLM models
  async listModels(): Promise<ApiResponse<LlmModelsResponse>> {
    return this.fetch('/api/llm/models')
  }

  // Dataset operations (via tool calls)
  async listDatasets(): Promise<ApiResponse<ListDatasetsResponse>> {
    const result = await this.callTool('list_datasets', {})
    if (result.success && result.data) {
      // Parse the tool result content
      const textContent = result.data.content.find((c) => c.type === 'text')
      if (textContent?.text) {
        try {
          const datasets = JSON.parse(textContent.text)
          return { success: true, data: { datasets } }
        } catch {
          return { success: true, data: { datasets: [] } }
        }
      }
    }
    return { success: false, error: result.error || 'Failed to list datasets' }
  }

  async describeDataset(name: string): Promise<ApiResponse<DescribeDatasetResponse>> {
    const result = await this.callTool('describe_dataset', { name })
    if (result.success && result.data) {
      const textContent = result.data.content.find((c) => c.type === 'text')
      if (textContent?.text) {
        try {
          const info = JSON.parse(textContent.text)
          return { success: true, data: info }
        } catch {
          return { success: false, error: 'Failed to parse dataset info' }
        }
      }
    }
    return { success: false, error: result.error || 'Failed to describe dataset' }
  }
}

// Singleton instance
export const api = new ApiClient()
