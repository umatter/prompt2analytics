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
    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      })
      if (!response.ok) {
        const text = await response.text()
        return { success: false, error: text || `HTTP ${response.status}` }
      }
      return response.json()
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : 'Network error' }
    }
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
    provider?: ProviderConfig,
    signal?: AbortSignal
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
      signal,
    })
  }

  // LLM models
  async listModels(): Promise<ApiResponse<LlmModelsResponse>> {
    return this.fetch('/api/llm/models')
  }

  // Check which API keys are available from environment variables
  async checkEnvKeys(): Promise<ApiResponse<{ openai: boolean; anthropic: boolean }>> {
    return this.fetch('/api/llm/env-keys')
  }

  // Dataset operations (via tool calls)
  async listDatasets(): Promise<ApiResponse<ListDatasetsResponse>> {
    const result = await this.callTool('list_datasets', {})
    if (result.success && result.data) {
      // Parse the tool result content
      const textContent = result.data.content.find((c) => c.type === 'text')
      if (textContent?.text) {
        // Check if "No datasets" message
        if (textContent.text.includes('No datasets currently loaded')) {
          return { success: true, data: { datasets: [] } }
        }
        // Parse the markdown response: "- **name**: N rows x M columns"
        const datasets: Array<{ name: string; row_count: number; column_count: number }> = []
        const regex = /- \*\*(.+?)\*\*: (\d+) rows x (\d+) columns/g
        let match
        while ((match = regex.exec(textContent.text)) !== null) {
          datasets.push({
            name: match[1],
            row_count: parseInt(match[2], 10),
            column_count: parseInt(match[3], 10),
          })
        }
        return { success: true, data: { datasets } }
      }
    }
    return { success: false, error: result.error || 'Failed to list datasets' }
  }

  async describeDataset(name: string): Promise<ApiResponse<DescribeDatasetResponse>> {
    const result = await this.callTool('describe_dataset', { dataset: name })
    if (result.success && result.data) {
      const textContent = result.data.content.find((c) => c.type === 'text')
      if (textContent?.text) {
        // Parse markdown response: "Dataset: N rows x M columns" and column names
        const text = textContent.text
        const dimMatch = text.match(/Dataset: (\d+) rows x (\d+) columns/)
        const columns: string[] = []
        const colRegex = /Column: (\w+)/g
        let colMatch
        while ((colMatch = colRegex.exec(text)) !== null) {
          columns.push(colMatch[1])
        }
        if (dimMatch) {
          return {
            success: true,
            data: {
              row_count: parseInt(dimMatch[1], 10),
              column_count: parseInt(dimMatch[2], 10),
              columns,
            },
          }
        }
      }
    }
    return { success: false, error: result.error || 'Failed to describe dataset' }
  }
}

// Singleton instance
export const api = new ApiClient()
