// API types that match the backend

export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
}

export interface Session {
  id: string
  created_at: string
  last_accessed: string
  dataset_count: number
  user_id: string | null
}

export interface CreateSessionResponse {
  session_id: string
}

export interface ToolDefinition {
  name: string
  description: string
  input_schema: Record<string, unknown>
}

export interface ToolResult {
  success: boolean
  content: ContentItem[]
  error?: string
}

export interface ContentItem {
  type: 'text' | 'image'
  text?: string
  data?: string
  mime_type?: string
}

// LLM types
export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  tool_calls?: ToolCall[]
  tool_results?: ToolResultItem[]
}

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
}

export interface ToolResultItem {
  tool_call_id: string
  content: string
  is_error: boolean
}

export interface ProviderConfig {
  provider_type: 'ollama' | 'anthropic' | 'openai'
  api_key?: string
  base_url?: string
  model: string
  temperature?: number
  max_tokens?: number
}

export interface LlmChatRequest {
  session_id: string
  message: string
  provider?: ProviderConfig
  history?: Message[]
}

export interface LlmChatResponse {
  message: Message
}

export interface LlmModelsResponse {
  provider: string
  models: string[]
}

// Dataset types
export interface DatasetInfo {
  name: string
  row_count: number
  column_count: number
  columns: string[]
}

export interface ListDatasetsResponse {
  datasets: DatasetInfo[]
}

export interface DescribeDatasetResponse extends DatasetInfo {
  dtypes: Record<string, string>
}
