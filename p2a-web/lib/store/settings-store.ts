import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ProviderConfig } from '@/lib/types/api'

type ProviderType = 'ollama' | 'anthropic' | 'openai'

interface SettingsState {
  // LLM Settings
  provider: ProviderType
  ollamaBaseUrl: string
  ollamaModel: string
  anthropicApiKey: string
  anthropicModel: string
  openaiApiKey: string
  openaiModel: string
  temperature: number
  maxTokens: number
  interpretResults: boolean  // Whether to have LLM interpret tool results

  // UI Settings
  theme: 'light' | 'dark' | 'system'

  // Actions
  setProvider: (provider: ProviderType) => void
  setOllamaBaseUrl: (url: string) => void
  setOllamaModel: (model: string) => void
  setAnthropicApiKey: (key: string) => void
  setAnthropicModel: (model: string) => void
  setOpenaiApiKey: (key: string) => void
  setOpenaiModel: (model: string) => void
  setTemperature: (temp: number) => void
  setMaxTokens: (tokens: number) => void
  setInterpretResults: (interpret: boolean) => void
  setTheme: (theme: 'light' | 'dark' | 'system') => void

  // Get current provider config
  getProviderConfig: () => ProviderConfig
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      // Default values
      provider: 'ollama',
      ollamaBaseUrl: 'http://localhost:11434',
      ollamaModel: 'llama3.2',
      anthropicApiKey: '',
      anthropicModel: 'claude-sonnet-4-20250514',
      openaiApiKey: '',
      openaiModel: 'gpt-4o',
      temperature: 0.7,
      maxTokens: 4096,
      interpretResults: true,  // Default to ON for backward compatibility
      theme: 'system',

      setProvider: (provider) => set({ provider }),
      setOllamaBaseUrl: (url) => set({ ollamaBaseUrl: url }),
      setOllamaModel: (model) => set({ ollamaModel: model }),
      setAnthropicApiKey: (key) => set({ anthropicApiKey: key }),
      setAnthropicModel: (model) => set({ anthropicModel: model }),
      setOpenaiApiKey: (key) => set({ openaiApiKey: key }),
      setOpenaiModel: (model) => set({ openaiModel: model }),
      setTemperature: (temp) => set({ temperature: temp }),
      setMaxTokens: (tokens) => set({ maxTokens: tokens }),
      setInterpretResults: (interpret) => set({ interpretResults: interpret }),
      setTheme: (theme) => set({ theme }),

      getProviderConfig: () => {
        const state = get()
        switch (state.provider) {
          case 'ollama':
            return {
              provider_type: 'ollama',
              base_url: state.ollamaBaseUrl,
              model: state.ollamaModel,
              temperature: state.temperature,
              max_tokens: state.maxTokens,
            }
          case 'anthropic':
            return {
              provider_type: 'anthropic',
              api_key: state.anthropicApiKey,
              model: state.anthropicModel,
              temperature: state.temperature,
              max_tokens: state.maxTokens,
            }
          case 'openai':
            return {
              provider_type: 'openai',
              api_key: state.openaiApiKey,
              model: state.openaiModel,
              temperature: state.temperature,
              max_tokens: state.maxTokens,
            }
        }
      },
    }),
    {
      name: 'p2a-settings',
      partialize: (state) => ({
        provider: state.provider,
        ollamaBaseUrl: state.ollamaBaseUrl,
        ollamaModel: state.ollamaModel,
        anthropicModel: state.anthropicModel,
        openaiModel: state.openaiModel,
        temperature: state.temperature,
        maxTokens: state.maxTokens,
        interpretResults: state.interpretResults,
        theme: state.theme,
        // Note: API keys are not persisted by default for security
      }),
    }
  )
)
