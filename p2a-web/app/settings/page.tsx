'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useSettingsStore } from '@/lib/store/settings-store'

export default function SettingsPage() {
  const {
    provider,
    ollamaBaseUrl,
    ollamaModel,
    anthropicApiKey,
    anthropicModel,
    openaiApiKey,
    openaiModel,
    temperature,
    maxTokens,
    theme,
    setProvider,
    setOllamaBaseUrl,
    setOllamaModel,
    setAnthropicApiKey,
    setAnthropicModel,
    setOpenaiApiKey,
    setOpenaiModel,
    setTemperature,
    setMaxTokens,
    setTheme,
  } = useSettingsStore()

  const [showAnthropicKey, setShowAnthropicKey] = useState(false)
  const [showOpenaiKey, setShowOpenaiKey] = useState(false)

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="h-14 border-b flex items-center justify-between px-4 bg-white dark:bg-gray-900">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </Link>
          <h1 className="text-xl font-semibold">Settings</h1>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-2xl mx-auto p-6 space-y-8">
        {/* LLM Provider Section */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border p-6">
          <h2 className="text-lg font-semibold mb-4">LLM Provider</h2>

          {/* Provider Selection */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Provider</label>
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value as 'ollama' | 'anthropic' | 'openai')}
                className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ollama">Ollama (Local)</option>
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="openai">OpenAI (GPT)</option>
              </select>
            </div>

            {/* Ollama Settings */}
            {provider === 'ollama' && (
              <div className="space-y-4 pt-4 border-t">
                <div>
                  <label className="block text-sm font-medium mb-2">Base URL</label>
                  <input
                    type="text"
                    value={ollamaBaseUrl}
                    onChange={(e) => setOllamaBaseUrl(e.target.value)}
                    placeholder="http://localhost:11434"
                    className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    URL of your Ollama server
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Model</label>
                  <input
                    type="text"
                    value={ollamaModel}
                    onChange={(e) => setOllamaModel(e.target.value)}
                    placeholder="llama3.2"
                    className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Model name (e.g., llama3.2, mistral, codellama)
                  </p>
                </div>
              </div>
            )}

            {/* Anthropic Settings */}
            {provider === 'anthropic' && (
              <div className="space-y-4 pt-4 border-t">
                <div>
                  <label className="block text-sm font-medium mb-2">API Key</label>
                  <div className="relative">
                    <input
                      type={showAnthropicKey ? 'text' : 'password'}
                      value={anthropicApiKey}
                      onChange={(e) => setAnthropicApiKey(e.target.value)}
                      placeholder="sk-ant-..."
                      className="w-full px-3 py-2 pr-10 border rounded-lg bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                      type="button"
                      onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      {showAnthropicKey ? (
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                        </svg>
                      ) : (
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      )}
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Your Anthropic API key (not stored permanently)
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Model</label>
                  <select
                    value={anthropicModel}
                    onChange={(e) => setAnthropicModel(e.target.value)}
                    className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="claude-sonnet-4-20250514">Claude Sonnet 4</option>
                    <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                    <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                    <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                  </select>
                </div>
              </div>
            )}

            {/* OpenAI Settings */}
            {provider === 'openai' && (
              <div className="space-y-4 pt-4 border-t">
                <div>
                  <label className="block text-sm font-medium mb-2">API Key</label>
                  <div className="relative">
                    <input
                      type={showOpenaiKey ? 'text' : 'password'}
                      value={openaiApiKey}
                      onChange={(e) => setOpenaiApiKey(e.target.value)}
                      placeholder="sk-..."
                      className="w-full px-3 py-2 pr-10 border rounded-lg bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                      type="button"
                      onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    >
                      {showOpenaiKey ? (
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                        </svg>
                      ) : (
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      )}
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Your OpenAI API key (not stored permanently)
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Model</label>
                  <select
                    value={openaiModel}
                    onChange={(e) => setOpenaiModel(e.target.value)}
                    className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="gpt-4o-mini">GPT-4o Mini</option>
                    <option value="gpt-4-turbo">GPT-4 Turbo</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                  </select>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Model Parameters */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border p-6">
          <h2 className="text-lg font-semibold mb-4">Model Parameters</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Temperature: {temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-gray-500 mt-1">
                Controls randomness. Lower = more focused, higher = more creative
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Max Tokens</label>
              <input
                type="number"
                min="256"
                max="32768"
                step="256"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Maximum number of tokens in the response
              </p>
            </div>
          </div>
        </section>

        {/* Appearance */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border p-6">
          <h2 className="text-lg font-semibold mb-4">Appearance</h2>
          <div>
            <label className="block text-sm font-medium mb-2">Theme</label>
            <div className="flex gap-2">
              {(['light', 'dark', 'system'] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTheme(t)}
                  className={`flex-1 px-4 py-2 rounded-lg border transition-colors ${
                    theme === t
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600'
                  }`}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* About */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border p-6">
          <h2 className="text-lg font-semibold mb-4">About</h2>
          <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
            <p><strong>prompt2analytics</strong> - Natural language data analytics</p>
            <p>Version: 0.1.0</p>
            <p>
              A Rust-powered analytics toolkit with 55+ tools for regression analysis,
              econometrics, machine learning, and visualization.
            </p>
          </div>
        </section>
      </main>
    </div>
  )
}
