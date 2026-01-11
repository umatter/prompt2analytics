'use client'

import { useEffect } from 'react'
import { ThreeColumnLayout } from '@/components/layout/ThreeColumnLayout'
import { ChatPanel } from '@/components/chat/ChatPanel'
import { DataPanel } from '@/components/data/DataPanel'
import { ResultsPanel } from '@/components/results/ResultsPanel'
import { useSessionStore } from '@/lib/store/session-store'

export default function Home() {
  const { initSession, isInitialized, error } = useSessionStore()

  useEffect(() => {
    if (!isInitialized) {
      initSession()
    }
  }, [isInitialized, initSession])

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600">Connection Error</h1>
          <p className="mt-2 text-gray-600">{error}</p>
          <p className="mt-4 text-sm text-gray-500">
            Make sure the p2a-mcp server is running on port 8080
          </p>
          <button
            onClick={() => initSession()}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Connecting to analytics server...</p>
        </div>
      </div>
    )
  }

  return (
    <ThreeColumnLayout
      left={<DataPanel />}
      center={<ChatPanel />}
      right={<ResultsPanel />}
    />
  )
}
