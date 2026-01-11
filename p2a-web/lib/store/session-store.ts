import { create } from 'zustand'
import { api } from '@/lib/api/client'

interface SessionState {
  sessionId: string | null
  isInitialized: boolean
  isLoading: boolean
  error: string | null

  // Actions
  initSession: () => Promise<void>
  clearSession: () => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  isInitialized: false,
  isLoading: false,
  error: null,

  initSession: async () => {
    if (get().isLoading) return

    set({ isLoading: true, error: null })

    try {
      const response = await api.createSession()

      if (response.success && response.data) {
        const sessionId = response.data.session_id
        api.setSessionId(sessionId)
        set({
          sessionId,
          isInitialized: true,
          isLoading: false,
        })
      } else {
        set({
          error: response.error || 'Failed to create session',
          isLoading: false,
        })
      }
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Connection failed',
        isLoading: false,
      })
    }
  },

  clearSession: () => {
    api.setSessionId('')
    set({
      sessionId: null,
      isInitialized: false,
      error: null,
    })
  },
}))
