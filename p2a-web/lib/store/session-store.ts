import { create } from 'zustand'
import { api } from '@/lib/api/client'

// Keep-alive interval: ping server every 5 minutes to prevent session expiration
const KEEP_ALIVE_INTERVAL_MS = 5 * 60 * 1000

interface SessionState {
  sessionId: string | null
  isInitialized: boolean
  isLoading: boolean
  error: string | null
  keepAliveTimer: ReturnType<typeof setInterval> | null

  // Actions
  initSession: () => Promise<void>
  clearSession: () => void
  startKeepAlive: () => void
  stopKeepAlive: () => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  isInitialized: false,
  isLoading: false,
  error: null,
  keepAliveTimer: null,

  startKeepAlive: () => {
    const { keepAliveTimer, sessionId } = get()

    // Clear existing timer if any
    if (keepAliveTimer) {
      clearInterval(keepAliveTimer)
    }

    if (!sessionId) return

    const timer = setInterval(async () => {
      const currentSessionId = get().sessionId
      if (!currentSessionId) {
        get().stopKeepAlive()
        return
      }

      try {
        const response = await api.getSession(currentSessionId)
        if (!response.success) {
          console.warn('[Session] Keep-alive failed, session may have expired')
          // Session expired - clear local state
          get().stopKeepAlive()
          set({
            error: 'Session expired. Please refresh the page to start a new session.',
          })
        }
      } catch (err) {
        console.warn('[Session] Keep-alive error:', err)
      }
    }, KEEP_ALIVE_INTERVAL_MS)

    set({ keepAliveTimer: timer })
    console.log('[Session] Keep-alive started (interval: 5 minutes)')
  },

  stopKeepAlive: () => {
    const { keepAliveTimer } = get()
    if (keepAliveTimer) {
      clearInterval(keepAliveTimer)
      set({ keepAliveTimer: null })
      console.log('[Session] Keep-alive stopped')
    }
  },

  initSession: async () => {
    if (get().isLoading) return

    // Stop any existing keep-alive
    get().stopKeepAlive()

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
        // Start keep-alive after session is created
        get().startKeepAlive()
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
    get().stopKeepAlive()
    api.setSessionId('')
    set({
      sessionId: null,
      isInitialized: false,
      error: null,
    })
  },
}))
