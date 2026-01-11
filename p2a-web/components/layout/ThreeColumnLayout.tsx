'use client'

import { ReactNode, useState } from 'react'
import Link from 'next/link'

interface ThreeColumnLayoutProps {
  left: ReactNode
  center: ReactNode
  right: ReactNode
}

export function ThreeColumnLayout({ left, center, right }: ThreeColumnLayoutProps) {
  const [leftCollapsed, setLeftCollapsed] = useState(false)
  const [rightCollapsed, setRightCollapsed] = useState(false)

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="h-14 border-b flex items-center justify-between px-4 bg-white dark:bg-gray-900">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">prompt2analytics</h1>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setLeftCollapsed(!leftCollapsed)}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
            title={leftCollapsed ? 'Show data panel' : 'Hide data panel'}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <button
            onClick={() => setRightCollapsed(!rightCollapsed)}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
            title={rightCollapsed ? 'Show results panel' : 'Hide results panel'}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
          </button>
          <div className="w-px h-6 bg-gray-200 dark:bg-gray-700 mx-1" />
          <Link
            href="/settings"
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
            title="Settings"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </Link>
        </div>
      </header>

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel - Data */}
        <aside
          className={`${
            leftCollapsed ? 'w-0' : 'w-72'
          } border-r bg-gray-50 dark:bg-gray-900 transition-all duration-300 overflow-hidden`}
        >
          <div className="h-full overflow-y-auto">
            {left}
          </div>
        </aside>

        {/* Center panel - Chat */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {center}
        </main>

        {/* Right panel - Results */}
        <aside
          className={`${
            rightCollapsed ? 'w-0' : 'w-96'
          } border-l bg-gray-50 dark:bg-gray-900 transition-all duration-300 overflow-hidden`}
        >
          <div className="h-full overflow-y-auto">
            {right}
          </div>
        </aside>
      </div>
    </div>
  )
}
