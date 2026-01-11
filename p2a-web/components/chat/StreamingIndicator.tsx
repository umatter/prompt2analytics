'use client'

export function StreamingIndicator() {
  return (
    <div className="flex items-center gap-1">
      <span className="animate-pulse">
        <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
      </span>
      <span className="animate-pulse">
        <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
      </span>
      <span className="animate-pulse">
        <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
      </span>
    </div>
  )
}
