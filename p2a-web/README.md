# p2a-web

React/Next.js frontend for prompt2analytics - a natural language data analytics platform.

## Features

- **Chat Interface**: Natural language interaction with 55+ analytics tools
- **Real-time Streaming**: WebSocket-based response streaming
- **Dataset Management**: Upload CSV, JSON, or Parquet files via drag-and-drop
- **Results Panel**: View regression outputs, summaries, and charts
- **LLM Integration**: Support for Ollama (local), Anthropic, and OpenAI
- **Theme Support**: Light, dark, and system-preference themes

## Prerequisites

- Node.js 20+
- npm or yarn
- p2a-mcp backend running with HTTP transport

## Quick Start

### 1. Start the Backend

```bash
cd crates/p2a-mcp
cargo run --features full -- --transport http --port 8080
```

### 2. Start the Frontend

```bash
cd p2a-web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
p2a-web/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout with providers
│   ├── page.tsx           # Main chat interface
│   ├── globals.css        # TailwindCSS styles
│   └── settings/          # Settings page
├── components/
│   ├── chat/              # Chat UI components
│   │   ├── ChatPanel.tsx
│   │   ├── ChatInput.tsx
│   │   ├── Message.tsx
│   │   ├── MessageList.tsx
│   │   ├── StreamingIndicator.tsx
│   │   └── ToolCall.tsx
│   ├── data/              # Dataset management
│   │   └── DataPanel.tsx
│   ├── results/           # Results display
│   │   └── ResultsPanel.tsx
│   ├── layout/            # Layout components
│   │   └── ThreeColumnLayout.tsx
│   └── providers/         # React context providers
│       └── ThemeProvider.tsx
├── lib/
│   ├── api/               # API clients
│   │   ├── client.ts      # HTTP API client
│   │   └── websocket.ts   # WebSocket streaming client
│   ├── store/             # Zustand state stores
│   │   ├── chat-store.ts
│   │   ├── datasets-store.ts
│   │   ├── results-store.ts
│   │   ├── session-store.ts
│   │   └── settings-store.ts
│   ├── hooks/             # Custom React hooks
│   │   └── useStreaming.ts
│   └── types/             # TypeScript types
│       └── api.ts
├── e2e/                   # Playwright E2E tests
│   └── app.spec.ts
└── playwright.config.ts   # Playwright configuration
```

## Available Scripts

```bash
# Development
npm run dev          # Start development server

# Build
npm run build        # Build for production
npm run start        # Start production server

# Testing
npm run test         # Run Playwright E2E tests
npm run test:ui      # Run tests with UI
npm run test:headed  # Run tests in headed mode

# Code Quality
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript type checker
```

## Configuration

### Environment Variables

Create a `.env.local` file:

```env
# Backend API URL (default: http://localhost:8080)
NEXT_PUBLIC_API_URL=http://localhost:8080
```

### LLM Settings

Configure LLM providers in the Settings page (`/settings`):

- **Ollama (Local)**: Default provider, requires Ollama running locally
- **Anthropic**: Requires API key (Claude models)
- **OpenAI**: Requires API key (GPT models)

## Architecture

### Frontend Stack

- **Framework**: Next.js 15 with App Router
- **UI**: React 19 with TailwindCSS v4
- **State**: Zustand with immer middleware
- **Markdown**: react-markdown with remark-gfm
- **Testing**: Playwright

### Communication

```
┌─────────────────┐     HTTP/REST      ┌─────────────────┐
│   p2a-web       │ ◄─────────────────► │   p2a-mcp       │
│   (Next.js)     │                     │   (Rust/Axum)   │
│                 │     WebSocket       │                 │
│   Port 3000     │ ◄─────────────────► │   Port 8080     │
└─────────────────┘                     └─────────────────┘
```

- **HTTP**: Session management, tool discovery, non-streaming chat
- **WebSocket**: Real-time streaming responses, tool execution updates

### State Management

| Store | Purpose |
|-------|---------|
| `session-store` | Backend session management |
| `chat-store` | Messages, streaming state |
| `datasets-store` | Loaded datasets, previews |
| `results-store` | Analysis results |
| `settings-store` | LLM config, theme (persisted) |

## Usage

### 1. Upload a Dataset

Drag and drop a CSV file onto the data panel, or click to browse.

### 2. Ask Questions

Type natural language queries in the chat input:

```
"Describe the dataset"
"Run an OLS regression of price on sqft and bedrooms"
"Create a histogram of the income column"
"Calculate summary statistics for all numeric columns"
```

### 3. View Results

Results appear in the right panel with:
- Expandable sections for detailed output
- Charts rendered inline
- Tool execution history

## Development

### Adding New Components

1. Create component in appropriate `components/` subdirectory
2. Use `'use client'` directive for client components
3. Import from Zustand stores as needed
4. Follow existing patterns for styling (TailwindCSS)

### Adding New API Endpoints

1. Add types to `lib/types/api.ts`
2. Add method to `lib/api/client.ts`
3. Use in components via stores or directly

### Running Tests

```bash
# Install Playwright browsers (first time)
npx playwright install

# Run all tests
npm run test

# Run specific test file
npx playwright test e2e/app.spec.ts

# Debug tests
npx playwright test --debug
```

## Troubleshooting

### "Connecting to analytics server..." stuck

- Ensure p2a-mcp is running: `cargo run --features full -- --transport http`
- Check port 8080 is not in use
- Verify CORS is enabled (default in dev)

### WebSocket connection failed

- Backend must support WebSocket (`--features websocket` or `--features full`)
- Check for proxy/firewall blocking WS connections

### Theme not applying

- Clear localStorage and refresh
- Check browser console for errors

## License

MIT - See parent project license.
