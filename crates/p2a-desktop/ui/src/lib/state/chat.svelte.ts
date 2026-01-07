// Chat state management using Svelte 5 runes

import type { Message } from '$lib/types';
import type { Conversation, StreamChunk, ToolCall, LlmToolResult } from '$lib/types/llm';

/** Extended message with LLM-specific fields */
export interface ChatMessage extends Message {
	toolCalls?: ToolCall[];
	toolResults?: LlmToolResult[];
	isStreaming?: boolean;
}

class ChatState {
	// Message state
	messages = $state<ChatMessage[]>([]);
	input = $state('');
	isProcessing = $state(false);

	// LLM conversation state
	currentConversationId = $state<string | null>(null);
	conversations = $state<Conversation[]>([]);
	isLlmMode = $state(true); // Default to LLM mode

	// Streaming state
	streamingContent = $state('');
	pendingToolCalls = $state<ToolCall[]>([]);

	// ========== Message Methods ==========

	addMessage(
		role: Message['role'],
		content: string,
		images?: string[],
		toolCalls?: ToolCall[],
		toolResults?: LlmToolResult[]
	) {
		this.messages.push({
			id: crypto.randomUUID(),
			role,
			content,
			images,
			timestamp: new Date(),
			toolCalls,
			toolResults
		});
	}

	addUserMessage(content: string) {
		this.addMessage('user', content);
	}

	addAssistantMessage(content: string, images?: string[], toolCalls?: ToolCall[]) {
		this.addMessage('assistant', content, images, toolCalls);
	}

	addErrorMessage(content: string) {
		this.addMessage('error', content);
	}

	addSystemMessage(content: string) {
		this.addMessage('system', content);
	}

	/** Add a streaming message placeholder */
	startStreamingMessage() {
		this.streamingContent = '';
		this.pendingToolCalls = [];
		this.messages.push({
			id: crypto.randomUUID(),
			role: 'assistant',
			content: '',
			timestamp: new Date(),
			isStreaming: true
		});
	}

	/** Update the streaming message content */
	updateStreamingContent(content: string) {
		this.streamingContent += content;
		// Update the last message if it's streaming
		const lastMsg = this.messages[this.messages.length - 1];
		if (lastMsg && lastMsg.isStreaming) {
			lastMsg.content = this.streamingContent;
		}
	}

	/** Add a tool call to pending */
	addPendingToolCall(toolCall: ToolCall) {
		this.pendingToolCalls.push(toolCall);
		// Update the last message
		const lastMsg = this.messages[this.messages.length - 1];
		if (lastMsg && lastMsg.isStreaming) {
			lastMsg.toolCalls = [...this.pendingToolCalls];
		}
	}

	/** Add tool result message */
	addToolResultMessage(toolResult: LlmToolResult) {
		this.addMessage('system', `Tool result: ${toolResult.content}`, undefined, undefined, [
			toolResult
		]);
	}

	/** Finalize the streaming message */
	finalizeStreamingMessage(finalContent?: string, toolCalls?: ToolCall[]) {
		const lastMsg = this.messages[this.messages.length - 1];
		if (lastMsg && lastMsg.isStreaming) {
			lastMsg.isStreaming = false;
			if (finalContent !== undefined) {
				lastMsg.content = finalContent;
			}
			if (toolCalls) {
				lastMsg.toolCalls = toolCalls;
			}
		}
		this.streamingContent = '';
		this.pendingToolCalls = [];
	}

	/** Handle stream chunk */
	handleStreamChunk(chunk: StreamChunk) {
		switch (chunk.type) {
			case 'text':
				this.updateStreamingContent(chunk.content);
				break;
			case 'tool_call':
				this.addPendingToolCall(chunk.tool_call);
				break;
			case 'tool_result':
				// Tool results are shown inline
				break;
			case 'done':
				this.finalizeStreamingMessage();
				break;
			case 'error':
				this.finalizeStreamingMessage();
				this.addErrorMessage(chunk.message);
				break;
		}
	}

	// ========== Input Methods ==========

	clearInput() {
		this.input = '';
	}

	setInput(value: string) {
		this.input = value;
	}

	setProcessing(value: boolean) {
		this.isProcessing = value;
	}

	// ========== Message Management ==========

	clearMessages() {
		this.messages = [];
	}

	/** Get the last N messages for context */
	getRecentMessages(count: number = 10): ChatMessage[] {
		return this.messages.slice(-count);
	}

	// ========== Conversation Methods ==========

	setConversationId(id: string | null) {
		this.currentConversationId = id;
	}

	setConversations(conversations: Conversation[]) {
		this.conversations = conversations;
	}

	addConversation(conversation: Conversation) {
		this.conversations.unshift(conversation);
	}

	removeConversation(id: string) {
		this.conversations = this.conversations.filter((c) => c.id !== id);
		if (this.currentConversationId === id) {
			this.currentConversationId = null;
			this.clearMessages();
		}
	}

	updateConversationTitle(id: string, newTitle: string) {
		const conv = this.conversations.find((c) => c.id === id);
		if (conv) {
			conv.title = newTitle;
		}
	}

	/** Start a new conversation */
	startNewConversation() {
		this.currentConversationId = null;
		this.clearMessages();
	}

	// ========== Mode Methods ==========

	setLlmMode(enabled: boolean) {
		this.isLlmMode = enabled;
	}

	toggleLlmMode() {
		this.isLlmMode = !this.isLlmMode;
	}
}

export const chatState = new ChatState();
