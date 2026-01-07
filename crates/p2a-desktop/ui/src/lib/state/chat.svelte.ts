// Chat state management using Svelte 5 runes

import type { Message } from '$lib/types';

class ChatState {
	messages = $state<Message[]>([]);
	input = $state('');
	isProcessing = $state(false);

	addMessage(role: Message['role'], content: string, images?: string[]) {
		this.messages.push({
			id: crypto.randomUUID(),
			role,
			content,
			images,
			timestamp: new Date()
		});
	}

	addUserMessage(content: string) {
		this.addMessage('user', content);
	}

	addAssistantMessage(content: string, images?: string[]) {
		this.addMessage('assistant', content, images);
	}

	addErrorMessage(content: string) {
		this.addMessage('error', content);
	}

	addSystemMessage(content: string) {
		this.addMessage('system', content);
	}

	clearInput() {
		this.input = '';
	}

	setInput(value: string) {
		this.input = value;
	}

	setProcessing(value: boolean) {
		this.isProcessing = value;
	}

	clearMessages() {
		this.messages = [];
	}
}

export const chatState = new ChatState();
