import {Locator, Page} from "@playwright/test";

export class ChatPage {
    page: Page;
    private readonly chatInput : Locator;
    private readonly sendButton : Locator;
    private readonly aiChatTitle : Locator;

    constructor(page: Page) {
        this.page = page;
        this.chatInput = this.page.getByRole('textbox', {name: 'Send a message… (Enter to send, Shift+Enter for newline)'});
        this.sendButton = this.page.getByRole('button', {name: 'Send'});
        this.aiChatTitle = this.page.getByText('AI QA Observatory');
    }

    async enterMessage(message: string) {
        await this.chatInput.fill(message);
    }

    async clickSend() {
        await this.sendButton.click();
    }

    async getAIChatTitle() {
        return await this.aiChatTitle.textContent();
    }

    async getLastMessage() {
        const messages = this.page.locator('//*[@id="chat"]//div[@class="msg user"]');
        const count = await messages.count();
        if (count > 0) {
            return await messages.nth(count - 1).textContent();
        }
        return null;
    }

    async getLastAssistantMessage() {
        const messages = this.page.locator('//*[@id="chat"]//div[@class="msg assistant"]');
        const count = await messages.count();
        if (count > 0) {
            return await messages.nth(count - 1).textContent();
        }
        return null;
    }

    async getTypingIndicator() {
        const typingIndicator = this.page.locator('.typing');
        if (await typingIndicator.isVisible()) {
            return await typingIndicator.textContent();
        }
        return null;
    }

    async waitForLastAssistantMessage() {
        const messages = this.page.locator('//*[@id="chat"]//div[@class="msg assistant"]');
        await messages.last().waitFor();
    }

    async waitForResponse() {
        await this.page.waitForSelector('.typing');
    }

    async waitForStreamComplete(timeout = 60000) {
        try {
            await this.page.locator('.msg.assistant').last().waitFor({
                state: 'visible',
                timeout,
            });

            await this.page.waitForFunction(
                () => document.querySelectorAll('.msg.assistant.streaming').length === 0,
                { timeout }
            );
            return true;
        } catch (e) {
            return false;
        }
    }
}


