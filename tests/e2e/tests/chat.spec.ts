import {test, expect} from "@playwright/test";

import {ChatPage} from "../pages/chatPage";

test.describe('Chat Page', () => {
    let chatPage: ChatPage;

    test("should display AI QA Observatory title", async ({page}) => {

        chatPage = new ChatPage(page);
        await page.goto('/');
        
        const title = await chatPage.getAIChatTitle();
        test.expect(title).toBe('AI QA Observatory');
    });


    test("should send a message and receive a response", async ({page}) => {

        chatPage = new ChatPage(page);
        await page.goto('/');

        await chatPage.enterMessage('Hello, how are you?');
        await chatPage.clickSend();

        await expect(await chatPage.getLastMessage()).toBe('Hello, how are you?');
        await expect(await chatPage.waitForStreamComplete()).toBeTruthy();
        await chatPage.waitForLastAssistantMessage();
        const assistantMessage = await chatPage.getLastAssistantMessage();
        console.log('Assistant Message:', assistantMessage);
        test.expect(assistantMessage).not.toBeNull();
    });
});