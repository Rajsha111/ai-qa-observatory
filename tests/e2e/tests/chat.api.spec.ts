import {expect, test} from "@playwright/test";

test('Post chat - API Contract Test', async ({request}) => {
    const start = Date.now();
    const response = await request.post('/chat', {
        data: {
            messages: [
                { role: 'user', content: 'Say hello in one word.' }
            ],
        }
    });
    const latency = Date.now() - start;
    test.expect(response.status()).toBe(200);
    const data = await response.json();
    expect(data).toHaveProperty('type', 'done');
    console.log('API Response:', data);
    expect(latency).toBeLessThan(10_000);
});   