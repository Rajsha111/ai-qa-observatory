import { afterAll, describe, expect, it } from "vitest";
import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import {analyzeQualityInputSchema} from "../../src/mcp/analyze-quality-tool.js";

describe("analyzeQualityInputSchema input schema contract", () => {
    it('is an object type', () => {
        expect(analyzeQualityInputSchema.type).toBe("object");
    });

    it('has required properties', () => {
        expect(analyzeQualityInputSchema.required).toEqual(["llm_response", "groundTruth", "criteria"]);
    });

    it('llm_response and groundTruth are strings', () => {
        expect(analyzeQualityInputSchema.properties.llm_response.type).toBe("string");
        expect(analyzeQualityInputSchema.properties.groundTruth.type).toBe("string");
    });
    
    it('criteria is an array of strings', () => {
        expect(analyzeQualityInputSchema.properties.criteria.type).toBe("array");
        expect(analyzeQualityInputSchema.properties.criteria.items.type).toBe("string");
    });

    it('mark all three properties as required', () => {
        expect(analyzeQualityInputSchema.required).toContain("llm_response");
        expect(analyzeQualityInputSchema.required).toContain("groundTruth");
        expect(analyzeQualityInputSchema.required).toContain("criteria");
    });
});

afterAll(() => {
  mkdirSync(join(__dirname, 'artifacts'), { recursive: true });
  writeFileSync(
    join(__dirname, 'artifacts', 'schema.json'),
    JSON.stringify({ schema: analyzeQualityInputSchema }, null, 2)
  );
});