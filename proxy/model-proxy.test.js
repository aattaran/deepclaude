import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    MODEL_REMAP,
    PRICING_PER_M,
    NON_ANTHROPIC_BACKENDS,
    ANTHROPIC_THINKING_BACKENDS,
    isHaikuModel,
    resolveAutoBackend,
    stripAllThinkingBlocks,
    stripUnsignedThinkingBlocks,
    stripDeepSeekUnsupportedFields,
    mapEffortToReasoning,
    normalizeThinkingBlocks,
    stripSamplingParamsOnThinking,
    normalizeOutputConfig,
    filterAssistantPrefill,
    warnOAuthToken,
} from './model-proxy.js';

// ---------------------------------------------------------------------------
// 1. Model remapping
// ---------------------------------------------------------------------------
describe('MODEL_REMAP', () => {
    it('has entries for gemini, deepseek, and openrouter', () => {
        assert.ok(MODEL_REMAP.gemini);
        assert.ok(MODEL_REMAP.deepseek);
        assert.ok(MODEL_REMAP.openrouter);
    });

    describe('gemini backend', () => {
        it('maps every claude model to gemini-3.5-flash', () => {
            for (const [claude, target] of Object.entries(MODEL_REMAP.gemini)) {
                assert.equal(target, 'gemini-3.5-flash', `${claude} should map to gemini-3.5-flash`);
            }
        });

        it('maps claude-haiku-4-5-20251001 to gemini-3.5-flash', () => {
            assert.equal(MODEL_REMAP.gemini['claude-haiku-4-5-20251001'], 'gemini-3.5-flash');
        });

        it('maps claude-opus-4-6 and -4-7 to gemini-3.5-flash', () => {
            assert.equal(MODEL_REMAP.gemini['claude-opus-4-6'], 'gemini-3.5-flash');
            assert.equal(MODEL_REMAP.gemini['claude-opus-4-7'], 'gemini-3.5-flash');
        });
    });

    describe('deepseek backend', () => {
        it('maps opus models to deepseek-v4-pro', () => {
            assert.equal(MODEL_REMAP.deepseek['claude-opus-4-6'], 'deepseek-v4-pro');
            assert.equal(MODEL_REMAP.deepseek['claude-opus-4-7'], 'deepseek-v4-pro');
        });

        it('maps sonnet and haiku models to deepseek-v4-flash', () => {
            assert.equal(MODEL_REMAP.deepseek['claude-sonnet-4-6'], 'deepseek-v4-flash');
            assert.equal(MODEL_REMAP.deepseek['claude-sonnet-4-5-20250929'], 'deepseek-v4-flash');
            assert.equal(MODEL_REMAP.deepseek['claude-haiku-4-5-20251001'], 'deepseek-v4-flash');
        });
    });

    describe('openrouter backend', () => {
        it('uses deepseek/ prefix on all targets', () => {
            for (const [claude, target] of Object.entries(MODEL_REMAP.openrouter)) {
                assert.ok(target.startsWith('deepseek/'), `${claude}: ${target} should start with deepseek/`);
            }
        });

        it('maps opus models to deepseek/deepseek-v4-pro', () => {
            assert.equal(MODEL_REMAP.openrouter['claude-opus-4-6'], 'deepseek/deepseek-v4-pro');
            assert.equal(MODEL_REMAP.openrouter['claude-opus-4-7'], 'deepseek/deepseek-v4-pro');
        });

        it('maps sonnet and haiku models to deepseek/deepseek-v4-flash', () => {
            assert.equal(MODEL_REMAP.openrouter['claude-sonnet-4-6'], 'deepseek/deepseek-v4-flash');
            assert.equal(MODEL_REMAP.openrouter['claude-sonnet-4-5-20250929'], 'deepseek/deepseek-v4-flash');
            assert.equal(MODEL_REMAP.openrouter['claude-haiku-4-5-20251001'], 'deepseek/deepseek-v4-flash');
        });
    });
});

// ---------------------------------------------------------------------------
// 2. Auto-routing
// ---------------------------------------------------------------------------
describe('isHaikuModel', () => {
    it('returns true for claude-haiku-4-5-20251001', () => {
        assert.equal(isHaikuModel('claude-haiku-4-5-20251001'), true);
    });

    it('returns true for bare claude-haiku prefix', () => {
        assert.equal(isHaikuModel('claude-haiku'), true);
    });

    it('returns false for a sonnet model', () => {
        assert.equal(isHaikuModel('claude-sonnet-4-6'), false);
        assert.equal(isHaikuModel('claude-sonnet-4-5-20250929'), false);
    });

    it('returns false for an opus model', () => {
        assert.equal(isHaikuModel('claude-opus-4-6'), false);
        assert.equal(isHaikuModel('claude-opus-4-7'), false);
    });

    it('returns false for empty string', () => {
        assert.equal(isHaikuModel(''), false);
    });

    it('returns false for null', () => {
        assert.equal(isHaikuModel(null), false);
    });

    it('returns false for undefined', () => {
        assert.equal(isHaikuModel(undefined), false);
    });

    it('returns false for a model string without claude prefix', () => {
        assert.equal(isHaikuModel('gpt-4o'), false);
    });
});

describe('resolveAutoBackend', () => {
    const mockBackends = {
        deepseek: { target: new URL('https://api.deepseek.com'), apiKey: 'ds-key', useBearer: false },
        gemini:   { target: new URL('https://generativelanguage.googleapis.com'), apiKey: 'gem-key', useBearer: false },
    };

    const backendsNoKeys = {
        deepseek: { target: new URL('https://api.deepseek.com'), apiKey: null, useBearer: false },
        gemini:   { target: new URL('https://generativelanguage.googleapis.com'), apiKey: '', useBearer: false },
    };

    const backendsOnlyDeepseek = {
        deepseek: { target: new URL('https://api.deepseek.com'), apiKey: 'ds-key', useBearer: false },
        gemini:   { target: new URL('https://generativelanguage.googleapis.com'), apiKey: null, useBearer: false },
    };

    const backendsOnlyGemini = {
        deepseek: { target: new URL('https://api.deepseek.com'), apiKey: null, useBearer: false },
        gemini:   { target: new URL('https://generativelanguage.googleapis.com'), apiKey: 'gem-key', useBearer: false },
    };

    // --- Both backends available ---

    it('routes haiku tier to gemini primary, deepseek fallback', () => {
        const result = resolveAutoBackend('claude-haiku-4-5-20251001', mockBackends);
        assert.ok(result);
        assert.equal(result.tier, 'haiku');
        assert.equal(result.ctx.name, 'gemini');
        assert.equal(result.ctx.model, 'gemini-3.5-flash');
        assert.equal(result.ctx.isNonAnthropic, true);
        assert.ok(result.fallback);
        assert.equal(result.fallback.name, 'deepseek');
        assert.equal(result.fallback.model, 'deepseek-v4-flash');
        assert.equal(result.fallback.isNonAnthropic, false);
    });

    it('routes sonnet tier to deepseek primary, gemini fallback', () => {
        const result = resolveAutoBackend('claude-sonnet-4-6', mockBackends);
        assert.ok(result);
        assert.equal(result.tier, 'sonnet_opus');
        assert.equal(result.ctx.name, 'deepseek');
        assert.equal(result.ctx.model, 'deepseek-v4-flash');
        assert.equal(result.ctx.isNonAnthropic, false);
        assert.ok(result.fallback);
        assert.equal(result.fallback.name, 'gemini');
        assert.equal(result.fallback.model, 'gemini-3.5-flash');
        assert.equal(result.fallback.isNonAnthropic, true);
    });

    it('routes opus tier to deepseek primary, gemini fallback', () => {
        const result = resolveAutoBackend('claude-opus-4-6', mockBackends);
        assert.ok(result);
        assert.equal(result.tier, 'sonnet_opus');
        assert.equal(result.ctx.name, 'deepseek');
        assert.equal(result.ctx.model, 'deepseek-v4-pro');
        assert.ok(result.fallback);
        assert.equal(result.fallback.name, 'gemini');
    });

    // --- Missing API keys ---

    it('returns null when neither backend has an API key', () => {
        const result = resolveAutoBackend('claude-haiku-4-5-20251001', backendsNoKeys);
        assert.equal(result, null);
    });

    it('uses primary only when fallback has no API key', () => {
        const result = resolveAutoBackend('claude-haiku-4-5-20251001', backendsOnlyGemini);
        assert.ok(result);
        assert.equal(result.ctx.name, 'gemini');
        assert.equal(result.fallback, null);
    });

    it('promotes fallback to primary when primary has no API key', () => {
        const result = resolveAutoBackend('claude-haiku-4-5-20251001', backendsOnlyDeepseek);
        assert.ok(result);
        assert.equal(result.ctx.name, 'deepseek');
        assert.equal(result.ctx.model, 'deepseek-v4-flash');
        assert.equal(result.fallback, null);
    });

    it('uses fallbackModel when promoting fallback and model is not in MODEL_REMAP', () => {
        // haiku-tier name not in MODEL_REMAP.deepseek, only deepseek available
        const result = resolveAutoBackend('claude-haiku-nonexistent', backendsOnlyDeepseek);
        assert.ok(result);
        assert.equal(result.ctx.name, 'deepseek');
        // claude-haiku-nonexistent is not in MODEL_REMAP.deepseek, so it falls
        // back to route.fallbackModel which is 'deepseek-v4-flash'
        assert.equal(result.ctx.model, 'deepseek-v4-flash');
    });

    it('preserves unknown model name when primary is available and model not in MODEL_REMAP', () => {
        // sonnet/opus tier with both backends, primary=deepseek
        const result = resolveAutoBackend('unknown-model-v1', mockBackends);
        assert.ok(result);
        assert.equal(result.ctx.name, 'deepseek');
        // MODEL_REMAP.deepseek['unknown-model-v1'] is undefined, so remap = original model
        assert.equal(result.ctx.model, 'unknown-model-v1');
    });

    it('context carries through target, apiKey, and useBearer from backends config', () => {
        const result = resolveAutoBackend('claude-sonnet-4-6', mockBackends);
        assert.equal(result.ctx.apiKey, 'ds-key');
        assert.equal(result.ctx.target.hostname, 'api.deepseek.com');
        assert.equal(result.ctx.useBearer, false);
    });
});

// ---------------------------------------------------------------------------
// 3. Pricing
// ---------------------------------------------------------------------------
describe('PRICING_PER_M', () => {
    it('has entries for all expected backends', () => {
        assert.ok(PRICING_PER_M.deepseek);
        assert.ok(PRICING_PER_M.openrouter);
        assert.ok(PRICING_PER_M.fireworks);
        assert.ok(PRICING_PER_M.gemini);
        assert.ok(PRICING_PER_M.anthropic);
        assert.ok(PRICING_PER_M._single);
    });

    it('deepseek and openrouter share identical rates', () => {
        assert.equal(PRICING_PER_M.deepseek.input, PRICING_PER_M.openrouter.input);
        assert.equal(PRICING_PER_M.deepseek.output, PRICING_PER_M.openrouter.output);
    });

    it('has correct deepseek rates', () => {
        assert.equal(PRICING_PER_M.deepseek.input, 0.44);
        assert.equal(PRICING_PER_M.deepseek.output, 0.87);
    });

    it('has correct gemini rates', () => {
        assert.equal(PRICING_PER_M.gemini.input, 1.50);
        assert.equal(PRICING_PER_M.gemini.output, 9.00);
    });

    it('has correct anthropic rates', () => {
        assert.equal(PRICING_PER_M.anthropic.input, 3.00);
        assert.equal(PRICING_PER_M.anthropic.output, 15.00);
    });

    it('has correct fireworks rates', () => {
        assert.equal(PRICING_PER_M.fireworks.input, 1.74);
        assert.equal(PRICING_PER_M.fireworks.output, 3.48);
    });

    it('_single matches deepseek rates', () => {
        assert.equal(PRICING_PER_M._single.input, PRICING_PER_M.deepseek.input);
        assert.equal(PRICING_PER_M._single.output, PRICING_PER_M.deepseek.output);
    });
});

// ---------------------------------------------------------------------------
// 4. NON_ANTHROPIC_BACKENDS
// ---------------------------------------------------------------------------
describe('NON_ANTHROPIC_BACKENDS', () => {
    it('contains exactly one entry', () => {
        assert.equal(NON_ANTHROPIC_BACKENDS.size, 1);
    });

    it('includes gemini', () => {
        assert.ok(NON_ANTHROPIC_BACKENDS.has('gemini'));
    });

    it('does NOT include deepseek', () => {
        assert.ok(!NON_ANTHROPIC_BACKENDS.has('deepseek'));
    });

    it('does NOT include openrouter', () => {
        assert.ok(!NON_ANTHROPIC_BACKENDS.has('openrouter'));
    });

    it('does NOT include anthropic', () => {
        assert.ok(!NON_ANTHROPIC_BACKENDS.has('anthropic'));
    });

    it('does NOT include fireworks', () => {
        assert.ok(!NON_ANTHROPIC_BACKENDS.has('fireworks'));
    });
});

// ---------------------------------------------------------------------------
// 4b. ANTHROPIC_THINKING_BACKENDS
// ---------------------------------------------------------------------------
describe('ANTHROPIC_THINKING_BACKENDS', () => {
    it('contains exactly two entries', () => {
        assert.equal(ANTHROPIC_THINKING_BACKENDS.size, 2);
    });

    it('includes deepseek', () => {
        assert.ok(ANTHROPIC_THINKING_BACKENDS.has('deepseek'));
    });

    it('includes anthropic', () => {
        assert.ok(ANTHROPIC_THINKING_BACKENDS.has('anthropic'));
    });

    it('does NOT include gemini', () => {
        assert.ok(!ANTHROPIC_THINKING_BACKENDS.has('gemini'));
    });

    it('does NOT include openrouter', () => {
        assert.ok(!ANTHROPIC_THINKING_BACKENDS.has('openrouter'));
    });

    it('does NOT include fireworks', () => {
        assert.ok(!ANTHROPIC_THINKING_BACKENDS.has('fireworks'));
    });
});

// ---------------------------------------------------------------------------
// 5. Thread sanitization
// ---------------------------------------------------------------------------
describe('stripAllThinkingBlocks', () => {
    it('removes all thinking blocks from assistant messages', () => {
        const body = {
            messages: [
                { role: 'user', content: [{ type: 'text', text: 'hello' }] },
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'deep thought' },
                        { type: 'text', text: 'world' },
                        { type: 'thinking', thinking: 'more thought' },
                    ],
                },
            ],
        };
        stripAllThinkingBlocks(body);
        assert.equal(body.messages[1].content.length, 1);
        assert.equal(body.messages[1].content[0].type, 'text');
        assert.equal(body.messages[1].content[0].text, 'world');
    });

    it('preserves text blocks when no thinking blocks exist', () => {
        const body = {
            messages: [
                {
                    role: 'assistant',
                    content: [
                        { type: 'text', text: 'only' },
                        { type: 'text', text: 'text' },
                    ],
                },
            ],
        };
        stripAllThinkingBlocks(body);
        assert.equal(body.messages[0].content.length, 2);
    });

    it('handles null body without throwing', () => {
        stripAllThinkingBlocks(null);
        stripAllThinkingBlocks(undefined);
    });

    it('handles body without messages key', () => {
        const body = { model: 'test' };
        stripAllThinkingBlocks(body);
        assert.deepEqual(body, { model: 'test' });
    });

    it('handles messages with string content (not array)', () => {
        const body = { messages: [{ role: 'user', content: 'plain string' }] };
        stripAllThinkingBlocks(body);
        assert.equal(body.messages[0].content, 'plain string');
    });

    it('handles empty messages array', () => {
        const body = { messages: [] };
        stripAllThinkingBlocks(body);
        assert.deepEqual(body, { messages: [] });
    });

    it('handles messages with empty content array', () => {
        const body = { messages: [{ role: 'assistant', content: [] }] };
        stripAllThinkingBlocks(body);
        assert.deepEqual(body.messages[0].content, []);
    });
});

describe('stripUnsignedThinkingBlocks', () => {
    it('removes thinking blocks without signature', () => {
        const body = {
            messages: [
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'unsigned' },
                        { type: 'text', text: 'response' },
                    ],
                },
            ],
        };
        stripUnsignedThinkingBlocks(body);
        assert.equal(body.messages[0].content.length, 1);
        assert.equal(body.messages[0].content[0].type, 'text');
    });

    it('keeps thinking blocks that have a signature', () => {
        const body = {
            messages: [
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'signed', signature: 'abc123' },
                        { type: 'text', text: 'response' },
                    ],
                },
            ],
        };
        stripUnsignedThinkingBlocks(body);
        assert.equal(body.messages[0].content.length, 2);
    });

    it('removes unsigned blocks and keeps signed blocks in mixed content', () => {
        const body = {
            messages: [
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'no sig' },
                        { type: 'thinking', thinking: 'has sig', signature: 'sig-xyz' },
                        { type: 'text', text: 'hello' },
                        { type: 'thinking', thinking: 'also no sig' },
                    ],
                },
            ],
        };
        stripUnsignedThinkingBlocks(body);
        assert.equal(body.messages[0].content.length, 2);
        assert.equal(body.messages[0].content[0].type, 'thinking');
        assert.equal(body.messages[0].content[0].signature, 'sig-xyz');
        assert.equal(body.messages[0].content[1].type, 'text');
    });

    it('handles null body without throwing', () => {
        stripUnsignedThinkingBlocks(null);
        stripUnsignedThinkingBlocks(undefined);
    });

    it('handles body without messages key', () => {
        const body = { model: 'test' };
        stripUnsignedThinkingBlocks(body);
        assert.deepEqual(body, { model: 'test' });
    });

    it('handles messages with string content (not array)', () => {
        const body = { messages: [{ role: 'user', content: 'plain string' }] };
        stripUnsignedThinkingBlocks(body);
        assert.equal(body.messages[0].content, 'plain string');
    });
});

// ---------------------------------------------------------------------------
// 6. Anthropic API compatibility: normalizeThinkingBlocks
// ---------------------------------------------------------------------------
describe('normalizeThinkingBlocks', () => {
    it('converts old budget_tokens ≤1024 to adaptive + effort=low', () => {
        const body = { thinking: { type: 'enabled', budget_tokens: 512 } };
        normalizeThinkingBlocks(body);
        assert.deepEqual(body.thinking, { type: 'adaptive' });
        assert.equal(body.effort, 'low');
    });

    it('maps budget_tokens ≤4096 to effort=medium', () => {
        const body = { thinking: { type: 'enabled', budget_tokens: 2048 } };
        normalizeThinkingBlocks(body);
        assert.equal(body.effort, 'medium');
    });

    it('maps budget_tokens ≤8192 to effort=high', () => {
        const body = { thinking: { type: 'enabled', budget_tokens: 6000 } };
        normalizeThinkingBlocks(body);
        assert.equal(body.effort, 'high');
    });

    it('maps budget_tokens >8192 to effort=max', () => {
        const body = { thinking: { type: 'enabled', budget_tokens: 10000 } };
        normalizeThinkingBlocks(body);
        assert.equal(body.effort, 'max');
    });

    it('does nothing if thinking is already adaptive', () => {
        const body = { thinking: { type: 'adaptive' }, effort: 'high' };
        normalizeThinkingBlocks(body);
        assert.deepEqual(body.thinking, { type: 'adaptive' });
        assert.equal(body.effort, 'high');
    });

    it('does nothing if thinking is missing', () => {
        const body = { model: 'claude-opus-4-7' };
        normalizeThinkingBlocks(body);
        assert.deepEqual(body, { model: 'claude-opus-4-7' });
    });

    it('handles null and undefined body', () => {
        normalizeThinkingBlocks(null);
        normalizeThinkingBlocks(undefined);
    });

    it('does nothing for non-enabled thinking type', () => {
        const body = { thinking: { type: 'disabled' } };
        normalizeThinkingBlocks(body);
        assert.deepEqual(body.thinking, { type: 'disabled' });
    });
});

// ---------------------------------------------------------------------------
// 7. Anthropic API compatibility: stripSamplingParamsOnThinking
// ---------------------------------------------------------------------------
describe('stripSamplingParamsOnThinking', () => {
    it('removes temperature, top_p, top_k when thinking is adaptive', () => {
        const body = {
            thinking: { type: 'adaptive' },
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 1000,
        };
        stripSamplingParamsOnThinking(body);
        assert.equal(body.temperature, undefined);
        assert.equal(body.top_p, undefined);
        assert.equal(body.top_k, undefined);
        assert.equal(body.max_tokens, 1000); // preserved
    });

    it('does nothing when thinking is not set', () => {
        const body = { temperature: 0.7, max_tokens: 1000 };
        stripSamplingParamsOnThinking(body);
        assert.equal(body.temperature, 0.7);
        assert.equal(body.max_tokens, 1000);
    });

    it('does nothing when thinking type is not adaptive', () => {
        const body = {
            thinking: { type: 'enabled', budget_tokens: 4096 },
            temperature: 0.7,
        };
        stripSamplingParamsOnThinking(body);
        assert.equal(body.temperature, 0.7);
    });

    it('handles null and undefined body', () => {
        stripSamplingParamsOnThinking(null);
        stripSamplingParamsOnThinking(undefined);
    });
});

// ---------------------------------------------------------------------------
// 8. Anthropic API compatibility: normalizeOutputConfig
// ---------------------------------------------------------------------------
describe('normalizeOutputConfig', () => {
    it('converts output_format string to output_config object', () => {
        const body = { output_format: 'text' };
        normalizeOutputConfig(body);
        assert.deepEqual(body.output_config, { format: 'text' });
        assert.equal(body.output_format, undefined);
    });

    it('does nothing if output_config already present', () => {
        const body = { output_config: { format: 'text' }, output_format: 'text' };
        normalizeOutputConfig(body);
        assert.deepEqual(body.output_config, { format: 'text' });
        assert.equal(body.output_format, undefined);
    });

    it('does nothing if neither field is present', () => {
        const body = { model: 'claude-opus-4-7' };
        normalizeOutputConfig(body);
        assert.deepEqual(body, { model: 'claude-opus-4-7' });
    });

    it('handles null and undefined body', () => {
        normalizeOutputConfig(null);
        normalizeOutputConfig(undefined);
    });
});

// ---------------------------------------------------------------------------
// 9. Anthropic API compatibility: filterAssistantPrefill
// ---------------------------------------------------------------------------
describe('filterAssistantPrefill', () => {
    it('strips assistant message with empty content array', () => {
        const body = {
            messages: [
                { role: 'user', content: 'hello' },
                { role: 'assistant', content: [] },
                { role: 'user', content: 'world' },
            ],
        };
        filterAssistantPrefill(body);
        assert.equal(body.messages.length, 2);
        assert.equal(body.messages[0].role, 'user');
        assert.equal(body.messages[1].role, 'user');
    });

    it('strips assistant message with missing content field', () => {
        const body = {
            messages: [
                { role: 'user', content: 'hello' },
                { role: 'assistant' },
            ],
        };
        filterAssistantPrefill(body);
        assert.equal(body.messages.length, 1);
    });

    it('preserves assistant message with valid content', () => {
        const body = {
            messages: [
                { role: 'user', content: 'hello' },
                { role: 'assistant', content: [{ type: 'text', text: 'world' }] },
            ],
        };
        filterAssistantPrefill(body);
        assert.equal(body.messages.length, 2);
    });

    it('preserves all user messages unchanged', () => {
        const body = {
            messages: [
                { role: 'user', content: 'hello' },
                { role: 'user', content: 'world' },
            ],
        };
        filterAssistantPrefill(body);
        assert.equal(body.messages.length, 2);
    });

    it('handles null and undefined body', () => {
        filterAssistantPrefill(null);
        filterAssistantPrefill(undefined);
    });

    it('handles body without messages key', () => {
        const body = { model: 'test' };
        filterAssistantPrefill(body);
        assert.deepEqual(body, { model: 'test' });
    });

    it('handles empty messages array', () => {
        const body = { messages: [] };
        filterAssistantPrefill(body);
        assert.deepEqual(body, { messages: [] });
    });
});

// ---------------------------------------------------------------------------
// 10. Anthropic API compatibility: warnOAuthToken
// ---------------------------------------------------------------------------
describe('warnOAuthToken', () => {
    it('logs warning for sk-ant-oat keys', () => {
        const originalWarn = console.warn;
        const warnings = [];
        console.warn = (...args) => warnings.push(args.join(' '));
        try {
            warnOAuthToken('sk-ant-oat-abc123');
            assert.equal(warnings.length, 1);
            assert.match(warnings[0], /OAuth API key/);
            assert.match(warnings[0], /sk-ant-api03/);
        } finally {
            console.warn = originalWarn;
        }
    });

    it('does not warn for sk-ant-api03 keys', () => {
        const originalWarn = console.warn;
        const warnings = [];
        console.warn = (...args) => warnings.push(args.join(' '));
        try {
            warnOAuthToken('sk-ant-api03-abc123');
            assert.equal(warnings.length, 0);
        } finally {
            console.warn = originalWarn;
        }
    });

    it('does not warn for null or undefined', () => {
        const originalWarn = console.warn;
        const warnings = [];
        console.warn = (...args) => warnings.push(args.join(' '));
        try {
            warnOAuthToken(null);
            warnOAuthToken(undefined);
            assert.equal(warnings.length, 0);
        } finally {
            console.warn = originalWarn;
        }
    });

    it('does not warn for empty string', () => {
        const originalWarn = console.warn;
        const warnings = [];
        console.warn = (...args) => warnings.push(args.join(' '));
        try {
            warnOAuthToken('');
            assert.equal(warnings.length, 0);
        } finally {
            console.warn = originalWarn;
        }
    });
});

// ---------------------------------------------------------------------------
// 11. stripDeepSeekUnsupportedFields
// ---------------------------------------------------------------------------
describe('stripDeepSeekUnsupportedFields', () => {
    it('removes image blocks from content arrays', () => {
        const body = {
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'describe this' },
                        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'abc' } },
                    ],
                },
            ],
        };
        stripDeepSeekUnsupportedFields(body);
        assert.equal(body.messages[0].content.length, 1);
        assert.equal(body.messages[0].content[0].type, 'text');
    });

    it('removes document blocks from content arrays', () => {
        const body = {
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'summarize' },
                        { type: 'document', source: { type: 'text', media_type: 'text/plain', data: '...' } },
                    ],
                },
            ],
        };
        stripDeepSeekUnsupportedFields(body);
        assert.equal(body.messages[0].content.length, 1);
        assert.equal(body.messages[0].content[0].type, 'text');
    });

    it('removes cache_control from content blocks', () => {
        const body = {
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'hello', cache_control: { type: 'ephemeral' } },
                        { type: 'text', text: 'world' },
                    ],
                },
            ],
        };
        stripDeepSeekUnsupportedFields(body);
        assert.equal(body.messages[0].content[0].cache_control, undefined);
        assert.equal(body.messages[0].content[1].cache_control, undefined);
    });

    it('removes top_k from request body', () => {
        const body = { top_k: 40, model: 'deepseek-v4-flash' };
        stripDeepSeekUnsupportedFields(body);
        assert.equal(body.top_k, undefined);
        assert.equal(body.model, 'deepseek-v4-flash');
    });

    it('removes disable_parallel_tool_use from request body', () => {
        const body = { disable_parallel_tool_use: true, model: 'deepseek-v4-flash' };
        stripDeepSeekUnsupportedFields(body);
        assert.equal(body.disable_parallel_tool_use, undefined);
        assert.equal(body.model, 'deepseek-v4-flash');
    });

    it('strips multiple unsupported fields in one call', () => {
        const body = {
            top_k: 40,
            disable_parallel_tool_use: true,
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'x' } },
                        { type: 'text', text: 'hi', cache_control: { type: 'ephemeral' } },
                    ],
                },
            ],
        };
        stripDeepSeekUnsupportedFields(body);
        assert.equal(body.top_k, undefined);
        assert.equal(body.disable_parallel_tool_use, undefined);
        assert.equal(body.messages[0].content.length, 1);
        assert.equal(body.messages[0].content[0].cache_control, undefined);
    });

    it('logs a warning when stripping fields', () => {
        const originalWarn = console.warn;
        const warnings = [];
        console.warn = (...args) => warnings.push(args.join(' '));
        try {
            const body = { top_k: 40, messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] };
            stripDeepSeekUnsupportedFields(body);
            assert.equal(warnings.length, 1);
            assert.match(warnings[0], /DeepSeek unsupported fields stripped/);
            assert.match(warnings[0], /top_k/);
        } finally {
            console.warn = originalWarn;
        }
    });

    it('does not warn when no unsupported fields are present', () => {
        const originalWarn = console.warn;
        const warnings = [];
        console.warn = (...args) => warnings.push(args.join(' '));
        try {
            const body = { model: 'deepseek-v4-flash', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] };
            stripDeepSeekUnsupportedFields(body);
            assert.equal(warnings.length, 0);
        } finally {
            console.warn = originalWarn;
        }
    });

    it('preserves thinking and tool_use blocks', () => {
        const body = {
            messages: [
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'deep thought' },
                        { type: 'text', text: 'response' },
                        { type: 'tool_use', id: 'tu_1', name: 'x', input: {} },
                    ],
                },
                {
                    role: 'user',
                    content: [
                        { type: 'tool_result', tool_use_id: 'tu_1', content: 'done' },
                    ],
                },
            ],
        };
        stripDeepSeekUnsupportedFields(body);
        assert.equal(body.messages[0].content.length, 3);
        assert.equal(body.messages[1].content.length, 1);
    });

    it('handles null and undefined body', () => {
        stripDeepSeekUnsupportedFields(null);
        stripDeepSeekUnsupportedFields(undefined);
    });

    it('handles body without messages key', () => {
        const body = { model: 'test' };
        stripDeepSeekUnsupportedFields(body);
        assert.deepEqual(body, { model: 'test' });
    });

    it('handles messages with string content (not array)', () => {
        const body = { messages: [{ role: 'user', content: 'plain string' }] };
        stripDeepSeekUnsupportedFields(body);
        assert.equal(body.messages[0].content, 'plain string');
    });

    it('handles empty messages array', () => {
        const body = { messages: [] };
        stripDeepSeekUnsupportedFields(body);
        assert.deepEqual(body, { messages: [] });
    });
});

// ---------------------------------------------------------------------------
// 12. mapEffortToReasoning
// ---------------------------------------------------------------------------
describe('mapEffortToReasoning', () => {
    it('maps Anthropic effort=low to reasoning_effort=low', () => {
        const body = { effort: 'low' };
        mapEffortToReasoning(body);
        assert.equal(body.reasoning_effort, 'low');
        assert.equal(body.effort, undefined);
    });

    it('maps Anthropic effort=medium to reasoning_effort=medium', () => {
        const body = { effort: 'medium' };
        mapEffortToReasoning(body);
        assert.equal(body.reasoning_effort, 'medium');
        assert.equal(body.effort, undefined);
    });

    it('maps Anthropic effort=high to reasoning_effort=high', () => {
        const body = { effort: 'high' };
        mapEffortToReasoning(body);
        assert.equal(body.reasoning_effort, 'high');
        assert.equal(body.effort, undefined);
    });

    it('maps Anthropic effort=max to reasoning_effort=high', () => {
        const body = { effort: 'max' };
        mapEffortToReasoning(body);
        assert.equal(body.reasoning_effort, 'high');
        assert.equal(body.effort, undefined);
    });

    it('does nothing when effort is not set', () => {
        const body = { model: 'test' };
        mapEffortToReasoning(body);
        assert.deepEqual(body, { model: 'test' });
    });

    it('does nothing for invalid effort values', () => {
        const body = { effort: 'extreme' };
        mapEffortToReasoning(body);
        assert.equal(body.effort, 'extreme');
        assert.equal(body.reasoning_effort, undefined);
    });

    it('preserves other fields when mapping effort', () => {
        const body = { effort: 'high', model: 'deepseek-v4-flash', thinking: { type: 'adaptive' } };
        mapEffortToReasoning(body);
        assert.equal(body.reasoning_effort, 'high');
        assert.equal(body.effort, undefined);
        assert.equal(body.model, 'deepseek-v4-flash');
        assert.deepEqual(body.thinking, { type: 'adaptive' });
    });

    it('handles null and undefined body', () => {
        mapEffortToReasoning(null);
        mapEffortToReasoning(undefined);
    });
});
