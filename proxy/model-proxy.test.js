import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    MODEL_REMAP,
    PRICING_PER_M,
    NON_ANTHROPIC_BACKENDS,
    isHaikuModel,
    resolveAutoBackend,
    stripAllThinkingBlocks,
    stripUnsignedThinkingBlocks,
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
