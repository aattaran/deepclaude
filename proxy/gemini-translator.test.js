import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  translateRequest,
  GeminiStreamTranslator,
  translateNonStreamingResponse,
} from './gemini-translator.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseSSE(raw) {
  const lines = raw.split('\n');
  const result = {};
  for (const line of lines) {
    if (line.startsWith('event: ')) result.event = line.slice(7);
    else if (line.startsWith('data: ')) result.data = JSON.parse(line.slice(6));
  }
  return result;
}

function collectEvents(stream, lines) {
  return new Promise((resolve) => {
    const chunks = [];
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('finish', () => {
      const all = chunks.join('').split('\n\n').filter(Boolean);
      resolve(all.map(parseSSE));
    });
    for (const l of lines) stream.write(l + '\n');
    stream.end();
  });
}

function geminiCandidate(parts, finishReason) {
  const candidate = { content: { parts, role: 'model' }, safetyRatings: [] };
  if (finishReason) candidate.finishReason = finishReason;
  return [candidate];
}

function geminiSSE(candidates, usageMetadata) {
  const obj = { candidates };
  if (usageMetadata) obj.usageMetadata = usageMetadata;
  return `data: ${JSON.stringify(obj)}`;
}

// ---------------------------------------------------------------------------
// translateRequest
// ---------------------------------------------------------------------------

describe('translateRequest', () => {
  it('converts a basic text message', () => {
    const { geminiBody, geminiModel } = translateRequest({
      messages: [{ role: 'user', content: 'hello' }],
    });
    assert.equal(geminiModel, 'gemini-3.5-flash');
    assert.deepEqual(geminiBody, {
      contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
    });
  });

  it('converts system prompt to systemInstruction', () => {
    const { geminiBody } = translateRequest({
      system: 'You are a helpful assistant.',
      messages: [{ role: 'user', content: 'hi' }],
    });
    assert.deepEqual(geminiBody.systemInstruction, {
      parts: [{ text: 'You are a helpful assistant.' }],
    });
  });

  it('converts tool definitions', () => {
    const { geminiBody } = translateRequest({
      messages: [{ role: 'user', content: 'weather?' }],
      tools: [{
        name: 'get_weather',
        description: 'Get the weather',
        input_schema: {
          type: 'object',
          properties: { location: { type: 'string' } },
          required: ['location'],
        },
      }],
    });
    assert.deepEqual(geminiBody.tools, [{
      functionDeclarations: [{
        name: 'get_weather',
        description: 'Get the weather',
        parameters: {
          type: 'object',
          properties: { location: { type: 'string' } },
          required: ['location'],
        },
      }],
    }]);
  });

  it('maps tool_choice correctly', () => {
    const r1 = translateRequest({ messages: [], tool_choice: { type: 'any' } });
    assert.deepEqual(r1.geminiBody.toolConfig, {
      functionCallingConfig: { mode: 'ANY' },
    });

    const r2 = translateRequest({ messages: [], tool_choice: { type: 'auto' } });
    assert.deepEqual(r2.geminiBody.toolConfig, {
      functionCallingConfig: { mode: 'AUTO' },
    });

    const r3 = translateRequest({
      messages: [],
      tool_choice: { type: 'tool', name: 'get_weather' },
    });
    assert.deepEqual(r3.geminiBody.toolConfig, {
      functionCallingConfig: {
        mode: 'ANY',
        allowedFunctionNames: ['get_weather'],
      },
    });
  });

  it('maps generation config fields', () => {
    const { geminiBody } = translateRequest({
      max_tokens: 200,
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
      stop_sequences: ['\n\n', '.'],
      messages: [{ role: 'user', content: 'hi' }],
    });
    assert.deepEqual(geminiBody.generationConfig, {
      maxOutputTokens: 200,
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      stopSequences: ['\n\n', '.'],
    });
  });

  it('converts assistant tool_use to functionCall parts', () => {
    const { geminiBody } = translateRequest({
      messages: [
        { role: 'user', content: 'weather?' },
        {
          role: 'assistant',
          content: [{
            type: 'tool_use',
            id: 'tu_123',
            name: 'get_weather',
            input: { location: 'Paris' },
          }],
        },
      ],
    });
    assert.equal(geminiBody.contents[1].role, 'model');
    assert.deepEqual(geminiBody.contents[1].parts, [{
      functionCall: { name: 'get_weather', args: { location: 'Paris' } },
    }]);
  });

  it('converts user tool_result to functionResponse parts', () => {
    const { geminiBody } = translateRequest({
      messages: [{
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: 'tu_123',
          content: '{"temp":22}',
        }],
      }],
    });
    assert.equal(geminiBody.contents.length, 1);
    assert.equal(geminiBody.contents[0].role, 'user');
    // The translator falls back to tool_use_id when no preceding tool_use block
    // provides a function name resolution.
    assert.deepEqual(geminiBody.contents[0].parts, [{
      functionResponse: {
        name: 'tu_123',
        response: { content: '{"temp":22}' },
      },
    }]);
  });

  it('converts mixed text + tool_use in one assistant message', () => {
    const { geminiBody } = translateRequest({
      messages: [{
        role: 'assistant',
        content: [
          { type: 'text', text: 'Let me check.' },
          {
            type: 'tool_use',
            id: 'tu_1',
            name: 'get_weather',
            input: { loc: 'Paris' },
          },
        ],
      }],
    });
    assert.equal(geminiBody.contents.length, 1);
    assert.equal(geminiBody.contents[0].role, 'model');
    assert.deepEqual(geminiBody.contents[0].parts, [
      { text: 'Let me check.' },
      { functionCall: { name: 'get_weather', args: { loc: 'Paris' } } },
    ]);
  });
});

// ---------------------------------------------------------------------------
// GeminiStreamTranslator
// ---------------------------------------------------------------------------

describe('GeminiStreamTranslator', () => {
  it('streams text with correct SSE event sequence', async () => {
    const translator = new GeminiStreamTranslator();
    const events = await collectEvents(translator, [
      geminiSSE(
        geminiCandidate([{ text: 'Hello ' }]),
        { promptTokenCount: 10, candidatesTokenCount: 5 },
      ),
      geminiSSE(
        geminiCandidate([{ text: 'world' }]),
        { promptTokenCount: 10, candidatesTokenCount: 5 },
      ),
    ]);

    assert.equal(events.length, 7);

    // 1. message_start
    assert.equal(events[0].event, 'message_start');
    assert.match(events[0].data.message.id, /^msg_gemini_\d+_\d+$/);
    assert.equal(events[0].data.message.role, 'assistant');
    assert.deepEqual(events[0].data.message.content, []);
    assert.equal(events[0].data.message.usage.input_tokens, 10);

    // 2. content_block_start (text)
    assert.equal(events[1].event, 'content_block_start');
    assert.equal(events[1].data.index, 0);
    assert.equal(events[1].data.content_block.type, 'text');
    assert.equal(events[1].data.content_block.text, '');

    // 3. content_block_delta — first text chunk
    assert.equal(events[2].event, 'content_block_delta');
    assert.equal(events[2].data.delta.type, 'text_delta');
    assert.equal(events[2].data.delta.text, 'Hello ');
    assert.equal(events[2].data.index, 0);

    // 4. content_block_delta — second text chunk
    assert.equal(events[3].event, 'content_block_delta');
    assert.equal(events[3].data.delta.type, 'text_delta');
    assert.equal(events[3].data.delta.text, 'world');
    assert.equal(events[3].data.index, 0);

    // 5. content_block_stop
    assert.equal(events[4].event, 'content_block_stop');
    assert.equal(events[4].data.index, 0);

    // 6. message_delta
    assert.equal(events[5].event, 'message_delta');
    assert.equal(events[5].data.delta.stop_reason, 'end_turn');
    assert.equal(events[5].data.delta.stop_sequence, null);
    assert.equal(events[5].data.usage.output_tokens, 5);

    // 7. message_stop
    assert.equal(events[6].event, 'message_stop');
  });

  it('streams tool call with correct SSE sequence', async () => {
    const translator = new GeminiStreamTranslator();
    const events = await collectEvents(translator, [
      geminiSSE(geminiCandidate([
        { functionCall: { name: 'get_weather', args: { location: 'Paris' } } },
      ])),
    ]);

    assert.equal(events.length, 6);

    assert.equal(events[0].event, 'message_start');

    assert.equal(events[1].event, 'content_block_start');
    assert.equal(events[1].data.index, 0);
    assert.equal(events[1].data.content_block.type, 'tool_use');
    assert.match(events[1].data.content_block.id, /^toolu_gemini_\d+_\d+$/);
    assert.equal(events[1].data.content_block.name, 'get_weather');
    assert.deepEqual(events[1].data.content_block.input, {});

    assert.equal(events[2].event, 'content_block_delta');
    assert.equal(events[2].data.index, 0);
    assert.equal(events[2].data.delta.type, 'input_json_delta');
    assert.equal(
      events[2].data.delta.partial_json,
      '{"location":"Paris"}',
    );

    assert.equal(events[3].event, 'content_block_stop');
    assert.equal(events[3].data.index, 0);

    assert.equal(events[4].event, 'message_delta');
    assert.equal(events[4].data.delta.stop_reason, 'end_turn');

    assert.equal(events[5].event, 'message_stop');
  });

  it('handles switching between text and tool blocks', async () => {
    const translator = new GeminiStreamTranslator();
    const events = await collectEvents(translator, [
      geminiSSE(geminiCandidate([{ text: 'Hello ' }])),
      geminiSSE(geminiCandidate([
        { functionCall: { name: 'get_weather', args: { loc: 'Paris' } } },
      ])),
      geminiSSE(geminiCandidate([{ text: 'Done' }])),
    ]);

    assert.equal(events.length, 12);
    assert.equal(events[0].event, 'message_start');

    // First text block at index 0
    assert.equal(events[1].data.index, 0);
    assert.equal(events[1].data.content_block.type, 'text');
    assert.equal(events[3].data.index, 0);

    // Tool block at index 1
    assert.equal(events[4].data.index, 1);
    assert.equal(events[4].data.content_block.type, 'tool_use');
    assert.equal(events[6].data.index, 1);

    // Second text block at index 2
    assert.equal(events[7].data.index, 2);
    assert.equal(events[7].data.content_block.type, 'text');
    assert.equal(events[9].data.index, 2);

    assert.equal(events[10].event, 'message_delta');
    assert.equal(events[11].event, 'message_stop');
  });

  it('exposes usage metadata via getters', async () => {
    const translator = new GeminiStreamTranslator();
    await collectEvents(translator, [
      geminiSSE(
        geminiCandidate([{ text: 'Hello' }]),
        { promptTokenCount: 50, candidatesTokenCount: 25 },
      ),
    ]);
    assert.equal(translator.inputTokens, 50);
    assert.equal(translator.outputTokens, 25);
  });
});

// ---------------------------------------------------------------------------
// translateNonStreamingResponse
// ---------------------------------------------------------------------------

describe('translateNonStreamingResponse', () => {
  it('converts a text response to Anthropic format', () => {
    const result = translateNonStreamingResponse({
      candidates: [{
        content: { parts: [{ text: 'Hello world' }], role: 'model' },
        finishReason: 'STOP',
      }],
      usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 },
    });

    assert.match(result.id, /^msg_gemini_\d+_\d+$/);
    assert.equal(result.type, 'message');
    assert.equal(result.role, 'assistant');
    assert.equal(result.model, 'gemini-3.5-flash');
    assert.equal(result.stop_reason, 'end_turn');
    assert.equal(result.stop_sequence, null);
    assert.deepEqual(result.content, [{ type: 'text', text: 'Hello world' }]);
    assert.deepEqual(result.usage, { input_tokens: 10, output_tokens: 5 });
  });

  it('converts functionCall to tool_use content blocks', () => {
    const result = translateNonStreamingResponse({
      candidates: [{
        content: {
          parts: [{
            functionCall: { name: 'get_weather', args: { location: 'Paris' } },
          }],
          role: 'model',
        },
        finishReason: 'STOP',
      }],
      usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 },
    });

    assert.equal(result.content.length, 1);
    assert.equal(result.content[0].type, 'tool_use');
    assert.match(result.content[0].id, /^toolu_gemini_\d+_\d+$/);
    assert.equal(result.content[0].name, 'get_weather');
    assert.deepEqual(result.content[0].input, { location: 'Paris' });
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe('edge cases', () => {
  it('handles empty messages array', () => {
    const { geminiBody } = translateRequest({ messages: [] });
    assert.deepEqual(geminiBody, { contents: [] });
  });

  it('maps unknown or missing finish reason to end_turn', () => {
    const r1 = translateNonStreamingResponse({
      candidates: [{
        content: { parts: [{ text: 'hi' }], role: 'model' },
        finishReason: 'UNKNOWN',
      }],
    });
    assert.equal(r1.stop_reason, 'end_turn');

    const r2 = translateNonStreamingResponse({
      candidates: [{
        content: { parts: [{ text: 'hi' }], role: 'model' },
      }],
    });
    assert.equal(r2.stop_reason, 'end_turn');

    const r3 = translateNonStreamingResponse({
      candidates: [{
        content: { parts: [{ text: 'hi' }], role: 'model' },
        finishReason: 'MAX_TOKENS',
      }],
    });
    assert.equal(r3.stop_reason, 'max_tokens');

    const r4 = translateNonStreamingResponse({});
    assert.equal(r4.stop_reason, 'end_turn');

    const r5 = translateNonStreamingResponse({ candidates: [] });
    assert.equal(r5.stop_reason, 'end_turn');
  });
});
