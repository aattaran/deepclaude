/**
 * Gemini API Translator — Anthropic ↔ Gemini format bridge.
 *
 * Converts Anthropic-format requests to Gemini API format and translates
 * Gemini streaming/non-streaming responses back to Anthropic-compatible output.
 *
 * Gemini 3.5 Flash (released May 19, 2026 at Google I/O):
 *   Base: https://generativelanguage.googleapis.com
 *   Streaming:  POST /v1beta/models/{model}:streamGenerateContent?alt=sse
 *   Non-stream: POST /v1beta/models/{model}:generateContent
 *
 * Auth: x-goog-api-key header (not ?key= query param — leaks in logs)
 * Pricing: $1.50/M input, $9.00/M output, $0.15/M cached (90% off)
 * Context: 1,048,576 input / 65,536 output tokens
 * Note: temperature/top_p/top_k silently ignored by Gemini 3.5
 */

import { Transform } from 'stream';

// ---------------------------------------------------------------------------
// Request translation: Anthropic → Gemini
// ---------------------------------------------------------------------------

/**
 * Convert an Anthropic messages request body to Gemini generateContent format.
 * Returns { geminiBody, geminiModel } — the model name is extracted because
 * Gemini bakes it into the URL, not the request body.
 */
export function translateRequest(anthropicBody) {
    const parsed = typeof anthropicBody === 'string'
        ? JSON.parse(anthropicBody)
        : anthropicBody;

    const gemini = { contents: [] };
    const geminiModel = parsed.model || 'gemini-3.5-flash';

    // system prompt → systemInstruction
    if (parsed.system) {
        const texts = extractTexts(parsed.system);
        if (texts.length > 0) {
            gemini.systemInstruction = { parts: texts.map(t => ({ text: t })) };
        }
    }

    // messages → contents
    // Track tool_use_id → function name for tool_result resolution
    const toolUseMap = new Map();
    if (Array.isArray(parsed.messages)) {
        for (const msg of parsed.messages) {
            // Record tool_use id→name mappings from assistant messages
            if (msg.role === 'assistant' && Array.isArray(msg.content)) {
                for (const block of msg.content) {
                    if (block.type === 'tool_use' && block.id && block.name) {
                        toolUseMap.set(block.id, { name: block.name, id: block.id });
                    }
                }
            }
            const converted = convertMessage(msg, toolUseMap);
            if (converted) {
                if (Array.isArray(converted)) {
                    gemini.contents.push(...converted);
                } else {
                    gemini.contents.push(converted);
                }
            }
        }
    }

    // tools → tools[].functionDeclarations
    if (Array.isArray(parsed.tools) && parsed.tools.length > 0) {
        gemini.tools = [{
            functionDeclarations: parsed.tools.map(t => ({
                name: t.name,
                description: t.description || '',
                parameters: t.input_schema || { type: 'object', properties: {} },
            })),
        }];
    }

    // tool_choice
    if (parsed.tool_choice) {
        if (parsed.tool_choice.type === 'any') {
            gemini.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
        } else if (parsed.tool_choice.type === 'auto') {
            gemini.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
        } else if (parsed.tool_choice.type === 'tool' && parsed.tool_choice.name) {
            gemini.toolConfig = {
                functionCallingConfig: {
                    mode: 'ANY',
                    allowedFunctionNames: [parsed.tool_choice.name],
                },
            };
        }
    }

    // max_tokens, temperature, stop_sequences → generationConfig
    const genConfig = {};
    if (parsed.max_tokens != null) genConfig.maxOutputTokens = parsed.max_tokens;
    if (parsed.temperature != null) genConfig.temperature = parsed.temperature;
    if (parsed.top_p != null) genConfig.topP = parsed.top_p;
    if (parsed.top_k != null) genConfig.topK = parsed.top_k;
    if (Array.isArray(parsed.stop_sequences) && parsed.stop_sequences.length > 0) {
        genConfig.stopSequences = parsed.stop_sequences;
    }
    // thinking → thinking_level (Gemini replaced thinking_budget with thinking_level enum)
    if (parsed.thinking) {
        const effort = parsed.thinking.effort;
        if (effort) {
            const levelMap = {
                'low': 'low',
                'medium': 'medium',
                'high': 'high',
                'max': 'high',
                'xhigh': 'high',
            };
            const tl = levelMap[effort];
            if (tl) {
                genConfig.thinkingLevel = tl;
            }
        }
    }

    // output_config → Gemini response format (if supported)
    // Anthropic replaced output_format with output_config.
    // Gemini does not currently support this, so we silently skip.
    if (parsed.output_config) {
        // Future: map to Gemini responseMimeType or responseSchema when available
    }

    if (Object.keys(genConfig).length > 0) {
        gemini.generationConfig = genConfig;
    }

    return { geminiBody: gemini, geminiModel };
}

function extractTexts(system) {
    if (typeof system === 'string') return [system];
    if (Array.isArray(system)) {
        return system
            .filter(block => block.type === 'text')
            .map(block => block.text);
    }
    return [];
}

/**
 * Extract text from a tool_result content field.
 * Anthropic tool_result.content is an array of content blocks like
 * [{type: "text", text: "result"}]. Extract .text from text blocks
 * and join with newlines instead of JSON-stringifying the whole array.
 */
function extractToolResultText(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return JSON.stringify(content);
    return content
        .filter(b => b.type === 'text')
        .map(b => b.text)
        .join('\n');
}

function convertMessage(msg, toolUseMap = new Map()) {
    const role = mapRole(msg.role);
    if (!role) return null;

    const content = msg.content;
    if (typeof content === 'string') {
        return { role, parts: [{ text: content }] };
    }

    if (!Array.isArray(content)) return null;

    const parts = [];
    const toolResults = [];

    for (const block of content) {
        if (block.type === 'text') {
            parts.push({ text: block.text });
        } else if (block.type === 'tool_use') {
            parts.push({
                functionCall: {
                    name: block.name,
                    args: block.input || {},
                },
            });
        } else if (block.type === 'tool_result') {
            // Resolve function name from tool_use_id. Gemini requires the
            // actual function name, not the Anthropic UUID.
            const fnEntry = toolUseMap.get(block.tool_use_id);
            const fnName = fnEntry?.name || block.tool_use_id || 'unknown';
            const fnId = fnEntry?.id || block.tool_use_id;
            // tool_result.content is an array of content blocks like
            // [{type: "text", text: "result"}]. Extract text parts explicitly
            // rather than JSON-stringifying the entire array.
            const contentText = extractToolResultText(block.content);
            toolResults.push({
                name: fnName,
                id: fnId,
                response: { content: contentText },
            });
        } else if (block.type === 'image' && block.source) {
            parts.push({
                inlineData: {
                    mimeType: block.source.media_type || 'image/png',
                    data: block.source.data,
                },
            });
        }
        // Skip thinking blocks — Gemini doesn't support them
    }

    // If we only have tool results, emit as a functionResponse user message
    if (toolResults.length > 0 && parts.length === 0) {
        return {
            role: 'user',
            parts: toolResults.map(tr => ({ functionResponse: tr })),
        };
    }

    // If we have both parts and tool results, the tool results need a separate message
    if (parts.length > 0) {
        const result = [{ role, parts }];
        if (toolResults.length > 0) {
            result.push({ role: 'user', parts: toolResults.map(tr => ({ functionResponse: tr })) });
        }
        return result.length === 1 ? result[0] : result;
    }

    return null;
}

function mapRole(anthropicRole) {
    switch (anthropicRole) {
        case 'user': return 'user';
        case 'assistant': return 'model';
        default:
            console.warn(`[gemini-translator] Unknown role "${anthropicRole}" — message dropped`);
            return null;
    }
}

// ---------------------------------------------------------------------------
// Response translation: Gemini SSE → Anthropic SSE
// ---------------------------------------------------------------------------

let _msgIdCounter = 0;

function nextMsgId() {
    return `msg_gemini_${Date.now()}_${++_msgIdCounter}`;
}

/**
 * Transform stream that converts Gemini SSE events to Anthropic SSE format.
 * Pipe this between the Gemini HTTP response and the UsageNormalizer.
 */
export class GeminiStreamTranslator extends Transform {
    constructor() {
        super();
        this._buf = '';
        this._msgId = null;
        this._blockIndex = 0;
        this._textBlockOpen = false;
        this._toolBlockOpen = false;
        this._inputTokens = 0;
        this._outputTokens = 0;
        this._sentMessageStart = false;
        this._finishReason = null;
    }

    get inputTokens() { return this._inputTokens; }
    get outputTokens() { return this._outputTokens; }

    _transform(chunk, _enc, cb) {
        this._buf += chunk.toString();
        const lines = this._buf.split('\n');
        // Keep the last partial line in buffer
        this._buf = lines.pop();

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data: ')) continue;
            const dataStr = trimmed.slice(6);
            if (dataStr === '[DONE]') continue;

            try {
                const data = JSON.parse(dataStr);
                this._processGeminiEvent(data);
            } catch {
                // Non-JSON line, skip
            }
        }
        cb();
    }

    _flush(cb) {
        if (this._buf.trim()) {
            const trimmed = this._buf.trim();
            if (trimmed.startsWith('data: ') && trimmed.slice(6) !== '[DONE]') {
                try {
                    const data = JSON.parse(trimmed.slice(6));
                    this._processGeminiEvent(data);
                } catch { /* skip */ }
            }
        }
        // Close any open blocks and emit final events
        this._closeOpenBlocks();
        this._emitMessageStop();
        cb();
    }

    _processGeminiEvent(data) {
        const candidates = data.candidates;
        if (!Array.isArray(candidates) || candidates.length === 0) return;

        const candidate = candidates[0];
        const content = candidate.content;
        if (!content) return;

        // Extract usage if present
        if (data.usageMetadata) {
            this._inputTokens = data.usageMetadata.promptTokenCount || 0;
            this._outputTokens = data.usageMetadata.candidatesTokenCount
                || data.usageMetadata.totalTokenCount
                || 0;
        }

        // Track finish reason (including STOP so we can distinguish from connection drops)
        if (candidate.finishReason) {
            this._finishReason = candidate.finishReason;
        }

        // Ensure message_start sent
        if (!this._sentMessageStart) {
            this._msgId = nextMsgId();
            this._emitMessageStart();
            this._sentMessageStart = true;
        }

        const parts = content.parts;
        if (!Array.isArray(parts)) return;

        for (const part of parts) {
            if (part.text != null) {
                this._emitText(part.text);
            } else if (part.functionCall) {
                this._emitToolCall(part.functionCall);
            }
        }
    }

    _emitMessageStart() {
        this.push(`event: message_start\ndata: ${JSON.stringify({
            type: 'message_start',
            message: {
                id: this._msgId,
                type: 'message',
                role: 'assistant',
                content: [],
                model: 'gemini-3.5-flash',
                usage: { input_tokens: this._inputTokens },
            },
        })}\n\n`);
    }

    _emitText(text) {
        // Close tool block if open (switching from tool to text)
        if (this._toolBlockOpen) {
            this._closeToolBlock();
        }

        // Open text block if not open
        if (!this._textBlockOpen) {
            this._textBlockOpen = true;
            this.push(`event: content_block_start\ndata: ${JSON.stringify({
                type: 'content_block_start',
                index: this._blockIndex,
                content_block: { type: 'text', text: '' },
            })}\n\n`);
        }

        this.push(`event: content_block_delta\ndata: ${JSON.stringify({
            type: 'content_block_delta',
            index: this._blockIndex,
            delta: { type: 'text_delta', text },
        })}\n\n`);
    }

    _emitToolCall(functionCall) {
        // Close text block if open (switching from text to tool)
        if (this._textBlockOpen) {
            this._closeTextBlock();
        }

        // Close previous tool block if open
        if (this._toolBlockOpen) {
            this._closeToolBlock();
        }

        // Open tool block
        this._toolBlockOpen = true;
        const toolId = `toolu_gemini_${Date.now()}_${this._blockIndex}`;
        const toolInput = functionCall.args || {};

        this.push(`event: content_block_start\ndata: ${JSON.stringify({
            type: 'content_block_start',
            index: this._blockIndex,
            content_block: {
                type: 'tool_use',
                id: toolId,
                name: functionCall.name,
                input: toolInput,
            },
        })}\n\n`);

        // Emit the full tool input as a single delta
        const inputJson = JSON.stringify(toolInput);
        this.push(`event: content_block_delta\ndata: ${JSON.stringify({
            type: 'content_block_delta',
            index: this._blockIndex,
            delta: {
                type: 'input_json_delta',
                partial_json: inputJson,
            },
        })}\n\n`);
    }

    _closeOpenBlocks() {
        if (this._textBlockOpen) this._closeTextBlock();
        if (this._toolBlockOpen) this._closeToolBlock();
    }

    _closeTextBlock() {
        this.push(`event: content_block_stop\ndata: ${JSON.stringify({
            type: 'content_block_stop',
            index: this._blockIndex,
        })}\n\n`);
        this._textBlockOpen = false;
        this._blockIndex++;
    }

    _closeToolBlock() {
        this.push(`event: content_block_stop\ndata: ${JSON.stringify({
            type: 'content_block_stop',
            index: this._blockIndex,
        })}\n\n`);
        this._toolBlockOpen = false;
        this._blockIndex++;
    }

    _emitMessageStop() {
        if (!this._sentMessageStart) return;

        const stopReason = mapFinishReason(this._finishReason);

        this.push(`event: message_delta\ndata: ${JSON.stringify({
            type: 'message_delta',
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: { output_tokens: this._outputTokens },
        })}\n\n`);

        this.push(`event: message_stop\ndata: ${JSON.stringify({
            type: 'message_stop',
        })}\n\n`);
    }
}

function mapFinishReason(reason) {
    switch (reason) {
        case 'MAX_TOKENS': return 'max_tokens';
        case 'SAFETY': return 'safety_blocked';
        case 'RECITATION': return 'recitation_blocked';
        case 'STOP':
        case null:
        case undefined: return 'end_turn';
        default: return 'end_turn';
    }
}

// ---------------------------------------------------------------------------
// Response translation: Gemini JSON → Anthropic JSON (non-streaming)
// ---------------------------------------------------------------------------

/**
 * Convert a Gemini non-streaming JSON response to Anthropic format.
 */
export function translateNonStreamingResponse(geminiBody) {
    const parsed = typeof geminiBody === 'string'
        ? JSON.parse(geminiBody)
        : geminiBody;

    const msgId = nextMsgId();
    const content = [];

    const candidate = parsed.candidates?.[0];
    if (candidate?.content?.parts) {
        for (const part of candidate.content.parts) {
            if (part.text != null) {
                content.push({ type: 'text', text: part.text });
            } else if (part.functionCall) {
                content.push({
                    type: 'tool_use',
                    id: `toolu_gemini_${Date.now()}_${content.length}`,
                    name: part.functionCall.name,
                    input: part.functionCall.args || {},
                });
            }
        }
    }

    const inputTokens = parsed.usageMetadata?.promptTokenCount || 0;
    const outputTokens = parsed.usageMetadata?.candidatesTokenCount
        || parsed.usageMetadata?.totalTokenCount
        || 0;

    const stopReason = mapFinishReason(candidate?.finishReason);

    return {
        id: msgId,
        type: 'message',
        role: 'assistant',
        content,
        model: 'gemini-3.5-flash',
        stop_reason: stopReason,
        stop_sequence: null,
        usage: {
            input_tokens: inputTokens,
            output_tokens: outputTokens,
        },
    };
}
