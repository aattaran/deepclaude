import { randomUUID } from 'crypto';

const TOOL_CALL_RE = /```tool_call\s*\n([\s\S]*?)\n```/g;
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled', 'expired', 'incomplete']);

/**
 * Convert Anthropic Messages API body → OpenAI Responses API body.
 * @param {object} anthropicBody
 * @param {{ model: string, window?: string, toolMode: 'native'|'emulate'|'none' }} opts
 */
export function buildResponsesRequest(anthropicBody, { model, window, toolMode = 'native' }) {
    const req = {
        model: model || anthropicBody.model,
        background: true,
        max_output_tokens: anthropicBody.max_tokens,
    };

    if (anthropicBody.temperature != null) req.temperature = anthropicBody.temperature;
    if (anthropicBody.top_p != null) req.top_p = anthropicBody.top_p;

    if (window) {
        req.metadata = { completion_window: window };
    }

    let instructions = '';
    if (anthropicBody.system) {
        instructions = flattenSystem(anthropicBody.system);
    }

    if (anthropicBody.thinking?.type === 'enabled') {
        req.enable_thinking = true;
        req.include_reasoning = true;
    }

    const input = [];
    for (const msg of anthropicBody.messages || []) {
        const role = msg.role === 'assistant' ? 'assistant' : 'user';
        const text = flattenMessageContent(msg.content, msg.role);
        if (!text.trim()) continue;
        input.push({ role, content: text });
    }
    req.input = input.length === 1 && input[0].role === 'user' ? input[0].content : input;

    if (toolMode === 'native' && anthropicBody.tools?.length) {
        req.tools = anthropicBody.tools.map(t => ({
            type: 'function',
            name: t.name,
            description: t.description || '',
            parameters: t.input_schema || { type: 'object', properties: {} },
        }));
    } else if (toolMode === 'emulate' && anthropicBody.tools?.length) {
        instructions = appendToolEmulationPrompt(instructions, anthropicBody.tools);
    }

    if (instructions) req.instructions = instructions;
    return req;
}

function flattenSystem(system) {
    if (typeof system === 'string') return system;
    if (!Array.isArray(system)) return '';
    return system
        .filter(b => b.type === 'text')
        .map(b => b.text)
        .join('\n\n');
}

function flattenMessageContent(content, role) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';

    const parts = [];
    for (const block of content) {
        switch (block.type) {
            case 'text':
                parts.push(block.text);
                break;
            case 'thinking':
                if (block.thinking) parts.push(`[thinking]\n${block.thinking}`);
                break;
            case 'tool_use':
                parts.push(`[tool_use name=${block.name} id=${block.id}]\n${JSON.stringify(block.input)}`);
                break;
            case 'tool_result':
                parts.push(`[tool_result tool_use_id=${block.tool_use_id}]\n${formatToolResult(block.content)}`);
                break;
            case 'image':
                parts.push('[image omitted — background providers do not support vision]');
                break;
            default:
                if (block.text) parts.push(block.text);
        }
    }
    return parts.join('\n\n');
}

function formatToolResult(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return JSON.stringify(content);
    return content
        .map(c => (c.type === 'text' ? c.text : JSON.stringify(c)))
        .join('\n');
}

function appendToolEmulationPrompt(instructions, tools) {
    const schemas = tools.map(t =>
        `- ${t.name}: ${t.description || ''}\n  schema: ${JSON.stringify(t.input_schema || {})}`
    ).join('\n');

    const emulation = [
        '',
        '## Tool calling (emulated)',
        'You have access to these tools:',
        schemas,
        '',
        'When you need to call a tool, respond with ONLY a fenced block:',
        '```tool_call',
        '{"name": "<tool_name>", "input": {<args>}, "id": "toolu_<unique>"}',
        '```',
        'Do not include other text when calling a tool.',
    ].join('\n');

    return instructions ? `${instructions}\n${emulation}` : emulation;
}

/**
 * Parse a completed Responses API result into Anthropic message shape.
 */
export function parseResult(response, toolMode = 'native') {
    const usage = {
        input_tokens: response.usage?.input_tokens || 0,
        output_tokens: response.usage?.output_tokens || 0,
    };

    // 'incomplete' (e.g. max_output_tokens hit) may still carry partial output.
    const failedStatuses = ['failed', 'cancelled', 'expired'];
    if (response.error || failedStatuses.includes(response.status)) {
        return {
            error: true,
            message: response.error?.message || `Response ${response.status}`,
            usage,
        };
    }

    const content = [];
    let hasToolUse = false;

    for (const item of response.output || []) {
        if (item.type === 'reasoning') {
            const text = item.summary?.map(s => s.text).join('\n') || item.content || '';
            if (text) content.push({ type: 'thinking', thinking: text });
            continue;
        }
        if (item.type === 'message') {
            for (const block of item.content || []) {
                if (block.type === 'output_text' && block.text) {
                    const parsed = extractToolCallsFromText(block.text, toolMode);
                    if (parsed.toolUses.length) {
                        content.push(...parsed.toolUses);
                        hasToolUse = true;
                        if (parsed.remainingText) {
                            content.push({ type: 'text', text: parsed.remainingText });
                        }
                    } else {
                        content.push({ type: 'text', text: block.text });
                    }
                }
            }
            continue;
        }
        if (item.type === 'function_call' || item.type === 'tool_call') {
            let input = item.input || item.arguments || {};
            if (typeof item.arguments === 'string') {
                try { input = JSON.parse(item.arguments); } catch { input = { _raw: item.arguments }; }
            }
            hasToolUse = true;
            content.push({
                type: 'tool_use',
                id: item.call_id || item.id || `toolu_${randomUUID().replace(/-/g, '').slice(0, 24)}`,
                name: item.name,
                input,
            });
        }
    }

    const stopReason = hasToolUse ? 'tool_use' : 'end_turn';
    return {
        id: response.id || `msg_${randomUUID().replace(/-/g, '').slice(0, 24)}`,
        type: 'message',
        role: 'assistant',
        model: response.model || 'unknown',
        content: content.length ? content : [{ type: 'text', text: '' }],
        stop_reason: stopReason,
        usage,
    };
}

function extractToolCallsFromText(text, toolMode) {
    const toolUses = [];
    let remaining = text;

    if (toolMode === 'emulate' || toolMode === 'native') {
        remaining = text.replace(TOOL_CALL_RE, (_, jsonStr) => {
            try {
                const obj = JSON.parse(jsonStr.trim());
                toolUses.push({
                    type: 'tool_use',
                    id: obj.id || `toolu_${randomUUID().replace(/-/g, '').slice(0, 24)}`,
                    name: obj.name,
                    input: obj.input || {},
                });
            } catch { /* keep in text */ }
            return '';
        }).trim();
    }

    return { toolUses, remainingText: remaining };
}

export function isTerminalStatus(status) {
    return TERMINAL_STATUSES.has(status);
}

export function pollIntervalForWindow(window) {
    switch (window) {
        case '15m': return 5_000;
        case '1h': return 15_000;
        case '6h':
        case '24h':
        case '48h': return 60_000;
        default: return 15_000;
    }
}

/**
 * Synthesize Anthropic SSE event stream from a parsed message.
 * @returns {string} full SSE body
 */
export function synthesizeSSE(message) {
    if (message.error) {
        return sseEvent({ type: 'error', error: { type: 'api_error', message: message.message } });
    }

    const events = [];
    const msgId = message.id;
    const model = message.model;
    const usage = message.usage || { input_tokens: 0, output_tokens: 0 };

    events.push(sseEvent({
        type: 'message_start',
        message: {
            id: msgId,
            type: 'message',
            role: 'assistant',
            model,
            content: [],
            stop_reason: null,
            usage: { input_tokens: usage.input_tokens, output_tokens: 0 },
        },
    }));

    let blockIndex = 0;
    for (const block of message.content) {
        if (block.type === 'thinking') {
            events.push(sseEvent({ type: 'content_block_start', index: blockIndex, content_block: { type: 'thinking', thinking: '' } }));
            events.push(sseEvent({ type: 'content_block_delta', index: blockIndex, delta: { type: 'thinking_delta', thinking: block.thinking } }));
            events.push(sseEvent({ type: 'content_block_stop', index: blockIndex }));
            blockIndex++;
        } else if (block.type === 'text') {
            events.push(sseEvent({ type: 'content_block_start', index: blockIndex, content_block: { type: 'text', text: '' } }));
            events.push(sseEvent({ type: 'content_block_delta', index: blockIndex, delta: { type: 'text_delta', text: block.text } }));
            events.push(sseEvent({ type: 'content_block_stop', index: blockIndex }));
            blockIndex++;
        } else if (block.type === 'tool_use') {
            events.push(sseEvent({
                type: 'content_block_start',
                index: blockIndex,
                content_block: { type: 'tool_use', id: block.id, name: block.name, input: {} },
            }));
            const inputJson = JSON.stringify(block.input);
            events.push(sseEvent({
                type: 'content_block_delta',
                index: blockIndex,
                delta: { type: 'input_json_delta', partial_json: inputJson },
            }));
            events.push(sseEvent({ type: 'content_block_stop', index: blockIndex }));
            blockIndex++;
        }
    }

    events.push(sseEvent({
        type: 'message_delta',
        delta: { stop_reason: message.stop_reason, stop_sequence: null },
        usage: { output_tokens: usage.output_tokens },
    }));
    events.push(sseEvent({ type: 'message_stop' }));

    return events.join('');
}

export function ssePing() {
    return 'event: ping\ndata: {"type":"ping"}\n\n';
}

function sseEvent(obj) {
    return `event: ${obj.type}\ndata: ${JSON.stringify(obj)}\n\n`;
}

export function toolsErrorIndicatesUnsupported(errBody) {
    const s = typeof errBody === 'string' ? errBody : JSON.stringify(errBody);
    return /tool/i.test(s) && (/unsupported|invalid|unknown|not allowed|not supported/i.test(s));
}
