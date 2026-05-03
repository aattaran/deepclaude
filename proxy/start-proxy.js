#!/usr/bin/env node
import { startModelProxy } from './model-proxy.js';

const targetUrl = process.argv[2] || process.env.CHEAPCLAUDE_TARGET_URL;
const apiKey = process.argv[3] || process.env.CHEAPCLAUDE_API_KEY;

if (!targetUrl || !apiKey) {
    console.error('Usage: node start-proxy.js <targetUrl> <apiKey>');
    process.exit(1);
}

const { port } = await startModelProxy({ targetUrl, apiKey });
console.log(port);
