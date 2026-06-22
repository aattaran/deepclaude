#!/usr/bin/env node
import { createServer } from 'http';
import { request as httpsRequest } from 'https';
import { request as httpRequest } from 'http';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';

// CLI args: DS_URL DS_KEY [MODE_FILE] [ROUTER_PORT]
const MODE_FILE = process.argv[4] || join(homedir(), '.deepclaude-mode');
const ROUTER_PORT = parseInt(process.argv[5]) || 3199;
const DS_PORT = ROUTER_PORT + 1;
const ANT_PORT = ROUTER_PORT + 2;
const TIMEOUT = 5 * 60 * 1000;
const DS_URL = process.argv[2] || 'https://api.deepseek.com/anthropic';
const DS_KEY = process.argv[3] || '';

function readMode() {
    try { if (existsSync(MODE_FILE)) { const m = readFileSync(MODE_FILE,'utf-8').trim(); if (m==='deepseek'||m==='anthropic') return m; } } catch (_) {}
    return 'deepseek';
}
function writeMode(m) { try { writeFileSync(MODE_FILE, m+'\n', 'utf-8'); } catch (_) {} }
writeMode('deepseek');

// Stateless HTTPS proxy to one upstream backend
function makeProxy(targetUrl, apiKey) {
    const tgt = new URL(targetUrl);
    const basePath = tgt.pathname.replace(/\/$/, '');
    const s = createServer((creq, cres) => {
        const ck = [];
        creq.on('data', c => ck.push(c));
        creq.on('end', () => {
            const body = Buffer.concat(ck);
            const h = { ...creq.headers, host: tgt.host };
            delete h['content-length'];
            if (apiKey) { delete h['authorization']; h['x-api-key'] = apiKey; }
            const opts = {
                hostname: tgt.hostname, port: 443,
                path: basePath + creq.url,
                method: creq.method,
                headers: { ...h, 'content-length': body.length },
                timeout: TIMEOUT
            };
            const r = httpsRequest(opts, (r2) => {
                cres.writeHead(r2.statusCode, r2.headers);
                r2.pipe(cres);
            });
            r.on('timeout', () => { r.destroy(); if (!cres.headersSent) { cres.writeHead(504,{}); cres.end(); } });
            r.on('error', () => { if (!cres.headersSent) { cres.writeHead(502,{}); cres.end(); } });
            r.end(body);
        });
    });
    return s;
}

// Router — reads mode file, routes to backend, auto-fallback on Anthropic errors
function makeRouter(dsPort, antPort) {
    const s = createServer((creq, cres) => {
        if (creq.url === '/_router/status') {
            cres.writeHead(200, {'content-type':'application/json'});
            cres.end(JSON.stringify({mode:readMode(), port: ROUTER_PORT, modeFile: MODE_FILE}));
            return;
        }
        const mode = readMode();
        const port = mode === 'anthropic' ? antPort : dsPort;
        const ck = [];
        creq.on('data', c => ck.push(c));
        creq.on('end', () => {
            const body = Buffer.concat(ck);
            const opts = {
                hostname: '127.0.0.1', port,
                path: creq.url, method: creq.method,
                headers: { ...creq.headers, host: '127.0.0.1:'+port, 'content-length': body.length },
                timeout: TIMEOUT
            };
            const r = httpRequest(opts, (r2) => {
                if (mode === 'anthropic' && (r2.statusCode === 401 || r2.statusCode === 402 || r2.statusCode === 403)) {
                    r2.resume();
                    writeMode('deepseek');
                    const opts2 = {
                        hostname: '127.0.0.1', port: dsPort,
                        path: creq.url, method: creq.method,
                        headers: { ...creq.headers, host: '127.0.0.1:'+dsPort, 'content-length': body.length },
                        timeout: TIMEOUT
                    };
                    const r3 = httpRequest(opts2, (r4) => { cres.writeHead(r4.statusCode, r4.headers); r4.pipe(cres); });
                    r3.on('error', () => { if (!cres.headersSent) { cres.writeHead(502,{}); cres.end(); } });
                    r3.end(body);
                    return;
                }
                cres.writeHead(r2.statusCode, r2.headers);
                r2.pipe(cres);
            });
            r.on('error', () => { if (!cres.headersSent) { cres.writeHead(502,{}); cres.end(); } });
            r.end(body);
        });
    });
    return s;
}

// Start
const dsServer = makeProxy(DS_URL, DS_KEY);
dsServer.listen(DS_PORT, '127.0.0.1', () => console.log('[deepclaude] DeepSeek on :'+DS_PORT));
const antServer = makeProxy('https://api.anthropic.com', '');
antServer.listen(ANT_PORT, '127.0.0.1', () => console.log('[deepclaude] Anthropic on :'+ANT_PORT));
const router = makeRouter(DS_PORT, ANT_PORT);
router.listen(ROUTER_PORT, '127.0.0.1', () => {
    console.log('[deepclaude] Router on :'+ROUTER_PORT+' (mode: '+readMode()+')');
    console.log('[deepclaude] Mode file: '+MODE_FILE);
});
