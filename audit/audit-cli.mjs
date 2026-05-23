/**
 * Audit CLI — runnable entry point.
 *
 * Usage:
 *   node audit/audit-cli.mjs                           # runs with sample data
 *   node audit/audit-cli.mjs --data trades.jsonl        # custom data file
 *   node audit/audit-cli.mjs --symbol MNQ --timeframe 7d
 *   node audit/audit-cli.mjs --format markdown          # output format
 */

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
const __dirname = dirname(fileURLToPath(import.meta.url));
import { audit, auditAndLog } from './performance-audit-hook.mjs';
import { runFullStressSuite } from './stress-test.mjs';
import { generateReport, runFullAudit } from './reporter.mjs';

// Parse CLI args
const args = process.argv.slice(2);
const opts = {
    dataFile: null,
    symbol: null,
    timeframe: '30d',
    format: 'text',
    stress: true,
    startingEquity: 50000,
};

for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
        case '--data':
            opts.dataFile = args[++i];
            break;
        case '--symbol':
            opts.symbol = args[++i];
            break;
        case '--timeframe':
            opts.timeframe = args[++i];
            break;
        case '--format':
            opts.format = args[++i];
            break;
        case '--equity':
            opts.startingEquity = parseInt(args[++i], 10);
            break;
        case '--no-stress':
            opts.stress = false;
            break;
        case '--help':
            printHelp();
            process.exit(0);
    }
}

// Load data
const dataFile = opts.dataFile || resolve(__dirname, 'sample-trades.jsonl');
if (!existsSync(dataFile)) {
    console.error(`ERROR: Data file not found: ${dataFile}`);
    console.error('Run with --help for usage. Generate sample data first.');
    process.exit(1);
}

const raw = readFileSync(dataFile, 'utf-8');
const trades = raw.trim().split('\n')
    .filter(line => line.trim() && !line.startsWith('#'))
    .map(line => JSON.parse(line));

console.log(`Loaded ${trades.length} trades from ${dataFile}\n`);

// Run full audit pipeline
const result = runFullAudit(
    () => audit(trades, { timeframe: opts.timeframe, symbol: opts.symbol, startingEquity: opts.startingEquity }),
    opts.stress ? () => runFullStressSuite(trades, { symbol: opts.symbol, startingEquity: opts.startingEquity }) : null,
    { format: opts.format },
);

// Exit with appropriate code
process.exit(result.audit.status === 'RED' ? 1 : 0);

function printHelp() {
    console.log(`
Audit CLI — Strategy Performance Auditor

Usage: node audit/audit-cli.mjs [options]

Options:
  --data <file>      Trade data file (JSONL, default: sample-trades.jsonl)
  --symbol <sym>     Filter to symbol (e.g., MNQ, NQ, MGC, MCL)
  --timeframe <tf>   Audit timeframe (default: 30d)
  --format <fmt>     Output format: text, markdown, json (default: text)
  --equity <usd>     Starting equity for stress test (default: 50000)
  --no-stress        Skip stress test
  --help             Show this help

Trade JSONL format (one JSON object per line):
  {"symbol":"MNQ","direction":"long","entry_price":19500,"exit_price":19550,
   "pnl_usd":25.0,"hold_sec":120,"exit_time":1716150000000,"entry_time":1716149880000}
`);
}
