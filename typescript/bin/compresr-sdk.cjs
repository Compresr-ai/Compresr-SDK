#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-require-imports */
const { main } = require('../dist/cjs/cli/index.js');

// Backup SIGINT: login() installs its own during the browser flow and does a
// clean shutdown; this handler only fires if we're outside that window.
let sigintCount = 0;
process.on('SIGINT', () => {
  sigintCount++;
  if (sigintCount >= 2) process.exit(130);
});

main().then(
  (rc) => process.exit(rc),
  (e) => {
    process.stderr.write(`fatal: ${e && e.stack ? e.stack : e}\n`);
    process.exit(1);
  }
);
