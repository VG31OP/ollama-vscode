const assert = require('node:assert/strict');
const Module = require('node:module');
const test = require('node:test');

class CancellationError extends Error {
  constructor(message = 'Canceled') {
    super(message);
    this.name = 'CancellationError';
  }
}

const vscode = {
  CancellationError,
  EventEmitter: class { constructor() { this.event = () => ({ dispose() {} }); } fire() {} dispose() {} },
  CancellationTokenSource: class { constructor() { this.token = { isCancellationRequested: false, onCancellationRequested: () => ({ dispose() {} }) }; } cancel() {} dispose() {} },
  LanguageModelChatMessageRole: { User: 1, Assistant: 2 },
  LanguageModelTextPart: class {},
  LanguageModelDataPart: class {},
  LanguageModelToolCallPart: class {},
  LanguageModelToolResultPart: class {},
  lm: { registerLanguageModelChatProvider: () => ({ dispose() {} }) },
  window: { showErrorMessage: () => undefined, createOutputChannel: () => ({ appendLine() {}, dispose() {} }) },
  workspace: { getConfiguration: () => ({ get: () => undefined }), onDidChangeConfiguration: () => ({ dispose() {} }) },
  Uri: { parse: (v) => v },
  env: { openExternal: () => undefined }
};

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') {
    return vscode;
  }
  return originalLoad.call(this, request, parent, isMain);
};

let isAbortedRequestError;
let isToolCallParseError;
let OllamaAPIError;
try {
  ({ isAbortedRequestError, isToolCallParseError, OllamaAPIError } = require('../out/provider'));
} finally {
  Module._load = originalLoad;
}

function apiError(status, responseError) {
  return new OllamaAPIError(
    `Ollama /api/chat failed with HTTP ${status}: ${responseError}`,
    status,
    '/api/chat',
    responseError
  );
}

// The exact payload Ollama rejected: a stray quote after the empty array.
const realParseError = `error parsing tool call: raw='{"command":"pytest -q","requestFileValidationCheck":[]","requestUnsandboxedExecution":false}', err=invalid character '"' after object key:value pair`;

test('classifies Ollama tool-call parse rejection as retryable', () => {
  assert.equal(isToolCallParseError(apiError(500, realParseError)), true);
  assert.equal(isToolCallParseError({ status: 500, responseError: realParseError }), true);
  assert.equal(isToolCallParseError({ status: 500, message: realParseError }), true);
});

test('does not retry other server errors', () => {
  assert.equal(isToolCallParseError(apiError(500, 'model requires more system memory')), false);
  assert.equal(isToolCallParseError(apiError(500, 'internal server error')), false);
});

test('does not retry client errors or non-API errors', () => {
  assert.equal(isToolCallParseError(apiError(404, 'model not found')), false);
  assert.equal(isToolCallParseError(apiError(400, 'bad request')), false);
  assert.equal(isToolCallParseError(new TypeError('terminated')), false);
  assert.equal(isToolCallParseError(undefined), false);
  assert.equal(isToolCallParseError(null), false);
});

test('recognizes a DOM-style AbortError', () => {
  const error = new Error('This operation was aborted');
  error.name = 'AbortError';
  assert.equal(isAbortedRequestError(error), true);
});

test('recognizes undici TypeError("terminated") with no cause', () => {
  assert.equal(isAbortedRequestError(new TypeError('terminated')), true);
});

test('recognizes an abort wrapped as the cause of a fetch failure', () => {
  const abort = new Error('This operation was aborted');
  abort.name = 'AbortError';
  assert.equal(isAbortedRequestError(new TypeError('fetch failed', { cause: abort })), true);
});

test('recognizes a terminated error nested deeper in the cause chain', () => {
  const nested = new TypeError('fetch failed', { cause: new TypeError('terminated') });
  assert.equal(isAbortedRequestError(nested), true);
});

test('guards against cyclic cause chains without infinite recursion or stack overflow', () => {
  const cyclic = new Error('cyclic error');
  cyclic.cause = cyclic;
  assert.equal(isAbortedRequestError(cyclic), false);

  const a = new Error('cycle a');
  const b = new Error('cycle b');
  a.cause = b;
  b.cause = a;
  assert.equal(isAbortedRequestError(a), false);
});

test('does not classify a genuine connection failure as an abort', () => {
  const refused = new Error('connect ECONNREFUSED 127.0.0.1:11434');
  refused.name = 'Error';
  assert.equal(isAbortedRequestError(new TypeError('fetch failed', { cause: refused })), false);
});

test('does not classify an HTTP error or non-objects as an abort', () => {
  assert.equal(isAbortedRequestError(new Error('Ollama /api/chat failed with HTTP 500')), false);
  assert.equal(isAbortedRequestError(undefined), false);
  assert.equal(isAbortedRequestError(null), false);
  assert.equal(isAbortedRequestError('terminated'), false);
});
