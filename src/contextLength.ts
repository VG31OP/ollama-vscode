export const minimumMachineContextLength = 64 * 1024;

interface OllamaProcessModel {
  name?: unknown;
  model?: unknown;
  context_length?: unknown;
}

export function machineContextLength(
  models: readonly unknown[],
  requestedModel: string
): number | undefined {
  const requestedKey = modelKey(requestedModel);

  for (const candidate of models) {
    if (!isRecord(candidate)) {
      continue;
    }

    const model = candidate as OllamaProcessModel;
    const matches = [model.name, model.model].some(name =>
      typeof name === 'string' && modelKey(name) === requestedKey
    );
    if (matches && isPositiveInteger(model.context_length)) {
      return model.context_length;
    }
  }

  return undefined;
}

export async function waitForMachineContextLength(
  listModels: () => Promise<readonly unknown[]>,
  requestedModel: string,
  isRequestSettled: () => boolean,
  waitForNextCheck: () => Promise<boolean>
): Promise<number | undefined> {
  while (true) {
    const settledBeforeCheck = isRequestSettled();
    const contextLength = machineContextLength(await listModels(), requestedModel);
    if (settledBeforeCheck) {
      return contextLength;
    }

    if (isRequestSettled()) {
      // Ignore a result from a lookup that began before the request settled.
      // Check once more now that Ollama has finished loading the model.
      continue;
    }
    if (!await waitForNextCheck()) {
      return undefined;
    }
  }
}

export async function waitForMachineContextCheckBeforeStreaming(
  check: Promise<void>,
  waitForTimeout: () => Promise<void>,
  cancelCheck: () => void
): Promise<void> {
  let checkFinished = false;
  await Promise.race([
    check.finally(() => { checkFinished = true; }),
    waitForTimeout().then(() => {
      if (!checkFinished) {
        cancelCheck();
      }
    })
  ]);
}

export function isMachineContextTooSmall(contextLength: number): boolean {
  return contextLength < minimumMachineContextLength;
}

export function formatContextLength(contextLength: number): string {
  return contextLength % 1024 === 0
    ? `${contextLength / 1024}K`
    : contextLength.toLocaleString('en-US');
}

function modelKey(name: string): string {
  const normalized = name.trim().toLowerCase();
  return normalized.endsWith(':latest')
    ? normalized.slice(0, -':latest'.length)
    : normalized;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export const defaultOllamaContextLength = 4096;

export function resolveDynamicContextLength(
  estimatedPromptTokens: number,
  maxContextWindow: number,
  maxOutputTokens: number,
  configuredNumCtx?: unknown
): number | undefined {
  if (typeof configuredNumCtx === 'number' && Number.isSafeInteger(configuredNumCtx) && configuredNumCtx > 0) {
    return configuredNumCtx;
  }

  // If the prompt fits within Ollama's default context window, do not pass num_ctx,
  // preserving the server's default lightweight memory allocation.
  if (estimatedPromptTokens <= defaultOllamaContextLength) {
    return undefined;
  }

  // Allocate context rounded up to the nearest 1024, capped at the model's supported context window.
  const requiredTokens = estimatedPromptTokens + maxOutputTokens;
  const target = Math.ceil(requiredTokens / 1024) * 1024;
  return Math.min(maxContextWindow, target);
}

export function parseContextExceededError(
  error: unknown
): { requestedTokens: number; availableTokens: number } | undefined {
  const message = typeof error === 'string'
    ? error
    : error instanceof Error
    ? error.message
    : isRecord(error) && typeof error.message === 'string'
    ? error.message
    : String(error ?? '');
  const match = message.match(
    /request\s*\((\d+)\s*tokens\)\s*exceeds\s*the\s*available\s*context\s*size\s*\((\d+)\s*tokens\)/i
  );
  if (!match) {
    return undefined;
  }
  const requestedTokens = Number(match[1]);
  const availableTokens = Number(match[2]);
  if (Number.isSafeInteger(requestedTokens) && Number.isSafeInteger(availableTokens)) {
    return { requestedTokens, availableTokens };
  }
  return undefined;
}

export function formatContextExceededError(
  requestedTokens: number,
  contextWindow: number,
  modelName: string
): string {
  const requested = formatContextLength(requestedTokens);
  const limit = formatContextLength(contextWindow);
  return `Prompt (${requested} tokens) exceeds ${modelName}'s maximum context window (${limit}). Shorten the prompt or remove attachments to continue.`;
}
