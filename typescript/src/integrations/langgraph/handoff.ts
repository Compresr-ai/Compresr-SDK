/**
 * `compresrHandoffTool` — supervisor → subagent handoff tool with
 * compression baked in for LangGraph.js multi-agent systems.
 *
 * Returns a LangChain `DynamicStructuredTool` that emits a `Command`
 * routing control to `agentName` with a compressed `task_description`
 * and (optionally) compressed `context` field.
 */
import { ToolMessage } from '@langchain/core/messages';
import { tool, type ToolRuntime } from '@langchain/core/tools';
import { Command } from '@langchain/langgraph';
import { z } from 'zod';

import type { CompressionClient } from '../../clients/compression.js';
import {
  buildClient,
  compressSafe,
  DEFAULT_MIN_TOKENS,
  DEFAULT_MODEL,
  DEFAULT_POLICY,
  DEFAULT_RATIO,
  type ErrorPolicy,
} from '../_shared/index.js';

export interface CompresrHandoffToolOptions {
  description?: string;
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  compressionModel?: string;
  targetCompressionRatio?: number;
  minTokens?: number;
  coarse?: boolean;
  onError?: ErrorPolicy;
}

/**
 * Build a handoff tool for `agentName`.
 *
 * @example
 * const supervisor = createAgent({
 *   model,
 *   tools: [
 *     compresrHandoffTool('researcher', { apiKey: process.env.COMPRESR_API_KEY }),
 *     compresrHandoffTool('writer',     { apiKey: process.env.COMPRESR_API_KEY }),
 *   ],
 * });
 */
export function compresrHandoffTool(agentName: string, options: CompresrHandoffToolOptions = {}) {
  const {
    description,
    apiKey,
    client,
    baseUrl,
    compressionModel = DEFAULT_MODEL,
    targetCompressionRatio = DEFAULT_RATIO,
    minTokens = DEFAULT_MIN_TOKENS,
    coarse,
    onError = DEFAULT_POLICY,
  } = options;

  const compresr = client ?? buildClient({ apiKey, baseUrl, caller: 'compresrHandoffTool' });
  const toolName = `transfer_to_${agentName}`;
  const toolDescription =
    description ??
    `Hand off the task to the \`${agentName}\` agent. ` +
      'Provide a clear task_description and any context the agent needs.';

  const schema = z.object({
    task_description: z.string().describe(`Task description for the ${agentName} agent.`),
    context: z
      .string()
      .optional()
      .default('')
      .describe('Optional context excerpts to share with the agent.'),
  });

  async function maybeCompress(text: string, query: string, label: string): Promise<string> {
    if (!text) return text;
    return await compressSafe(compresr, {
      context: text,
      query,
      compressionModel,
      targetCompressionRatio,
      ...(coarse !== undefined ? { coarse } : {}),
      minTokens,
      onError,
      contextLabel: label,
    });
  }

  return tool(
    async (
      input: { task_description: string; context?: string },
      runtime: ToolRuntime
    ): Promise<Command> => {
      // latte_v1 requires a query — the task description itself is the natural
      // one for the context excerpt (what the subagent will do with it).
      const compressedTask = await maybeCompress(
        input.task_description,
        `task for ${agentName}`,
        'handoff:task',
      );
      const ctx = input.context ?? '';
      const compressedCtx = ctx
        ? await maybeCompress(
            ctx,
            input.task_description || `context for ${agentName}`,
            'handoff:context',
          )
        : '';

      const ack = new ToolMessage({
        content: `Successfully transferred to ${agentName}.`,
        tool_call_id: runtime.toolCallId ?? '',
        name: toolName,
      });

      return new Command({
        goto: agentName,
        update: {
          messages: [ack],
          task_description: compressedTask,
          context: compressedCtx,
        },
        graph: Command.PARENT,
      });
    },
    {
      name: toolName,
      description: toolDescription,
      schema,
    }
  );
}
