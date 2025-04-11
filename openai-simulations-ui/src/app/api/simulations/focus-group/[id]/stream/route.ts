import { NextRequest } from 'next/server';
import path from 'path';
import fs from 'fs/promises';
import fsSync from 'fs'; // For watchFile and existsSync
import { Readable } from "stream";
// Removed: import { getValidatedId } from '@/lib/params';

// SSE helper to format messages
function encodeSSE(data: any): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

// SSE helper for custom events
function encodeSSEEvent(event: string, data: any): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } } // Standard way to get params in Route Handler
) {
  let logFileWatcher: fsSync.StatWatcher | null = null;
  let paramsFileWatcher: fsSync.StatWatcher | null = null;
  let isClosed = false; // Flag to prevent multiple close attempts

  // Inline validation logic
  if (!params || !params.id) {
    return new Response(JSON.stringify({ error: "Missing simulation ID" }), { status: 400 });
  }
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  if (typeof id !== 'string' || id.trim() === '') {
     return new Response(JSON.stringify({ error: "Invalid simulation ID format" }), { status: 400 });
  }

    const simulationDir = path.join(process.cwd(), 'public', 'simulations', 'focus-group', id);
  const liveLogPath = path.join(simulationDir, "live.log");
  const parametersPath = path.join(simulationDir, "parameters.json");

  // Ensure simulation directory exists before proceeding
    try {
      await fs.access(simulationDir);
    } catch (error) {
    console.error(`Simulation directory not found for ID ${id}:`, error);
    return new Response(JSON.stringify({ error: 'Simulation not found' }), { status: 404 });
    }
    
    // Set headers for SSE
    const headers = new Headers({
      'Content-Type': 'text/event-stream',
      'Connection': 'keep-alive',
      'Cache-Control': 'no-cache',
    });
    
    const stream = new ReadableStream({
      async start(controller) {
      let lastLogSize = 0;
      let previousLogContent = '';
      let currentStatus = 'running';

      const cleanup = () => {
        if (isClosed) return;
        console.log(`Cleaning up stream for ${id}`);
        if (logFileWatcher) fsSync.unwatchFile(liveLogPath, logFileWatcher);
        if (paramsFileWatcher) fsSync.unwatchFile(parametersPath, paramsFileWatcher);
        logFileWatcher = null;
        paramsFileWatcher = null;
        try {
            if (!controller.desiredSize === null || controller.desiredSize && controller.desiredSize > 0) {
                 controller.close();
            }
        } catch (e) {
            console.warn(`Error closing controller for ${id}:`, e);
        }
        isClosed = true;
      };

      const sendLogUpdates = async () => {
        try {
          const stats = await fs.stat(liveLogPath);
          if (stats.size > lastLogSize) {
            const currentLogContent = await fs.readFile(liveLogPath, 'utf-8');
            const newContent = currentLogContent.substring(previousLogContent.length);
            const newLines = newContent.split('\n');

            newLines.forEach(line => {
              if (line.startsWith('STREAM: ')) {
                const content = line.substring(8);
                controller.enqueue(encodeSSE({ line: content }));
              }
            });
            previousLogContent = currentLogContent; // Update previous content
            lastLogSize = stats.size;
          }
        } catch (err: any) {
          // Log file might not exist yet, ignore ENOENT
          if (err.code !== 'ENOENT') {
            console.error(`Error reading live log for ${id}:`, err);
          }
        }
      };

      const checkStatus = async () => {
          if (isClosed) return; // Don't check status if already closing
          try {
              const paramsContent = await fs.readFile(parametersPath, 'utf8');
              const paramsData = JSON.parse(paramsContent);
              if (paramsData.status !== 'running' && currentStatus === 'running') {
                  console.log(`Simulation ${id} status changed to ${paramsData.status}, closing stream.`);
                  currentStatus = paramsData.status;
                  // Send final log updates before closing
                  await sendLogUpdates();
                  controller.enqueue(encodeSSEEvent("status", { status: currentStatus }));
                  cleanup();
              }
          } catch (err: any) {
              // Ignore if params file doesn't exist yet or parse error
               if (err.code !== 'ENOENT') {
                   console.warn(`Error reading/parsing parameters file for ${id}:`, err.message);
               }
          }
      };


      // --- Initial Setup ---
       try {
            // Send initial log content if any
            await sendLogUpdates();
            // Check initial status immediately
            await checkStatus();

            if (currentStatus === 'running') {
                 // --- Setup Watchers ---
                 logFileWatcher = fsSync.watchFile(liveLogPath, { interval: 1000 }, async (curr, prev) => {
                     if (curr.mtimeMs > prev.mtimeMs) {
                         await sendLogUpdates();
                     }
                 });

                 paramsFileWatcher = fsSync.watchFile(parametersPath, { interval: 1000 }, async (curr, prev) => {
                     if (curr.mtimeMs > prev.mtimeMs) {
                          await checkStatus();
                     }
                 });

                 // Cleanup on client disconnect
                 request.signal.addEventListener('abort', cleanup);
            } else {
                // If already completed/failed on initial check, close immediately
                cleanup();
            }

       } catch (startErr: any) {
            console.error(`Error during stream start for ${id}:`, startErr);
            controller.enqueue(encodeSSE({ error: `Stream setup failed: ${startErr.message}` }));
            cleanup();
       }
    },
    cancel(reason) {
        console.log(`Stream cancelled for ${id}. Reason:`, reason);
         if (logFileWatcher) fsSync.unwatchFile(liveLogPath, logFileWatcher);
         if (paramsFileWatcher) fsSync.unwatchFile(parametersPath, paramsFileWatcher);
         logFileWatcher = null;
         paramsFileWatcher = null;
         isClosed = true;
      }
    });
    
    return new Response(stream, { headers });
}