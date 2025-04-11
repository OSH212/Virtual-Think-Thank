import { NextRequest } from 'next/server';
import path from 'path';
import fs from 'fs/promises';
import fsSync from 'fs';
import { Readable } from "stream";

// SSE helpers (same as other streams)
function encodeSSE(data: any): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}
function encodeSSEEvent(event: string, data: any): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  let logFileWatcher: fsSync.StatWatcher | null = null;
  let paramsFileWatcher: fsSync.StatWatcher | null = null;
  let isClosed = false;

  // --- ID Validation (Inline) ---
  if (!params || !params.id) {
    return new Response(JSON.stringify({ error: "Missing simulation ID" }), { status: 400 });
  }
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  if (typeof id !== 'string' || id.trim() === '') {
     return new Response(JSON.stringify({ error: "Invalid simulation ID format" }), { status: 400 });
  }
  // --- End Validation ---

  // --- Paths (Adapted for Survey) ---
  const simulationDir = path.join(process.cwd(), 'public', 'simulations', 'survey', id);
  const liveLogPath = path.join(simulationDir, "live.log");
  const parametersPath = path.join(simulationDir, "parameters.json");
  // --- End Paths ---

  try {
      await fs.access(simulationDir);
    } catch (error) {
    console.error(`Survey Simulation directory not found for ID ${id}:`, error);
    return new Response(JSON.stringify({ error: 'Survey Simulation not found' }), { status: 404 });
  }

  const headers = new Headers({
      'Content-Type': 'text/event-stream',
      'Connection': 'keep-alive',
      'Cache-Control': 'no-cache',
  });

  const stream = new ReadableStream({
    async start(controller) {
      let lastLogSize = 0;
      let previousLogContent = ''; // Store previous content to send only diffs
      let currentStatus = 'running';

      const cleanup = () => {
        if (isClosed) return;
        console.log(`Cleaning up survey stream for ${id}`);
        if (logFileWatcher) fsSync.unwatchFile(liveLogPath, logFileWatcher);
        if (paramsFileWatcher) fsSync.unwatchFile(parametersPath, paramsFileWatcher);
        logFileWatcher = null;
        paramsFileWatcher = null;
        try {
           if (!controller.desiredSize === null || (controller.desiredSize && controller.desiredSize > 0)) {
                 controller.close();
           }
        } catch (e) {
            console.warn(`Error closing survey controller for ${id}:`, e);
        }
        isClosed = true;
      };

      // Function to send log updates (send only new lines)
      const sendLogUpdates = async () => {
         if (isClosed) return;
         try {
             await fs.access(liveLogPath); // Check existence
             const stats = await fs.stat(liveLogPath);

             if (stats.size > lastLogSize || !previousLogContent) { // Read if size changed or initial load
                const currentLogContent = await fs.readFile(liveLogPath, 'utf-8');
                let newContent = currentLogContent;
                if(previousLogContent) {
                   newContent = currentLogContent.substring(previousLogContent.length);
                }

                const newLines = newContent.split('\n');

                newLines.forEach(line => {
                   if(line.trim()){ // Send non-empty lines
                     // Python script prefixes output with "STREAM: "
                     if (line.startsWith('STREAM: ')) {
                        controller.enqueue(encodeSSE({ line: line.substring(8).trim() }));
                     } else {
                         // If no prefix, send the raw line (might be an error or unexpected output)
                         // Consider filtering or handling differently if needed
                         controller.enqueue(encodeSSE({ line: line.trim() }));
                     }
                   }
                });
                previousLogContent = currentLogContent;
                lastLogSize = stats.size;
             }
         } catch (err: any) {
             if (err.code !== 'ENOENT') { // Ignore "file not found" initially
                console.error(`Error reading survey live log for ${id}:`, err);
             }
         }
      };

      // Function to check parameters.json for status changes
      const checkStatus = async () => {
          if (isClosed) return;
          try {
              await fs.access(parametersPath); // Check existence first
              const paramsContent = await fs.readFile(parametersPath, 'utf8');
              const paramsData = JSON.parse(paramsContent);
              if (paramsData.status !== 'running' && currentStatus === 'running') {
                  console.log(`Survey Simulation ${id} status changed to ${paramsData.status}, closing stream.`);
                  currentStatus = paramsData.status;
                  await sendLogUpdates(); // Send final log updates
                  controller.enqueue(encodeSSEEvent("status", { status: currentStatus }));
                  cleanup();
              }
          } catch (err: any) {
               if (err.code !== 'ENOENT') {
                   console.warn(`Error reading/parsing survey parameters file for ${id}:`, err.message);
               }
          }
      };

      // --- Initial Setup & Watchers ---
       try {
            await sendLogUpdates(); // Send initial content
            await checkStatus(); // Check initial status

            if (currentStatus === 'running') {
                 logFileWatcher = fsSync.watchFile(liveLogPath, { interval: 1000 }, async (curr, prev) => {
                     if (curr.mtimeMs > prev.mtimeMs || curr.size !== prev.size) { // Check size too
                         await sendLogUpdates();
                     }
                 });

                 paramsFileWatcher = fsSync.watchFile(parametersPath, { interval: 1000 }, async (curr, prev) => {
                     if (curr.mtimeMs > prev.mtimeMs) {
                          await checkStatus();
                     }
                 });
                 request.signal.addEventListener('abort', cleanup); // Client disconnect
            } else {
                cleanup(); // Already completed/failed
            }
       } catch (startErr: any) {
            console.error(`Error during survey stream start for ${id}:`, startErr);
            controller.enqueue(encodeSSE({ error: `Stream setup failed: ${startErr.message}` }));
            cleanup();
       }
    },
    cancel(reason) {
        console.log(`Survey Stream cancelled for ${id}. Reason:`, reason);
        cleanup(); // Ensure cleanup happens on cancel
      }
    });

    return new Response(stream, { headers });
} 