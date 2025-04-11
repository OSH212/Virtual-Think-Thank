import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import fsPromises from "fs/promises";
import fs from "fs"; // Use fs for createWriteStream
import path from "path";
import { v4 as uuidv4 } from "uuid";

const execPromise = promisify(exec);

// Helper function to check if a file exists
async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fsPromises.access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { topic, targetAudience, numParticipants, numRounds } = body;
    
    // Generate a unique ID for this simulation
    const simulationId = uuidv4();
    
    // Create directories to store results
    const simulationDir = path.join(process.cwd(), "public", "simulations", "focus-group", simulationId);
    await fsPromises.mkdir(simulationDir, { recursive: true });
    
    // The Python script will create the 'visualizations' sub-directory if needed
    // const vizDir = path.join(simulationDir, "visualizations");
    // await fs.mkdir(vizDir, { recursive: true }); // No longer strictly necessary here
    
    // Save simulation parameters
    const parametersPath = path.join(simulationDir, "parameters.json");
    await fsPromises.writeFile(
      parametersPath,
      JSON.stringify({
        id: simulationId,
        topic,
        targetAudience,
        numParticipants,
        numRounds,
        status: "running",
        startTime: new Date().toISOString(),
      }, null, 2)
    );
    
    // Log file for live stream
    const liveLogPath = path.join(simulationDir, "live.log");
    
    // Construct the path to the Python script relative to the current working directory (openai-simulations-ui)
    const pythonScriptPath = path.join(process.cwd(), "..", "focus_group_simulation", "focus_group_simulation.py");
    // Path to the virtual environment's Python executable
    const pythonExecutable = path.join(process.cwd(), "..", "myenv", "bin", "python"); // Adjust if your venv path differs
    
    // Construct command to run Python script from the parent directory (project root)
    const commandArgs = [`"${pythonScriptPath}"`, `--simulation_id "${simulationId}"`, `--topic "${topic}"`, `--target_audience "${targetAudience}"`, `--num_participants ${numParticipants}`, `--num_rounds ${numRounds}`];
    const command = `cd "${path.join(process.cwd(), "..")}" && "${pythonExecutable}" ${commandArgs.join(" ")}`;
    console.log("Executing command:", command);
    
    // Create a writable stream for the live log file
    const logStream = fs.createWriteStream(liveLogPath, { flags: 'a' }); // Append mode

    logStream.on('error', (err) => {
        console.error(`Error writing to live log file ${liveLogPath}:`, err);
        // Optionally try to update status to failed
        updateStatus(simulationId, "failed").catch(console.error);
        // Attempt to kill the child process if log writing fails?
        if (childProcess && !childProcess.killed) {
          childProcess.kill();
        }
    });

    // Execute the command
    const childProcess = exec(command);

    // --- Stream stdout/stderr to the live log file --- 
    if (childProcess.stdout) {
        childProcess.stdout.pipe(logStream);
    } else {
        console.error("Child process stdout stream not available.");
    }

    if (childProcess.stderr) {
        // Also pipe stderr to the log for debugging
        childProcess.stderr.pipe(logStream);
    } else {
        console.error("Child process stderr stream not available.");
    }
    // ---------------------------------------------------- 

    // When process completes, update the status
    childProcess.on('exit', async (code) => {
      console.log(`Python process for ${simulationId} exited with code: ${code}`);
      const status = code === 0 ? "completed" : "failed";
      
      try {
        // Ensure log stream is closed
        logStream.end();
        await updateStatus(simulationId, status);
      } catch (error) {
        console.error("Error updating simulation status in exit handler:", error);
      }
    });

    // Handle errors during process spawning itself
    childProcess.on('error', (err) => {
        console.error("Failed to start Python process:", err);
        logStream.end(); // Ensure log stream is closed on error too
        updateStatus(simulationId, "failed").catch(console.error); // Attempt to update status
    });

    return NextResponse.json({ simulationId }, { status: 201 });
  } catch (error) {
    console.error("Error starting simulation:", error);
    return NextResponse.json({ error: "Failed to start simulation" }, { status: 500 });
  }
}

export async function GET() {
  // Return list of all completed simulations
  try {
    const simulationsDir = path.join(process.cwd(), "public", "simulations", "focus-group");
    
    // Check if directory exists
    try {
      await fsPromises.access(simulationsDir);
    } catch {
      await fsPromises.mkdir(simulationsDir, { recursive: true });
      return NextResponse.json({ simulations: [] }, { status: 200 });
    }
    
    const simulations = await fsPromises.readdir(simulationsDir);
    const simulationData = await Promise.all(
      simulations.map(async (sim) => {
        try {
          const paramsPath = path.join(simulationsDir, sim, "parameters.json");
          const paramsContent = await fsPromises.readFile(paramsPath, 'utf8');
          return JSON.parse(paramsContent);
        } catch {
          return null;
        }
      })
    );
    
    // Filter out null values and sort by startTime descending
    const validSimulations = simulationData
      .filter(Boolean)
      .sort((a, b) => new Date(b.startTime).getTime() - new Date(a.startTime).getTime());
    
    return NextResponse.json({ simulations: validSimulations }, { status: 200 });
  } catch (error) {
    console.error("Error getting simulations:", error);
    return NextResponse.json({ error: "Failed to get simulations" }, { status: 500 });
  }
}

// Helper function to update parameters.json status
async function updateStatus(simulationId: string, status: "running" | "completed" | "failed") {
    console.log(`Updating status for ${simulationId} to ${status}`);
    const parametersPath = path.join(process.cwd(), "public", "simulations", "focus-group", simulationId, "parameters.json");
    if (await fileExists(parametersPath)) {
        const params = JSON.parse(await fsPromises.readFile(parametersPath, 'utf8'));
        params.status = status;
        params.endTime = new Date().toISOString();
        await fsPromises.writeFile(parametersPath, JSON.stringify(params, null, 2));
        console.log(`Status updated successfully for ${simulationId}.`);
    } else {
        console.error(`parameters.json not found for ${simulationId}. Cannot update status.`);
    }
}
