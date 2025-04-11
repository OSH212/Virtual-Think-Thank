import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import fsPromises from "fs/promises";
import fs from "fs";
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

// Helper function to update parameters.json status for IDI
async function updateIdiStatus(simulationId: string, status: "running" | "completed" | "failed") {
    console.log(`Updating IDI status for ${simulationId} to ${status}`);
    const parametersPath = path.join(process.cwd(), "public", "simulations", "idi", simulationId, "parameters.json");
    try {
        if (await fileExists(parametersPath)) {
            const params = JSON.parse(await fsPromises.readFile(parametersPath, 'utf8'));
            params.status = status;
            if (status !== 'running') {
                params.endTime = new Date().toISOString();
            }
            await fsPromises.writeFile(parametersPath, JSON.stringify(params, null, 2));
            console.log(`IDI Status updated successfully for ${simulationId}.`);
        } else {
            console.error(`IDI parameters.json not found for ${simulationId}. Cannot update status.`);
        }
    } catch (error) {
        console.error(`Error updating IDI status for ${simulationId}:`, error);
    }
}


export async function POST(request: NextRequest) {
  let logStream: fs.WriteStream | null = null;
  let childProcess: ReturnType<typeof exec> | null = null;
  const simulationId = uuidv4(); // Generate ID early for error handling

  try {
    const body = await request.json();
    const { topic, targetAudience, numQuestions } = body;
    
    // Input validation (basic)
    if (!topic || !targetAudience || !numQuestions) {
      return NextResponse.json({ error: "Missing required parameters: topic, targetAudience, numQuestions" }, { status: 400 });
    }
    if (typeof numQuestions !== 'number' || numQuestions < 3 || numQuestions > 15) {
      return NextResponse.json({ error: "Invalid numQuestions parameter (must be between 3 and 15)" }, { status: 400 });
    }

    // Create directories to store results for IDI
    const simulationDir = path.join(process.cwd(), "public", "simulations", "idi", simulationId);
    await fsPromises.mkdir(simulationDir, { recursive: true });
    
    // Save simulation parameters
    const parametersPath = path.join(simulationDir, "parameters.json");
    await fsPromises.writeFile(
      parametersPath,
      JSON.stringify({
        id: simulationId,
        topic,
        targetAudience,
        numQuestions,
        status: "running",
        startTime: new Date().toISOString(),
      }, null, 2)
    );
    
    // Log file for live stream
    const liveLogPath = path.join(simulationDir, "live.log");
    
    // Construct the path to the Python script 
    const pythonScriptPath = path.join(process.cwd(), "..", "idi_simulation", "idi_simulation.py");
    // Path to the virtual environment's Python executable
    const pythonExecutable = path.join(process.cwd(), "..", "myenv", "bin", "python"); 
    
    // Construct command 
    // Ensure arguments with spaces are properly quoted
    const commandArgs = [
        `"${pythonScriptPath}"`, 
        `--simulation_id "${simulationId}"`, 
        `--topic "${topic.replace(/"/g, '\\"')}"`, // Escape potential quotes in topic
        `--target_audience "${targetAudience.replace(/"/g, '\\"')}"`, // Escape potential quotes in audience
        `--num_questions ${numQuestions}`
    ];
    const command = `cd "${path.join(process.cwd(), "..")}" && "${pythonExecutable}" ${commandArgs.join(" ")}`;
    console.log("Executing IDI command:", command);
    
    // Create a writable stream for the live log file
    logStream = fs.createWriteStream(liveLogPath, { flags: 'a' }); 

    logStream.on('error', (err) => {
        console.error(`Error writing to IDI live log file ${liveLogPath}:`, err);
        updateIdiStatus(simulationId, "failed").catch(console.error);
        if (childProcess && !childProcess.killed) {
          childProcess.kill();
        }
    });

    // Execute the command
    childProcess = exec(command);

    // --- Stream stdout/stderr to the live log file --- 
    childProcess.stdout?.pipe(logStream);
    childProcess.stderr?.pipe(logStream); // Pipe stderr as well for debugging
    // ---------------------------------------------------- 

    childProcess.on('exit', async (code) => {
      console.log(`IDI Python process for ${simulationId} exited with code: ${code}`);
      const status = code === 0 ? "completed" : "failed";
      logStream?.end(); // Ensure stream is closed
      await updateIdiStatus(simulationId, status);
    });

    childProcess.on('error', (err) => {
        console.error("Failed to start IDI Python process:", err);
        logStream?.end(); 
        updateIdiStatus(simulationId, "failed").catch(console.error); 
    });

    return NextResponse.json({ simulationId }, { status: 201 });

  } catch (error) {
    console.error("Error starting IDI simulation:", error);
    // Ensure status is marked as failed if error occurs before process starts
    await updateIdiStatus(simulationId, "failed"); 
    logStream?.end();
    return NextResponse.json({ error: "Failed to start IDI simulation" }, { status: 500 });
  }
}

// Optional: Implement GET to list IDI simulations if needed, similar to focus group
export async function GET() {
    // Similar logic to focus group GET, but listing from /public/simulations/idi
     try {
        const simulationsDir = path.join(process.cwd(), "public", "simulations", "idi");
        
        try {
          await fsPromises.access(simulationsDir);
        } catch {
          // If the directory doesn't exist, create it and return empty list
          await fsPromises.mkdir(simulationsDir, { recursive: true });
          return NextResponse.json({ simulations: [] }, { status: 200 });
        }
        
        const simulationIds = await fsPromises.readdir(simulationsDir);
        const simulationDataPromises = simulationIds.map(async (simId) => {
            try {
              const paramsPath = path.join(simulationsDir, simId, "parameters.json");
              const paramsContent = await fsPromises.readFile(paramsPath, 'utf8');
              return JSON.parse(paramsContent);
            } catch (readError) {
               console.error(`Failed to read parameters for IDI sim ${simId}:`, readError);
               return null; // Skip sims with missing/corrupt parameter files
            }
        });
        
        const simulationData = await Promise.all(simulationDataPromises);
        
        // Filter out null values and sort by startTime descending
        const validSimulations = simulationData
          .filter(Boolean) // Remove null entries
          .sort((a, b) => new Date(b.startTime).getTime() - new Date(a.startTime).getTime());
        
        return NextResponse.json({ simulations: validSimulations }, { status: 200 });
      } catch (error) {
        console.error("Error getting IDI simulations:", error);
        return NextResponse.json({ error: "Failed to get IDI simulations" }, { status: 500 });
      }
} 