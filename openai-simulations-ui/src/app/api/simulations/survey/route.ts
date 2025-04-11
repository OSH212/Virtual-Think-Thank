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

// Helper function to update parameters.json status for Survey
async function updateSurveyStatus(simulationId: string, status: "running" | "completed" | "failed", error?: string) {
    console.log(`Updating survey status for ${simulationId} to ${status}`);
    const parametersPath = path.join(process.cwd(), "public", "simulations", "survey", simulationId, "parameters.json");
    try {
        if (await fileExists(parametersPath)) {
            const params = JSON.parse(await fsPromises.readFile(parametersPath, 'utf8'));
            params.status = status;
            if (status !== 'running') {
                params.endTime = new Date().toISOString();
            }
            if (error) {
                params.error = error; // Store error message if provided
            }
            await fsPromises.writeFile(parametersPath, JSON.stringify(params, null, 2));
            console.log(`Survey Status updated successfully for ${simulationId}.`);
        } else {
            console.error(`Survey parameters.json not found for ${simulationId}. Cannot update status.`);
        }
    } catch (updateError) {
        console.error(`Error updating survey status for ${simulationId}:`, updateError);
    }
}


export async function POST(request: NextRequest) {
  let logStream: fs.WriteStream | null = null;
  let childProcess: ReturnType<typeof exec> | null = null;
  const simulationId = uuidv4(); // Generate ID early for error handling

  try {
    const body = await request.json();
    // Match names from the form/schema
    const { topic, researchObjectives, targetAudience, numRespondents } = body;

    // Input validation (basic) - align with schema
    if (!topic || !researchObjectives || !targetAudience || numRespondents === undefined) {
      return NextResponse.json({ error: "Missing required parameters: topic, researchObjectives, targetAudience, numRespondents" }, { status: 400 });
    }
    if (typeof numRespondents !== 'number' || numRespondents < 5 || numRespondents > 200) {
      return NextResponse.json({ error: "Invalid numRespondents parameter (must be between 5 and 200)" }, { status: 400 });
    }

    // Create directories to store results for Survey
    const simulationDir = path.join(process.cwd(), "public", "simulations", "survey", simulationId);
    await fsPromises.mkdir(simulationDir, { recursive: true });

    // Save simulation parameters (match Python script expected keys)
    const parametersPath = path.join(simulationDir, "parameters.json");
    await fsPromises.writeFile(
      parametersPath,
      JSON.stringify({
        id: simulationId,
        topic,
        research_objectives: researchObjectives, // Use underscore for Python
        target_audience: targetAudience,     // Use underscore for Python
        num_respondents: numRespondents,       // Use underscore for Python
        status: "running",
        startTime: new Date().toISOString(),
      }, null, 2)
    );

    // Log file for live stream
    const liveLogPath = path.join(simulationDir, "live.log");

    // Construct the path to the Python script
    const pythonScriptPath = path.join(process.cwd(), "..", "survey_simulation", "survey_simulation.py");
    // Path to the virtual environment's Python executable
    const pythonExecutable = path.join(process.cwd(), "..", "myenv", "bin", "python");

    // Construct command
    // Ensure arguments with spaces are properly quoted and escape internal quotes
    const commandArgs = [
        `"${pythonScriptPath}"`,
        `--simulation_id "${simulationId}"`,
        `--topic "${topic.replace(/"/g, '\\"')}"`,
        `--objectives "${researchObjectives.replace(/"/g, '\\"')}"`, // Match python arg
        `--audience "${targetAudience.replace(/"/g, '\\"')}"`,     // Match python arg
        `--respondents ${numRespondents}`                         // Match python arg
    ];
    const command = `cd "${path.join(process.cwd(), "..")}" && "${pythonExecutable}" ${commandArgs.join(" ")}`;
    console.log("Executing Survey command:", command);

    // Create a writable stream for the live log file
    logStream = fs.createWriteStream(liveLogPath, { flags: 'a' });

    logStream.on('error', (err) => {
        console.error(`Error writing to survey live log file ${liveLogPath}:`, err);
        updateSurveyStatus(simulationId, "failed", `Log stream error: ${err.message}`).catch(console.error);
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

    childProcess.on('exit', async (code, signal) => {
      console.log(`Survey Python process for ${simulationId} exited with code: ${code}, signal: ${signal}`);
      const status = code === 0 ? "completed" : "failed";
      let errorMessage = status === "failed" ? `Process exited with code ${code}` : undefined;
      if(signal) errorMessage += ` (received signal: ${signal})`;

      logStream?.end(); // Ensure stream is closed
      await updateSurveyStatus(simulationId, status, errorMessage);
    });

    childProcess.on('error', (err) => {
        console.error("Failed to start survey Python process:", err);
        logStream?.end();
        updateSurveyStatus(simulationId, "failed", `Process spawn error: ${err.message}`).catch(console.error);
    });

    return NextResponse.json({ simulationId }, { status: 201 });

  } catch (error: any) {
    console.error("Error starting survey simulation:", error);
    // Ensure status is marked as failed if error occurs before process starts
    await updateSurveyStatus(simulationId, "failed", `API route error: ${error.message || error}`);
    logStream?.end();
    return NextResponse.json({ error: `Failed to start survey simulation: ${error.message || error}` }, { status: 500 });
  }
}

// GET request to list all survey simulations
export async function GET() {
     try {
        const simulationsDir = path.join(process.cwd(), "public", "simulations", "survey");

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
               console.error(`Failed to read parameters for survey sim ${simId}:`, readError);
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
        console.error("Error getting survey simulations:", error);
        return NextResponse.json({ error: "Failed to get survey simulations" }, { status: 500 });
      }
} 