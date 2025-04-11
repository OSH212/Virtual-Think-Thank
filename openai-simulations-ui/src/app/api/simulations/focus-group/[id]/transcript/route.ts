import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import fsSync from "fs";
import path from "path";
import csvParser from 'csv-parser';
import { Readable } from "stream";
import { getValidatedId } from '@/lib/params';

// Helper to convert CSV to JSON
async function csvToJson(csvContent: string): Promise<any[]> {
  return new Promise((resolve, reject) => {
    const results: any[] = [];
    const stream = Readable.from([csvContent]);
    
    stream
      .pipe(csvParser())
      .on('data', (data) => results.push(data))
      .on('end', () => {
        resolve(results);
      })
      .on('error', (err) => {
        reject(err);
      });
  });
}

// Helper function to properly await dynamic params in Next.js 15
async function getValidatedParams(params: any): Promise<{ id: string }> {
  // Await and destructure the params object
  const resolvedParams = await Promise.resolve(params);
  return { id: resolvedParams.id };
}

// This is the correct signature for Next.js 15 API routes with dynamic params
export async function GET(request: NextRequest, props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  try {
    const id = getValidatedId(params);
    const simulationDir = path.join(process.cwd(), "public", "simulations", "focus-group", id);
    
    // Check if the simulation exists
    try {
      await fs.access(simulationDir);
    } catch (error) {
      return NextResponse.json({ error: "Simulation not found" }, { status: 404 });
    }
    
    // Check if we have a stored transcript
    const transcriptPath = path.join(simulationDir, "transcript.json");
    if (fsSync.existsSync(transcriptPath)) {
      const transcriptContent = await fs.readFile(transcriptPath, 'utf8');
      return NextResponse.json({ transcript: JSON.parse(transcriptContent) });
    }
    
    // If not, we need to check if the simulation is running and read the live transcript
    const paramsPath = path.join(simulationDir, "parameters.json");
    const paramsContent = await fs.readFile(paramsPath, 'utf8');
    const simulationParams = JSON.parse(paramsContent);
    
    if (simulationParams.status !== "running") {
      // Simulation is not running and no transcript found
      return NextResponse.json({ transcript: [] });
    }
    
    // Try to read transcript from the latest focus group output file
    try {
      // Find most recent transcript CSV in the root directory
      const rootDir = path.join(process.cwd(), "..");
      const files = await fs.readdir(rootDir);
      const transcriptFiles = files.filter(file => file.startsWith('focus_group_transcript_'));
      
      if (transcriptFiles.length === 0) {
        // No transcript files found
        return NextResponse.json({ transcript: [] });
      }
      
      // Sort by creation time, newest first
      const fileStats = await Promise.all(
        transcriptFiles.map(async file => ({
          file,
          stat: await fs.stat(path.join(rootDir, file))
        }))
      );
      
      fileStats.sort((a, b) => b.stat.mtime.getTime() - a.stat.mtime.getTime());
      const latestTranscript = fileStats[0].file;
      
      // Read and parse the CSV
      const csvContent = await fs.readFile(path.join(rootDir, latestTranscript), 'utf8');
      const rows = await csvToJson(csvContent);
      
      // Convert to our transcript format
      const transcript = rows.map(row => ({
        speaker: row.Speaker || "",
        text: row.Dialogue || ""
      }));
      
      // Store the transcript for future requests
      await fs.writeFile(transcriptPath, JSON.stringify(transcript));
      
      return NextResponse.json({ transcript });
    } catch (error) {
      console.error("Error reading transcript:", error);
      return NextResponse.json({ transcript: [] });
    }
  } catch (error) {
    console.error("Error:", error);
    return NextResponse.json({ error: "Failed to get transcript" }, { status: 500 });
  }
} 