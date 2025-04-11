import React from 'react';
import path from 'path';
import fs from 'fs/promises';
import { notFound } from 'next/navigation';
import { Metadata } from 'next';
import { Card, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import MarkdownRenderer from '@/components/markdown-renderer';
import StreamComponent from '@/components/stream-component';
import Image from 'next/image';
import { ReloadIcon } from "@radix-ui/react-icons"
import { VisualizationGallery } from '@/components/focus-group/visualization-gallery';
import { Vortex } from '@/components/ui/vortex';

interface PageProps {
  params: { id: string };
  searchParams?: { [key: string]: string | string[] | undefined };
}

// Type definition for parameters.json content
interface SimulationData {
  id: string;
  topic: string;
  targetAudience: string;
  numParticipants: number;
  numRounds: number;
  status: 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  error?: string;
}

// Function to get simulation data
async function getSimulationData(id: string): Promise<SimulationData | null> {
  try {
    const dataPath = path.join(process.cwd(), 'public', 'simulations', 'focus-group', id, 'parameters.json');
    const data = await fs.readFile(dataPath, 'utf8');
    return JSON.parse(data) as SimulationData;
  } catch (error) {
    console.error(`Error reading simulation data for ID ${id}:`, error);
    return null;
  }
}

// Function to get simulation report
async function getSimulationReport(id: string): Promise<string | null> {
  try {
    const reportPath = path.join(process.cwd(), 'public', 'simulations', 'focus-group', id, 'report.md');
    const report = await fs.readFile(reportPath, 'utf8');
    return report;
  } catch (error) {
    console.error(`Error reading simulation report for ID ${id}:`, error);
    return null;
  }
}

// Function to get visualization files
async function getVisualizationFiles(id: string): Promise<string[]> {
  try {
    const publicPath = `/simulations/focus-group/${id}/visualizations`;
    const serverPath = path.join(process.cwd(), 'public', publicPath);
    console.log(`Checking for visualizations at: ${serverPath}`);
    try {
      await fs.access(serverPath);
    } catch (error) {
      console.log(`Visualization directory not found: ${serverPath}`);
      return [];
    }
    const files = await fs.readdir(serverPath);
    const imageFiles = files.filter(file => /\.(png|jpg|jpeg|gif|svg)$/i.test(file));
    console.log(`Found ${imageFiles.length} visualization files`);
    return imageFiles.map(file => `${publicPath}/${file}`);
  } catch (error) {
    console.error(`Error reading visualization files: ${error}`);
    return [];
  }
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  if (!params || !params.id) {
    return { title: 'Invalid Simulation', description: 'Missing simulation ID.' };
  }
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  if (typeof id !== 'string' || id.trim() === '') {
    return { title: 'Invalid Simulation', description: 'Invalid simulation ID format.' };
  }

  try {
    const data = await getSimulationData(id);
    if (!data) {
      return { title: 'Simulation Not Found' };
    }
    return {
      title: `Focus Group Results: ${data.topic}`,
      description: `Results for focus group simulation on ${data.topic} with ${data.numParticipants} participants`,
    };
  } catch (error) {
    console.error("Error generating metadata for focus group:", error);
    return {
      title: 'Error Loading Simulation',
      description: 'There was an error loading the simulation metadata.'
    };
  }
}

export default async function FocusGroupResultsPage({ params }: PageProps) {
  if (!params || !params.id) {
    notFound();
  }
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  if (typeof id !== 'string' || id.trim() === '') {
    notFound();
  }

  const data = await getSimulationData(id);
  if (!data) {
    notFound();
  }

  return (
    <div className="relative min-h-screen"> 
      <Vortex
        backgroundColor="black"
        rangeY={800}
        particleCount={500}
        baseHue={200}
        containerClassName="fixed inset-0 w-full h-full z-0"
        className=""
      />

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-start overflow-y-auto py-10">
        <div className="container mx-auto w-full max-w-4xl px-4"> 
          {data.status === 'running' && (
            <div className="flex flex-col items-center justify-center space-y-4">
              <h1 className="text-3xl font-bold text-white">Simulation In Progress</h1>
              <p className="text-white/70">Topic: {data.topic}</p>
              <Alert className="max-w-md bg-neutral-600/40 border-neutral-500/50 text-neutral-100">
                <ReloadIcon className="h-5 w-5 animate-spin" />
                <AlertTitle className="text-neutral-100">Running Simulation...</AlertTitle>
                <AlertDescription className="text-neutral-200">
                  The focus group simulation is currently running. Results will appear here once completed.
                  You can monitor the live discussion below.
                </AlertDescription>
              </Alert>
              <Card className="w-full max-w-3xl mt-6 backdrop-blur-sm bg-card/80 dark:bg-card/80"> 
                <CardContent className="p-4">
                  <h2 className="text-xl font-semibold mb-2">Live Discussion Feed</h2>
                  <StreamComponent endpoint={`/api/simulations/focus-group/${id}/stream`} />
                </CardContent>
              </Card>
            </div>
          )}

          {data.status === 'failed' && (
            <div className="flex flex-col items-center justify-center space-y-4">
              <h1 className="text-3xl font-bold text-red-300">Simulation Failed</h1>
              <p className="text-white/70">Topic: {data.topic}</p>
              <Alert variant="destructive" className="max-w-md bg-red-900/60 border-red-600/70 text-red-100">
                <AlertTitle className="text-red-100">Error</AlertTitle>
                <AlertDescription className="text-red-200">
                  The focus group simulation failed to complete. Check server logs for details.
                  {data.error && <p className="mt-2"><strong>Details:</strong> {data.error}</p>}
                  Look for errors in the file: public/simulations/focus-group/{id}/live.log
                </AlertDescription>
              </Alert>
            </div>
          )}

          {data.status === 'completed' && (
            <div className="space-y-6 mb-10">
              <h1 className="text-3xl font-bold text-center text-white mb-6">{data.topic}</h1>

              {(() => {
                const reportPromise = getSimulationReport(id);
                const vizPromise = getVisualizationFiles(id);
                return (
                  <React.Suspense fallback={<p className="text-white text-center">Loading results...</p>}>
                    <CompletedResultsContent
                       data={data}
                       reportPromise={reportPromise}
                       vizPromise={vizPromise}
                    />
                  </React.Suspense>
                );
              })()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- Helper Component for Completed State ---
async function CompletedResultsContent({ data, reportPromise, vizPromise }: {
  data: SimulationData;
  reportPromise: Promise<string | null>;
  vizPromise: Promise<string[]>;
}) {
  const report = await reportPromise;
  const visualizationFiles = await vizPromise;

  const renderVisualizations = () => {
    if (!visualizationFiles || visualizationFiles.length === 0) {
      return null;
    }
    return (
       <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80"> 
           <CardContent className="p-4">
               <h2 className="text-xl font-semibold mb-4">Visualizations</h2>
               <VisualizationGallery images={visualizationFiles} /> 
           </CardContent>
       </Card>
    );
  };

  return (
    <>
      <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80"> 
        <CardContent className="p-4">
          <h2 className="text-xl font-semibold mb-2">Parameters</h2>
          <div className="space-y-1 text-sm">
            <p><strong>Target Audience:</strong> {data.targetAudience}</p>
            <p><strong>Participants:</strong> {data.numParticipants}</p>
            <p><strong>Rounds:</strong> {data.numRounds}</p>
            <p><strong>Status:</strong> <span className="font-semibold">{data.status}</span></p>
            {data.startTime && (
              <p><strong>Started:</strong> {new Date(data.startTime).toLocaleString()}</p>
            )}
            {data.endTime && (
              <p><strong>Completed:</strong> {new Date(data.endTime).toLocaleString()}</p>
            )}
          </div>
        </CardContent>
      </Card>

      {report ? (
        <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80"> 
          <CardContent className="p-4">
            <h2 className="text-xl font-semibold mb-4">Analysis Report</h2>
            <div className="prose max-w-none dark:prose-invert"> 
               <MarkdownRenderer content={report} />
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80"> 
           <CardContent className="p-4">
               <p className="text-muted-foreground">Analysis report not available.</p>
           </CardContent>
        </Card>
      )}

      {renderVisualizations()}
    </>
  );
}
