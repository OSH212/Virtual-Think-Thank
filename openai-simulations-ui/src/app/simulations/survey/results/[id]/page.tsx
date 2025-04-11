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
import { ReloadIcon } from "@radix-ui/react-icons";
import { VisualizationGallery } from '@/components/focus-group/visualization-gallery'; // Reusing gallery component
import { Vortex } from '@/components/ui/vortex'; // Import Vortex

// --- Types (Adapted for Survey) ---
interface PageProps {
  params: { id: string };
  searchParams?: { [key: string]: string | string[] | undefined };
}

interface SurveySimulationData {
  id: string;
  topic: string;
  research_objectives: string; // Python script uses underscores
  target_audience: string; // Python script uses underscores
  num_respondents: number; // Python script uses underscores
  status: 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  error?: string; // Added to potentially display errors from parameters.json
}

// --- Helper Functions (Adapted for Survey) ---
async function getSurveySimulationData(id: string): Promise<SurveySimulationData | null> {
  try {
    const dataPath = path.join(process.cwd(), 'public', 'simulations', 'survey', id, 'parameters.json');
    const data = await fs.readFile(dataPath, 'utf8');
    return JSON.parse(data) as SurveySimulationData;
  } catch (error) {
    console.error(`Error reading survey simulation data for ID ${id}:`, error);
    return null;
  }
}

async function getSurveySimulationReport(id: string): Promise<string | null> {
  try {
    const reportPath = path.join(process.cwd(), 'public', 'simulations', 'survey', id, 'report.md');
    const report = await fs.readFile(reportPath, 'utf8');
    return report;
  } catch (error) {
    console.error(`Error reading survey simulation report for ID ${id}:`, error);
    return null;
  }
}

async function getSurveyVisualizationFiles(id: string): Promise<string[]> {
  try {
    const publicPath = `/simulations/survey/${id}/visualizations`;
    const serverPath = path.join(process.cwd(), 'public', publicPath);

    console.log(`Checking for survey visualizations at: ${serverPath}`);

    try {
      await fs.access(serverPath);
    } catch (error) {
      console.log(`Survey Visualization directory not found: ${serverPath}`);
      return [];
    }

    const files = await fs.readdir(serverPath);
    // Filter for common image types, adjust if needed
    const imageFiles = files.filter(file => /\.(png|jpg|jpeg|gif|svg)$/i.test(file));
    console.log(`Found ${imageFiles.length} survey visualization files`);
    return imageFiles.map(file => `${publicPath}/${file}`);
  } catch (error) {
    console.error(`Error reading survey visualization files: ${error}`);
    return [];
  }
}

// --- Metadata Generation (Adapted for Survey) ---
export async function generateMetadata({ params }: { params: { id: string } }): Promise<Metadata> {
  const { id } = params;
  if (!id || typeof id !== 'string' || id.trim() === '') {
    return { title: 'Invalid Simulation' };
  }

  try {
    const data = await getSurveySimulationData(id);
    if (!data) return { title: 'Simulation Not Found' };
    return {
      title: `Survey Results: ${data.topic}`,
      description: `Results for survey simulation on ${data.topic}`,
    };
  } catch (error) {
    console.error("Error generating metadata for survey:", error);
    return { title: 'Error Loading Simulation' };
  }
}

// --- Main Page Component (Adapted for Survey) ---
export default async function SurveyResultsPage({ params }: PageProps) {
  const { id } = params;
  if (!id || typeof id !== 'string' || id.trim() === '') {
     notFound();
  }

  const data = await getSurveySimulationData(id);
  if (!data) notFound();

  // Remove the solid background class, add relative positioning
  return (
    <div className="relative min-h-screen"> 
      {/* Add fixed Vortex background */}
      <Vortex
        backgroundColor="black"
        rangeY={800}
        particleCount={500}
        baseHue={240} // Matching setup page hue
        containerClassName="fixed inset-0 w-full h-full z-0"
        className=""
      />

      {/* Add scrollable content container */}
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-start overflow-y-auto py-10">
        {/* Container for max-width and centering */}
        <div className="container mx-auto w-full max-w-4xl px-4"> 
          {data.status === 'running' && (
            <div className="flex flex-col items-center justify-center space-y-4">
              <h1 className="text-3xl font-bold text-white">Survey Simulation In Progress</h1>
              <p className="text-white/70">Topic: {data.topic}</p>
              <Alert className="max-w-md bg-neutral-600/40 border-neutral-500/50 text-neutral-100">
                <ReloadIcon className="h-5 w-5 animate-spin" />
                <AlertTitle className="text-neutral-100">Running Simulation...</AlertTitle>
                <AlertDescription className="text-neutral-200">
                  The survey simulation is currently running. Results will appear here once completed.
                  You can monitor the progress below.
                </AlertDescription>
              </Alert>
              <Card className="w-full max-w-3xl mt-6 backdrop-blur-sm bg-card/80 dark:bg-card/80"> 
                <CardContent className="p-4">
                  <h2 className="text-xl font-semibold mb-2">Live Simulation Feed</h2>
                   {/* Point StreamComponent to the Survey stream endpoint */}
                  <StreamComponent endpoint={`/api/simulations/survey/${id}/stream`} />
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
                  The survey simulation failed to complete. Check server logs for details.
                  {data.error && <p className="mt-2"><strong>Details:</strong> {data.error}</p>}
                  Look for detailed errors in the file: public/simulations/survey/{id}/live.log
                </AlertDescription>
              </Alert>
            </div>
          )}

          {data.status === 'completed' && (
            <div className="space-y-6 mb-10"> {/* Added mb-10 for bottom spacing */}
              <h1 className="text-3xl font-bold text-center text-white mb-6">{data.topic}</h1>
              {(() => {
                const reportPromise = getSurveySimulationReport(id);
                const vizPromise = getSurveyVisualizationFiles(id);
                return (
                  <React.Suspense fallback={<p className="text-white text-center">Loading results...</p>}>
                    <SurveyCompletedResultsContent
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

// --- Helper Component for Completed State (Adapted for Survey) ---
async function SurveyCompletedResultsContent({ data, reportPromise, vizPromise }: {
  data: SurveySimulationData;
  reportPromise: Promise<string | null>;
  vizPromise: Promise<string[]>;
}) {
  const report = await reportPromise;
  const visualizationFiles = await vizPromise;

  const renderVisualizations = () => {
    if (!visualizationFiles || visualizationFiles.length === 0) return null;
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
             {/* Display survey specific parameters */}
            <p><strong>Research Objectives:</strong> {data.research_objectives}</p>
            <p><strong>Target Audience:</strong> {data.target_audience}</p>
            <p><strong>Number of Respondents:</strong> {data.num_respondents}</p>
            <p><strong>Status:</strong> <span className="font-semibold">{data.status}</span></p>
            {data.startTime && (<p><strong>Started:</strong> {new Date(data.startTime).toLocaleString()}</p>)}
            {data.endTime && (<p><strong>Completed:</strong> {new Date(data.endTime).toLocaleString()}</p>)}
          </div>
        </CardContent>
      </Card>

      {report ? (
        <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80"> 
          <CardContent className="p-4">
            <h2 className="text-xl font-semibold mb-4">Analysis Report</h2>
            <div className="prose max-w-none dark:prose-invert"> {/* Added dark:prose-invert for dark bg */}
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