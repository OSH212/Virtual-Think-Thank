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
import { VisualizationGallery } from '@/components/focus-group/visualization-gallery'; // Can reuse if needed
import { Vortex } from '@/components/ui/vortex'; // Import Vortex

// --- Types (Adjust if IDI params differ significantly) ---
interface PageProps {
  params: { id: string };
  searchParams?: { [key: string]: string | string[] | undefined };
}

interface IdiSimulationData {
  id: string;
  topic: string;
  targetAudience: string;
  numQuestions: number; // Different from focus group
  status: 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  error?: string; // Add error field
}

// --- Helper Functions (Adapted for IDI) ---
async function getIdiSimulationData(id: string): Promise<IdiSimulationData | null> {
  try {
    // Adjusted path for IDI
    const dataPath = path.join(process.cwd(), 'public', 'simulations', 'idi', id, 'parameters.json'); 
    const data = await fs.readFile(dataPath, 'utf8');
    return JSON.parse(data) as IdiSimulationData;
  } catch (error) {
    console.error(`Error reading IDI simulation data for ID ${id}:`, error);
    return null;
  }
}

async function getIdiSimulationReport(id: string): Promise<string | null> {
  try {
     // Adjusted path for IDI
    const reportPath = path.join(process.cwd(), 'public', 'simulations', 'idi', id, 'report.md');
    const report = await fs.readFile(reportPath, 'utf8');
    return report;
  } catch (error) {
    console.error(`Error reading IDI simulation report for ID ${id}:`, error);
    return null;
  }
}

async function getIdiVisualizationFiles(id: string): Promise<string[]> {
  try {
     // Adjusted path for IDI
    const publicPath = `/simulations/idi/${id}/visualizations`;
    const serverPath = path.join(process.cwd(), 'public', publicPath);
    
    console.log(`Checking for IDI visualizations at: ${serverPath}`);
    
    try {
      await fs.access(serverPath);
    } catch (error) {
      console.log(`IDI Visualization directory not found: ${serverPath}`);
      return [];
    }
    
    const files = await fs.readdir(serverPath);
    const imageFiles = files.filter(file => /\.(png|jpg|jpeg|gif|svg)$/i.test(file));
    console.log(`Found ${imageFiles.length} IDI visualization files`);
    return imageFiles.map(file => `${publicPath}/${file}`);
  } catch (error) {
    console.error(`Error reading IDI visualization files: ${error}`);
    return [];
  }
}

// --- Metadata Generation (FIXED) ---
export async function generateMetadata({ params }: { params: { id: string } }): Promise<Metadata> {
  // Destructure params directly as it's guaranteed by Next.js in generateMetadata
  const { id } = params; 

  if (!id || typeof id !== 'string' || id.trim() === '') {
    // Handle invalid ID early
    return { title: 'Invalid Simulation' };
  }

  try {
    const data = await getIdiSimulationData(id); // Use awaited ID
    if (!data) return { title: 'Simulation Not Found' };
    return {
      title: `IDI Results: ${data.topic}`,
      description: `Results for in-depth interview on ${data.topic}`,
    };
  } catch (error) {
    console.error("Error generating metadata:", error); // Log error
    return { title: 'Error Loading Simulation' };
  }
}

// --- Main Page Component (Adapted for IDI) ---
export default async function IdiResultsPage({ params }: PageProps) {
   // Destructure ID directly here as well for consistency
  const { id } = params; 
  if (!id || typeof id !== 'string' || id.trim() === '') {
     notFound();
  }

  const data = await getIdiSimulationData(id); 

  if (!data) notFound(); 

  // Remove solid background, add relative positioning
  return (
    <div className="relative min-h-screen"> 
      {/* Add fixed Vortex */}
      <Vortex
        backgroundColor="black"
        rangeY={800}
        particleCount={500}
        baseHue={200} // Matching setup page hue
        containerClassName="fixed inset-0 w-full h-full z-0"
        className=""
      />

       {/* Add scrollable content container */}
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-start overflow-y-auto py-10">
        {/* Container for max-width and centering */}
        <div className="container mx-auto w-full max-w-4xl px-4"> 
          {data.status === 'running' && (
            <div className="flex flex-col items-center justify-center space-y-4">
              <h1 className="text-3xl font-bold text-white">IDI Simulation In Progress</h1>
              <p className="text-white/70">Topic: {data.topic}</p>
              <Alert className="max-w-md bg-neutral-600/40 border-neutral-500/50 text-neutral-100"> 
                <ReloadIcon className="h-5 w-5 animate-spin" />
                <AlertTitle className="text-neutral-100">Running Simulation...</AlertTitle>
                <AlertDescription className="text-neutral-200"> 
                  The interview simulation is currently running. Results will appear here once completed.
                  You can monitor the live discussion below.
                </AlertDescription>
              </Alert>
              <Card className="w-full max-w-3xl mt-6 backdrop-blur-sm bg-card/80 dark:bg-card/80"> 
                <CardContent className="p-4">
                  <h2 className="text-xl font-semibold mb-2">Live Interview Feed</h2>
                   {/* Point StreamComponent to the IDI stream endpoint */}
                  <StreamComponent endpoint={`/api/simulations/idi/${id}/stream`} /> 
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
                  The IDI simulation failed to complete. Check server logs for details.
                  {data.error && <p className="mt-2"><strong>Details:</strong> {data.error}</p>}
                  Look for errors in the file: public/simulations/idi/{id}/live.log {/* Adjusted path */}
                </AlertDescription>
              </Alert>
            </div>
          )}

          {data.status === 'completed' && (
            <div className="space-y-6 mb-10"> {/* Added mb-10 for bottom spacing */}
              <h1 className="text-3xl font-bold text-center text-white mb-6">{data.topic}</h1>
              {(() => {
                // Use IDI helper functions
                const reportPromise = getIdiSimulationReport(id); 
                const vizPromise = getIdiVisualizationFiles(id);
                return (
                  <React.Suspense fallback={<p className="text-white text-center">Loading results...</p>}> 
                    {/* Pass IDI-specific data type */}
                    <IdiCompletedResultsContent 
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

// --- Helper Component for Completed State (Adapted for IDI) ---
async function IdiCompletedResultsContent({ data, reportPromise, vizPromise }: {
  data: IdiSimulationData; // Use IDI data type
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
               <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
               {visualizationFiles.map((file, index) => (
                   <div key={index} className="border rounded-lg p-2 flex justify-center items-center bg-white"> 
                   <Image
                       src={file}
                       alt={`Visualization ${index + 1}`}
                       width={500}
                       height={400}
                       style={{ objectFit: 'contain', maxWidth: '100%', height: 'auto' }}
                   />
                   </div>
               ))}
               </div>
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
             {/* Display IDI specific parameter */}
            <p><strong>Approx. Questions:</strong> {data.numQuestions}</p> 
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