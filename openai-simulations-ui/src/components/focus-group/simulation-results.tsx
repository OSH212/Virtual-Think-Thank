"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import MarkdownRenderer from "@/components/markdown-renderer";
import { useParams } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { ExternalLink, MaximizeIcon } from "lucide-react";
import Link from 'next/link';

interface SimulationResultsProps {
  markdown: string;
}

/**
 * SimulationResults Component
 * 
 * Renders a focus group simulation report in a structured format.
 * The report is expected to be in Markdown format with sections for:
 * - Simulation Parameters
 * - Participant Profiles
 * - Analysis Results
 */
export function SimulationResults({ markdown }: SimulationResultsProps) {
  const [content, setContent] = useState(markdown);
  const [knowledgeGraphHtmlExists, setKnowledgeGraphHtmlExists] = useState(false);
  const [knowledgeGraphPngExists, setKnowledgeGraphPngExists] = useState(false);
  const [visualizationFiles, setVisualizationFiles] = useState<string[]>([]);
  const params = useParams();
  const simulationId = params.id as string;

  useEffect(() => {
    setContent(markdown);
    
    // Check if files exist
    if (simulationId) {
      // Check for HTML knowledge graph
      fetch(`/simulations/focus-group/${simulationId}/visualizations/knowledge_graph.html`, { method: 'HEAD' })
        .then(res => {
          setKnowledgeGraphHtmlExists(res.ok);
        })
        .catch(() => {
          setKnowledgeGraphHtmlExists(false);
        });
      
      // Check for PNG knowledge graph
      fetch(`/simulations/focus-group/${simulationId}/visualizations/knowledge_graph.png`, { method: 'HEAD' })
        .then(res => {
          setKnowledgeGraphPngExists(res.ok);
        })
        .catch(() => {
          setKnowledgeGraphPngExists(false);
        });
      
      // Extract visualization file references from markdown
      const visualizationLinks = extractVisualizationLinks(markdown);
      setVisualizationFiles(visualizationLinks);
    }
  }, [markdown, simulationId]);

  if (!content) {
    return (
      <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80">
        <CardContent className="p-4">
          <p className="text-muted-foreground">Report not available.</p>
        </CardContent>
      </Card>
    );
  }

  // Parse report sections
  const sections = parseReportSections(content);

  return (
    <div className="space-y-6">
      <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80">
        <CardContent className="p-4">
          <h2 className="text-xl font-semibold mb-4">Simulation Parameters</h2>
          <div className="prose max-w-none dark:prose-invert">
            {sections.parameters ? (
              <MarkdownRenderer content={sections.parameters} />
            ) : (
              <p className="text-muted-foreground">Parameters information not available.</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80">
        <CardContent className="p-4">
          <h2 className="text-xl font-semibold mb-4">Participant Profiles</h2>
          <div className="prose max-w-none dark:prose-invert">
            {sections.profiles ? (
              <MarkdownRenderer content={sections.profiles} />
            ) : (
              <p className="text-muted-foreground">Participant profiles not available.</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80">
        <CardContent className="p-4">
          <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
          <div className="prose max-w-none dark:prose-invert">
            {sections.analysis ? (
              <MarkdownRenderer content={sections.analysis} />
            ) : (
              <p className="text-muted-foreground">Analysis results not available.</p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Visualization Section */}
      <Card className="backdrop-blur-sm bg-card/80 dark:bg-card/80">
        <CardContent className="p-4">
          <h2 className="text-xl font-semibold mb-4">Visualizations</h2>
          
          {/* Static Knowledge Graph */}
          {knowledgeGraphPngExists && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Network Graph - Static View</h3>
              <div className="border rounded-md overflow-hidden p-2 bg-white">
                <img 
                  src={`/simulations/focus-group/${simulationId}/visualizations/knowledge_graph.png`}
                  alt="Knowledge Graph Static View"
                  className="w-full h-auto max-h-[500px] object-contain"
                />
              </div>
              {knowledgeGraphHtmlExists && (
                <div className="mt-2 text-right">
                  <Button variant="outline" size="sm" asChild>
                    <Link href={`/simulations/focus-group/${simulationId}/visualizations/knowledge_graph.html`} target="_blank">
                      <ExternalLink className="h-4 w-4 mr-1" />
                      Open Interactive Version
                    </Link>
                  </Button>
                </div>
              )}
            </div>
          )}
          
          {/* Interactive Knowledge Graph */}
          {knowledgeGraphHtmlExists && !knowledgeGraphPngExists && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Network Graph - Interactive</h3>
              <p className="mb-3 text-sm text-muted-foreground">
                This visualization shows relationships between participants. You can drag nodes, zoom with your mouse wheel, 
                and hover over connections to see details.
              </p>
              <div className="w-full h-[500px] border rounded-md overflow-hidden bg-white">
                <iframe 
                  src={`/simulations/focus-group/${simulationId}/visualizations/knowledge_graph.html`}
                  className="w-full h-full border-0"
                  title="Participant Interaction Network"
                  sandbox="allow-scripts allow-same-origin"
                />
              </div>
              <div className="mt-2 text-right">
                <Button variant="outline" size="sm" asChild>
                  <Link href={`/simulations/focus-group/${simulationId}/visualizations/knowledge_graph.html`} target="_blank">
                    <MaximizeIcon className="h-4 w-4 mr-1" />
                    Open in Full Screen
                  </Link>
                </Button>
              </div>
            </div>
          )}
          
          {/* Other Visualizations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            {visualizationFiles.filter(file => 
              file.endsWith('.png') && 
              !file.includes('knowledge_graph')
            ).map((file, index) => (
              <div key={index} className="border rounded-md p-2 bg-white">
                <img 
                  src={`/simulations/focus-group/${simulationId}/visualizations/${file}`}
                  alt={file.replace('.png', '')}
                  className="w-full h-auto object-contain"
                />
                <p className="text-center text-sm mt-2">{formatFileName(file)}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Parse the report into sections
 * 
 * @param report Full markdown report
 * @returns Object with sections of the report
 */
function parseReportSections(report: string) {
  // Default section headers to look for
  const parametersSectionHeader = '## 1. Simulation Parameters';
  const profilesSectionHeader = '## 2. Participant Profiles';
  const analysisSectionHeader = '## 3. Analysis Results';

  // Find section indices
  const parametersIndex = report.indexOf(parametersSectionHeader);
  const profilesIndex = report.indexOf(profilesSectionHeader);
  const analysisIndex = report.indexOf(analysisSectionHeader);
  
  // Extract sections
  const parameters = parametersIndex >= 0 && profilesIndex >= 0 
    ? report.substring(parametersIndex, profilesIndex).trim()
    : null;
  
  const profiles = profilesIndex >= 0 && analysisIndex >= 0
    ? report.substring(profilesIndex, analysisIndex).trim()
    : null;
  
  const analysis = analysisIndex >= 0
    ? report.substring(analysisIndex).trim()
    : null;

  return {
    parameters,
    profiles,
    analysis
  };
}

/**
 * Extract visualization file links from markdown
 * 
 * @param markdown Report markdown
 * @returns Array of visualization filenames
 */
function extractVisualizationLinks(markdown: string): string[] {
  const files: string[] = [];
  
  // Match markdown image links ![text](url)
  const imgRegex = /!\[.*?\]\((\/simulations\/focus-group\/.*?\/visualizations\/([^)]+))\)/g;
  let match;
  
  while ((match = imgRegex.exec(markdown)) !== null) {
    const filename = match[2];
    if (filename && !files.includes(filename)) {
      files.push(filename);
    }
  }
  
  // Also match regular links [text](url)
  const linkRegex = /\[.*?\]\((\/simulations\/focus-group\/.*?\/visualizations\/([^)]+))\)/g;
  
  while ((match = linkRegex.exec(markdown)) !== null) {
    const filename = match[2];
    if (filename && !files.includes(filename)) {
      files.push(filename);
    }
  }
  
  return files;
}

/**
 * Format file names for display
 * 
 * @param filename Raw filename
 * @returns Formatted display name
 */
function formatFileName(filename: string): string {
  return filename
    .replace(/\.png$/, '')
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
