'use client';

import React, { useEffect, useState, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { useRouter } from 'next/navigation';
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageCircle, User, Flag, ChevronRight } from "lucide-react";

interface StreamComponentProps {
  endpoint: string;
}

// Helper function to determine if a line is from the moderator
const isModeratorLine = (line: string) => line.startsWith('Moderator:');

// Helper function to determine if a line is from a participant
const isParticipantLine = (line: string) => {
  return line.includes('(Participant_') && !line.startsWith('---');
};

// Helper function to determine if a line is a section header (e.g., "--- Round 1 ---")
const isSectionHeader = (line: string) => line.startsWith('---') && line.endsWith('---');

// Helper to extract participant name
const extractParticipantName = (line: string) => {
  const nameMatch = line.match(/^(.*?)\s+\(Participant_\d+\):/);
  return nameMatch ? nameMatch[1] : 'Participant';
};

const StreamComponent: React.FC<StreamComponentProps> = ({ endpoint }) => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<string[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);
  const router = useRouter();
  const transcriptEndRef = useRef<null | HTMLDivElement>(null);

  useEffect(() => {
    if (!endpoint) {
      setError('No stream URL provided');
      setLoading(false);
      return;
    }

    const fetchData = () => {
      try {
        // Close any existing event source
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }

        // Create new EventSource
        const eventSource = new EventSource(endpoint);
        eventSourceRef.current = eventSource;

        eventSource.onopen = () => {
          console.log('SSE connection established');
          setLoading(false);
        };

        eventSource.onmessage = (event) => {
          setError(null);
          try {
            const data = JSON.parse(event.data);
            if (data && typeof data.line === 'string') {
              setTranscript((prev) => [...prev, data.line]);
            } else {
              console.warn("Received unexpected message format:", data);
            }
          } catch (parseError) {
            console.error('Error parsing SSE message:', parseError);
            setError('Error processing stream data.');
          }
        };

        eventSource.addEventListener('status', (event) => {
          try {
            const statusData = JSON.parse(event.data);
            console.log("Received status event:", statusData);
            if (statusData.status === 'completed' || statusData.status === 'failed') {
              console.log(`Simulation status is ${statusData.status}. Refreshing page...`);
              eventSource.close();
              setTimeout(() => router.refresh(), 500);
            }
          } catch (parseError) {
            console.error('Error parsing status event:', parseError);
          }
        });

        eventSource.onerror = (err) => {
          console.error('EventSource error:', err);
          setError('Error connecting to stream. Please try again later.');
          setLoading(false);
          eventSource.close();
          eventSourceRef.current = null;
        };

        return () => {
          eventSource.close();
          eventSourceRef.current = null;
        };
      } catch (err) {
        console.error('Error setting up EventSource:', err);
        setError('Failed to connect to stream');
        setLoading(false);
      }
    };

    const cleanup = fetchData();
    
    return cleanup;
  }, [endpoint, router]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  if (loading && transcript.length === 0) {
    return (
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-center p-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            <p className="ml-2">Loading discussion...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-4">
          <div className="text-red-500">
            <p className="font-bold">Error:</p>
            <p>{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4">
        <ScrollArea className="h-[500px] w-full rounded-md border p-4 text-sm bg-gray-100/90 dark:bg-black/20 backdrop-blur-sm">
          {transcript.length === 0 && !loading && (
            <p className="text-muted-foreground">Waiting for the focus group to begin...</p>
          )}
          {transcript.map((line, index) => {
            // Format different types of lines
            if (isSectionHeader(line)) {
              return (
                <div key={index} className="flex justify-center my-4">
                  <div className="bg-blue-900/50 text-blue-100 dark:text-blue-100 px-4 py-2 rounded-full font-semibold text-sm">
                    <Flag className="inline-block mr-2 h-4 w-4" />
                    {line.replace(/---/g, '').trim()}
                  </div>
                </div>
              );
            } else if (isModeratorLine(line)) {
              return (
                <div key={index} className="flex items-start mb-4">
                  <div className="flex-shrink-0 bg-purple-700 text-white p-2 rounded-full mr-3">
                    <MessageCircle className="h-5 w-5" />
                  </div>
                  <div className="flex-1">
                    <p className="font-bold text-purple-800 dark:text-purple-300 mb-1">Moderator</p>
                    <div className="bg-purple-100/70 dark:bg-purple-950/40 p-3 rounded-lg text-purple-950 dark:text-purple-100">
                      {line.replace('Moderator:', '').trim()}
                    </div>
                  </div>
                </div>
              );
            } else if (isParticipantLine(line)) {
              const participantName = extractParticipantName(line);
              const participantId = line.match(/\(Participant_(\d+)\)/)?.[1];
              // Generate a consistent color based on participant ID
              const hue = participantId ? (parseInt(participantId) * 60) % 360 : 180;
              
              return (
                <div key={index} className="flex items-start mb-4 pl-6">
                  <div className="flex-shrink-0 mr-3" style={{ 
                    backgroundColor: `hsl(${hue}, 70%, 40%)`,
                    borderRadius: '50%',
                    padding: '0.5rem',
                    color: 'white'
                  }}>
                    <User className="h-5 w-5" />
                  </div>
                  <div className="flex-1">
                    <p className="font-bold mb-1" style={{ color: `hsl(${hue}, 70%, 40%)` }}>
                      {participantName}
                    </p>
                    <div className="bg-white/80 dark:bg-gray-800/50 p-3 rounded-lg text-gray-900 dark:text-gray-100">
                      {line.substring(line.indexOf(':') + 1).trim()}
                    </div>
                  </div>
                </div>
              );
            } else if (line.includes('thinking')) {
              return (
                <div key={index} className="flex justify-center my-2">
                  <div className="text-gray-600 dark:text-gray-400 text-xs italic flex items-center">
                    <div className="animate-pulse mr-2">•••</div>
                    {line}
                  </div>
                </div>
              );
            } else {
              return (
                <div key={index} className="text-gray-700 dark:text-gray-300 my-2 text-sm">
                  <ChevronRight className="inline-block h-4 w-4 mr-1 text-gray-500" />
                  {line}
                </div>
              );
            }
          })}
          <div ref={transcriptEndRef} />
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default StreamComponent; 