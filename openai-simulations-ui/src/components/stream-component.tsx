'use client';

import React, { useEffect, useState, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { useRouter } from 'next/navigation';
import { ScrollArea } from "@/components/ui/scroll-area";

interface StreamComponentProps {
  endpoint: string;
}

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
        <ScrollArea className="h-[400px] w-full rounded-md border p-4 text-sm bg-muted/40">
          {transcript.length === 0 && !loading && (
            <p className="text-muted-foreground">No content available yet...</p>
          )}
          {transcript.map((line, index) => (
            <p key={index} className="mb-1">{line}</p>
          ))}
          <div ref={transcriptEndRef} />
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default StreamComponent; 