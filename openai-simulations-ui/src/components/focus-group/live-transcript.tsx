"use client";

import React from 'react';
import { useEffect, useState, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { RefreshCw } from 'lucide-react';

interface Transcript {
  speaker: string;
  text: string;
}

interface LiveTranscriptProps {
  simulationId: string;
  initialTranscript: Transcript[] | null;
  status: string;
}

export function LiveTranscript({ simulationId, initialTranscript, status }: LiveTranscriptProps) {
  const [transcript, setTranscript] = useState<Transcript[]>(initialTranscript || []);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(status === 'running');
  const eventSourceRef = useRef<EventSource | null>(null);

  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    // Scroll to the bottom when transcript updates
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current;
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  }, [transcript]);

  useEffect(() => {
    // Initialize with any transcript data we have
    if (initialTranscript?.length) {
      setTranscript(initialTranscript);
    }

    // Only set up streaming for running simulations
    if (status === 'running') {
      startStreaming();
    }

    return () => {
      // Clean up the event source when component unmounts
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [simulationId, status]);

  const startStreaming = () => {
    setIsStreaming(true);
    setError(null);
    
    // Close any existing EventSource before creating a new one
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    try {
      // Create a new EventSource connection for SSE
      const eventSource = new EventSource(`/api/simulations/focus-group/${simulationId}/stream`);
      
      eventSourceRef.current = eventSource;
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.transcript) {
            setTranscript(data.transcript);
          }
        } catch (err) {
          console.error('Error parsing transcript data:', err);
        }
      };
      
      eventSource.onerror = (err) => {
        console.error('EventSource error:', err);
        setError('Connection to transcript stream failed');
        setIsStreaming(false);
        eventSource.close();
      };
      
    } catch (err) {
      console.error('Error setting up streaming:', err);
      setError('Failed to set up streaming connection');
      setIsStreaming(false);
      // Fallback to polling when streaming fails
      fetchTranscript();
    }
  };

  const fetchTranscript = async () => {
    if (loading) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/simulations/focus-group/${simulationId}/transcript`);
      
      if (!response.ok) {
        if (response.status === 404) {
          setTranscript([]);
          return;
        }
        throw new Error(`Failed to fetch transcript: ${response.statusText}`);
      }
      
      const data = await response.json();
      setTranscript(data.transcript || []);
    } catch (err) {
      console.error('Error fetching transcript:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch transcript');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    if (isStreaming) {
      // Restart streaming
      startStreaming();
    } else {
      // Manual refresh with polling
      fetchTranscript();
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center mb-4">
        <div>
          {isStreaming && (
            <div className="flex items-center">
              <div className="h-3 w-3 bg-green-500 rounded-full mr-2 animate-pulse"></div>
              <span className="text-sm text-green-600">Live updating</span>
            </div>
          )}
          {error && <div className="text-sm text-red-500">{error}</div>}
        </div>
        <Button 
          variant="outline" 
          size="sm" 
          onClick={handleRefresh} 
          disabled={loading}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>
      
      <div 
        ref={scrollAreaRef}
        className="border rounded-md h-[600px] overflow-auto p-4"
      >
        {transcript.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-gray-500">
                {status === 'running' 
                  ? 'Waiting for the simulation to start producing output...' 
                  : 'No transcript data available'}
              </p>
              {status === 'running' && (
                <div className="flex justify-center mt-4 space-x-1">
                  <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                  <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                  <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '600ms'}}></div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {transcript.map((item, index) => (
              <div key={index} className="space-y-1">
                <p className="font-semibold">{item.speaker}</p>
                <p className="text-gray-700">{item.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 