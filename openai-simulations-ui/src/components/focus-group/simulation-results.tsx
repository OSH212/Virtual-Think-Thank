"use client";

import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Card } from '@/components/ui/card';

interface SimulationResultsProps {
  markdown: string;
}

export function SimulationResults({ markdown }: SimulationResultsProps) {
  const [content, setContent] = useState(markdown);

  useEffect(() => {
    setContent(markdown);
  }, [markdown]);

  return (
    <div className="markdown-body prose max-w-none dark:prose-invert">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}
