"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

// Schema based on idi_simulation.py arguments
const formSchema = z.object({
  topic: z.string().min(10, {
    message: "Topic must be at least 10 characters.",
  }).default("Experiences with remote work technologies"),
  targetAudience: z.string().min(10, {
    message: "Target audience description must be at least 10 characters.",
  }).default("Software developers aged 28-45 working fully remotely for tech companies"),
  numQuestions: z.coerce.number().int().min(3, {
    message: "Must have at least 3 main questions.",
  }).max(15, { 
    message: "Cannot exceed 15 main questions.",
  }).default(8),
});

export function IdiSimulationForm() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      topic: "Experiences with remote work technologies",
      targetAudience: "Software developers aged 28-45 working fully remotely for tech companies",
      numQuestions: 8,
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    setIsLoading(true);
    console.log("Submitting IDI Form Values:", values);

    try {
      // --- Make the API call to the new IDI endpoint ---
      const response = await fetch('/api/simulations/idi', { // Updated endpoint
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Unknown API error" }));
        console.error("API Error Response:", errorData);
        throw new Error(errorData.error || `Failed to start simulation (Status: ${response.status})`);
      }

      const result = await response.json();
      const simulationId = result.simulationId;

      if (!simulationId) {
         throw new Error("API did not return a simulation ID.");
      }
      
      console.log("IDI Simulation Started. ID:", simulationId);
      
      toast.success("IDI Simulation started successfully!", {
          description: `Redirecting to results page... (ID: ${simulationId})`,
      });

      // --- Correct the redirection path for IDI ---
      router.push(`/simulations/idi/results/${simulationId}`); // Updated path

    } catch (error) {
      console.error("Error starting IDI simulation:", error);
      toast.error("Failed to Start IDI Simulation", {
        description: error instanceof Error ? error.message : "An unexpected error occurred.",
      });
      setIsLoading(false); 
    } 
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <FormField
          control={form.control}
          name="topic"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Interview Topic</FormLabel>
              <FormControl>
                <Textarea
                  placeholder="Enter the main topic for the in-depth interview"
                  className="resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                 The core subject the interview will explore.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <FormField
          control={form.control}
          name="targetAudience"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Target Audience / Respondent Profile</FormLabel>
              <FormControl>
                <Textarea
                  placeholder="Describe the desired respondent profile in detail"
                  className="resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                Provide detailed demographic, psychographic, and behavioral characteristics for the AI respondent.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <FormField
          control={form.control}
          name="numQuestions"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Approximate Number of Main Questions</FormLabel>
              <FormControl>
                <Input
                  type="number"
                  min={3}
                  max={15}
                  {...field}
                />
              </FormControl>
              <FormDescription>
                The target number of primary questions the interviewer will aim to cover (minimum 3).
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <Button type="submit" className="w-full" disabled={isLoading}>
          {isLoading ? "Starting Simulation..." : "Start IDI Simulation"}
        </Button>
      </form>
    </Form>
  );
} 