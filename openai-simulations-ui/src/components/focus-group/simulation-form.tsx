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

const formSchema = z.object({
  topic: z.string().min(10, {
    message: "Topic must be at least 10 characters.",
  }).default("Consumer preferences for sustainable footwear"),
  targetAudience: z.string().min(10, {
    message: "Target audience description must be at least 10 characters.",
  }).default("Urban professionals aged 25-40 interested in sustainability"),
  numParticipants: z.coerce.number().int().min(2, {
    message: "Must have at least 2 participants.",
  }).default(4),
  numRounds: z.coerce.number().int().min(1, {
    message: "Must have at least 1 round.",
  }).default(3),
});

export function FocusGroupSimulationForm() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      topic: "Consumer preferences for sustainable footwear",
      targetAudience: "Urban professionals aged 25-40 interested in sustainability",
      numParticipants: 4,
      numRounds: 3,
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    setIsLoading(true);
    // Remove simulation message for now, backend API will handle it
    // toast.info("Starting focus group simulation...", { ... });
    
    console.log("Submitting Form Values:", values);

    try {
      // --- Make the actual API call ---
      const response = await fetch('/api/simulations/focus-group', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
      });

      if (!response.ok) {
        // Handle API errors
        const errorData = await response.json().catch(() => ({ error: "Unknown API error" }));
        console.error("API Error Response:", errorData);
        throw new Error(errorData.error || `Failed to start simulation (Status: ${response.status})`);
      }

      const result = await response.json();
      const simulationId = result.simulationId;

      if (!simulationId) {
         throw new Error("API did not return a simulation ID.");
      }
      
      console.log("Simulation Started. ID:", simulationId);
      
      toast.success("Simulation started successfully!", {
          description: `Redirecting to results page... (ID: ${simulationId})`,
      });

      // --- Correct the redirection path ---
      router.push(`/simulations/focus-group/results/${simulationId}`);

    } catch (error) {
      console.error("Error starting simulation:", error);
      toast.error("Failed to Start Simulation", {
        description: error instanceof Error ? error.message : "An unexpected error occurred.",
      });
      setIsLoading(false); // Ensure loading state is reset on error
    }
    // Removed: setIsLoading(false); - let it stay loading until redirect or error
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <FormField
          control={form.control}
          name="topic"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Topic</FormLabel>
              <FormControl>
                <Textarea
                  placeholder="Enter the focus group discussion topic"
                  className="resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                The main topic of discussion for the focus group.
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
              <FormLabel>Target Audience</FormLabel>
              <FormControl>
                <Textarea
                  placeholder="Describe the target audience"
                  className="resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                Describe the demographic, psychographic and behavioral characteristics of your target audience.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FormField
            control={form.control}
            name="numParticipants"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Number of Participants</FormLabel>
                <FormControl>
                  <Input
                    type="number"
                    min={2}
                    {...field}
                  />
                </FormControl>
                <FormDescription>
                  The number of virtual participants (minimum 2).
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <FormField
            control={form.control}
            name="numRounds"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Number of Rounds</FormLabel>
                <FormControl>
                  <Input
                    type="number"
                    min={1}
                    {...field}
                  />
                </FormControl>
                <FormDescription>
                  The number of discussion rounds (minimum 1).
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        
        <Button type="submit" className="w-full" disabled={isLoading}>
          {isLoading ? "Starting Simulation..." : "Start Simulation"}
        </Button>
      </form>
    </Form>
  );
}
