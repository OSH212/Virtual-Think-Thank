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

// Schema based on survey_simulation.py arguments
const formSchema = z.object({
  topic: z.string().min(5, {
    message: "Topic must be at least 5 characters.",
  }).default("Public opinion on renewable energy sources"),
  researchObjectives: z.string().min(10, {
    message: "Research objectives must be at least 10 characters.",
  }).default("Understand general awareness, perceived benefits/drawbacks, and support for different renewable energy types."),
  targetAudience: z.string().min(10, {
    message: "Target audience description must be at least 10 characters.",
  }).default("General adult population in the United States, diverse in age, location, and political leaning."),
  numRespondents: z.coerce.number().int().min(5, {
    message: "Must have at least 5 respondents.",
  }).max(200, { // Adjusted max based on potential cost/time
    message: "Cannot exceed 200 respondents.",
  }).default(50),
});

export function SurveySimulationForm() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      topic: "Public opinion on renewable energy sources",
      researchObjectives: "Understand general awareness, perceived benefits/drawbacks, and support for different renewable energy types.",
      targetAudience: "General adult population in the United States, diverse in age, location, and political leaning.",
      numRespondents: 50,
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    setIsLoading(true);
    console.log("Submitting Survey Form Values:", values);

    try {
      // Make the API call to the new survey endpoint
      const response = await fetch('/api/simulations/survey', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Unknown API error" }));
        console.error("API Error Response:", errorData);
        throw new Error(errorData.error || `Failed to start survey simulation (Status: ${response.status})`);
      }

      const result = await response.json();
      const simulationId = result.simulationId;

      if (!simulationId) {
         throw new Error("API did not return a simulation ID.");
      }

      console.log("Survey Simulation Started. ID:", simulationId);

      toast.success("Survey Simulation started successfully!", {
          description: `Redirecting to results page... (ID: ${simulationId})`,
      });

      // Redirect to the survey results page
      router.push(`/simulations/survey/results/${simulationId}`);

    } catch (error) {
      console.error("Error starting survey simulation:", error);
      toast.error("Failed to Start Survey Simulation", {
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
              <FormLabel>Survey Topic</FormLabel>
              <FormControl>
                <Input
                  placeholder="Enter the main topic for the survey"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                 The core subject the survey will cover.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="researchObjectives"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Research Objectives</FormLabel>
              <FormControl>
                <Textarea
                  placeholder="Describe the specific research goals and questions the survey aims to answer."
                  className="resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                What do you want to learn from this survey? Be specific.
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
                  placeholder="Describe the desired respondents in detail (demographics, psychographics, etc.)"
                  className="resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                Who should be responding to this survey? This guides AI respondent generation.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="numRespondents"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Number of Respondents</FormLabel>
              <FormControl>
                <Input
                  type="number"
                  min={5}
                  max={200} // Consistent with schema
                  {...field}
                />
              </FormControl>
              <FormDescription>
                The number of AI respondents to simulate (minimum 5).
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <Button type="submit" className="w-full" disabled={isLoading}>
          {isLoading ? "Starting Simulation..." : "Start Survey Simulation"}
        </Button>
      </form>
    </Form>
  );
} 