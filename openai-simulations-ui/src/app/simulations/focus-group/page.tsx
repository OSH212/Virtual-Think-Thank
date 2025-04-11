import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FocusGroupSimulationForm } from "@/components/focus-group/simulation-form";
import Link from "next/link";
import { Vortex } from "@/components/ui/vortex";

export default function FocusGroupSimulation() {
  return (
    <Vortex
      backgroundColor="black"
      rangeY={800}
      particleCount={500}
      baseHue={200}
      className="container mx-auto py-10 min-h-screen flex flex-col items-center"
    >
      <div className="w-full max-w-4xl z-10">
        <div className="flex items-center justify-between mb-6">
          <Link href="/">
            <Button variant="outline">Back to Home</Button>
          </Link>
        </div>
        
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-white dark:text-white">
            Focus Group Simulation
          </h1>
          <p className="text-white/70 dark:text-white/70">
            Create an AI-powered focus group to gather qualitative insights
          </p>
        </div>
        
        <Card className="max-w-3xl mx-auto backdrop-blur-sm">
          <CardHeader>
            <CardTitle>
              Simulation Parameters
            </CardTitle>
            <CardDescription>
              Configure the focus group simulation parameters. More specific parameters will yield more accurate results.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FocusGroupSimulationForm />
          </CardContent>
        </Card>
      </div>
    </Vortex>
  );
}
