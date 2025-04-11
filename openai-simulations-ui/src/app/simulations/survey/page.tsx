import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SurveySimulationForm } from "@/components/survey/simulation-form";
import Link from "next/link";
import { Vortex } from "@/components/ui/vortex";

export default function SurveySimulationPage() {
  return (
    <div className="relative min-h-screen">
      <Vortex
        backgroundColor="black"
        rangeY={800}
        particleCount={500}
        baseHue={240}
        containerClassName="fixed inset-0 w-full h-full z-0"
        className=""
      />

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-start overflow-y-auto py-10">
        <div className="w-full max-w-4xl container mx-auto px-4">
          <div className="flex items-center justify-between mb-6">
            <Link href="/">
              <Button variant="outline">Back to Home</Button>
            </Link>
          </div>

          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold tracking-tight text-white dark:text-white">
              Survey Simulation
            </h1>
            <p className="text-white/70 dark:text-white/70">
              Configure and launch a survey simulation with AI-generated respondents.
            </p>
          </div>

          <Card className="max-w-3xl mx-auto backdrop-blur-sm bg-card/80 dark:bg-card/80 mb-10">
            <CardHeader>
              <CardTitle>
                Simulation Parameters
              </CardTitle>
              <CardDescription>
                Define the topic, objectives, audience, and scale for your survey simulation.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SurveySimulationForm />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 