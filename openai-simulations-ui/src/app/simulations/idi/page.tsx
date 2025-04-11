import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { IdiSimulationForm } from "@/components/idi/simulation-form";
import Link from "next/link";
import { Vortex } from "@/components/ui/vortex";

export default function IdiSimulation() {
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
          <h1 className="text-3xl font-bold tracking-tight text-white">
            In-Depth Interview (IDI)
          </h1>
          <Link href="/">
            <Button variant="outline">Back to Home</Button>
          </Link>
        </div>
        
        <div className="text-center mb-8">
          <p className="text-white/70">
            Configure and launch a one-on-one interview simulation with an AI respondent.
          </p>
        </div>
        
        <Card className="max-w-3xl mx-auto backdrop-blur-sm"> 
          <CardHeader>
            <CardTitle>
              Simulation Parameters
            </CardTitle>
            <CardDescription>
              Configure the IDI simulation parameters. Provide details for a more accurate respondent persona.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <IdiSimulationForm />
          </CardContent>
        </Card>
      </div>
    </Vortex>
  );
} 