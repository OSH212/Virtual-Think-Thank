import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { CardSpotlight } from "@/components/ui/card-spotlight";
import { Vortex } from "@/components/ui/vortex";

const simulations = [
  {
    href: "/simulations/focus-group",
    title: "Focus Group Simulation",
    description: "Simulate a focus group discussion with AI participants.",
    imgSrc: "/focus-group.png",
    disabled: false,
  },
  {
    href: "/simulations/idi",
    title: "In-Depth Interview (IDI)",
    description: "Conduct a one-on-one interview with an AI respondent.",
    imgSrc: "/interview.png",
    disabled: false,
  },
  {
    href: "/simulations/survey",
    title: "Survey Simulation",
    description: "Design and deploy surveys answered by AI respondents.",
    imgSrc: "/survey.png",
    disabled: false,
  },
];

export default function Home() {
  return (
    <Vortex
      backgroundColor="black"
      rangeY={800}
      particleCount={500}
      baseHue={120}
      className="flex min-h-screen flex-col items-center justify-center p-24"
    >
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex mb-16">
        <h1 className="text-4xl font-bold text-center text-neutral-200 dark:text-neutral-200 w-full">
          AI Agent Simulations
        </h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 w-full max-w-6xl">
        {simulations.map((sim) => (
          <Link href={sim.disabled ? "#" : sim.href} key={sim.href} passHref legacyBehavior>
            <a className={`block h-full ${sim.disabled ? 'opacity-50 cursor-not-allowed relative' : ''}`}>
              <CardSpotlight className="h-full flex flex-col items-center justify-center">
                <div className="p-6 text-center">
                  <img 
                    src={sim.imgSrc} 
                    alt={`${sim.title} illustration`} 
                    className="w-32 h-32 object-contain mx-auto mb-4 rounded-lg"
                  />
                  <h2 className="text-xl font-semibold mb-2 text-neutral-200 dark:text-neutral-200">{sim.title}</h2>
                  <p className="text-sm text-neutral-400 dark:text-neutral-400">{sim.description}</p>
                </div>
              </CardSpotlight>
              {sim.disabled && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
                  <span className="bg-yellow-500 text-black text-xs font-bold px-2 py-1 rounded">Coming Soon</span>
                </div>
              )}
            </a>
          </Link>
        ))}
      </div>

      <footer className="mt-16 text-center text-neutral-500 dark:text-neutral-500 text-sm">
        <p>Powered by Advanced AI Models</p>
        <p>&copy; {new Date().getFullYear()} Simulation Corp. All rights reserved.</p>
      </footer>
    </Vortex>
  );
}
