#!/usr/bin/env python
"""
Test script for the full survey simulation pipeline.
Uses mock implementations to avoid API calls.
"""

import asyncio
import os
import json
import datetime
import tempfile
from dotenv import load_dotenv
from survey_simulation import run_survey_simulation


import sys
import mock_agents
sys.modules['agents'] = mock_agents

# Use mock implementations instead of real agents
from mock_agents import (
    Agent, 
    SurveyDesignerAgent,
    RespondentGeneratorAgent,
    SurveyResponseAgent,
    SurveyAnalystAgent,
    Runner
)

# Import the main run function
from survey_simulation import run_survey_simulation

# Load environment variables from .env file
load_dotenv()

# Override the imported modules with our mock versions
import survey_simulation
survey_simulation.Agent = Agent
survey_simulation.SurveyDesignerAgent = SurveyDesignerAgent
survey_simulation.RespondentGeneratorAgent = RespondentGeneratorAgent
survey_simulation.SurveyResponseAgent = SurveyResponseAgent
survey_simulation.SurveyAnalystAgent = SurveyAnalystAgent
survey_simulation.Runner = Runner

async def test_full_simulation():
    """Test the full survey simulation workflow."""
    print("\n=== Testing Full Survey Simulation ===\n")
    
    # Create temp directory for outputs
    tmp_dir = "test_simulation_output"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Run with minimal parameters
    print("Running survey simulation on 'Electric Vehicles'...")
    results = await run_survey_simulation(
        topic="Electric Vehicles",
        research_objectives="Understand barriers to EV adoption",
        target_audience="Car owners aged 25-50",
        num_respondents=5
    )
    
    # Print some information about the results
    if results:
        print("\nSimulation complete!")
        print(f"Generated {len(results.get('respondents', []))} respondent profiles")
        print(f"Collected {len(results.get('responses', []))} response sets")
        
        # Print locations of output files
        output_dir = results.get('output_directory', '')
        if output_dir and os.path.exists(output_dir):
            print(f"\nOutput saved to: {output_dir}")
            print("Files generated:")
            for root, _, files in os.walk(output_dir):
                for file in files:
                    print(f"  - {os.path.join(root, file)}")
    else:
        print("Simulation failed to produce results.")

async def main():
    """Main test function."""
    start_time = datetime.datetime.now()
    print(f"Starting survey simulation at {start_time}")
    
    await test_full_simulation()
    
    end_time = datetime.datetime.now()
    print(f"\nTests finished at {end_time}")
    print(f"Total time: {end_time - start_time}")

if __name__ == "__main__":
    asyncio.run(main()) 