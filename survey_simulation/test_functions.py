#!/usr/bin/env python
"""
Simple test script to verify that the survey simulation functions work properly
using mock implementations to avoid API calls.
"""

import asyncio
import os
from dotenv import load_dotenv

# Use mock implementation instead of real agents
from mock_agents import (
    Agent, 
    SurveyDesignerAgent,
    RespondentGeneratorAgent,
    SurveyResponseAgent,
    SurveyAnalystAgent,
    Runner
)

from survey_simulation import (
    design_survey,
    process_survey_data,
    collect_survey_responses,
    analyze_survey_data,
    generate_visualizations
)

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

async def test_functions():
    """
    Test the core survey simulation functions.
    """
    print("\n=== Testing Survey Simulation Functions ===")
    
    # Test parameters
    topic = "Electric Vehicles"
    research_objectives = "Understand barriers to EV adoption"
    target_audience = "Car owners aged 25-50"
    num_respondents = 5  # Small number for testing
    
    # 1. Test design_survey function
    print("\n1. Testing survey design...")
    try:
        survey = design_survey(topic, research_objectives, target_audience)
        print(f"Survey designed with {len(survey.get('sections', []))} sections")
        print(f"Survey title: {survey.get('survey_title', 'No title')}")
        
        # Count questions
        question_count = sum(len(section.get("questions", [])) for section in survey.get("sections", []))
        print(f"Total questions: {question_count}")
    except Exception as e:
        print(f"Error in survey design: {e}")
        return
    
    # 2. Test respondent generation
    print("\n2. Testing respondent generation...")
    try:
        respondents = await survey_simulation.generate_respondents(target_audience, num_respondents)
        print(f"Generated {len(respondents)} respondent profiles")
        if respondents:
            print(f"Sample respondent: {respondents[0].get('respondent_id', 'Unknown')}, "
                  f"{respondents[0].get('age', 'Unknown')}, {respondents[0].get('occupation', 'Unknown')}")
    except Exception as e:
        print(f"Error in respondent generation: {e}")
        return
    
    # 3. Test survey responses collection
    print("\n3. Testing response collection...")
    try:
        responses = collect_survey_responses(survey, respondents[:2])  # Just test with 2 respondents
        print(f"Collected {len(responses)} responses")
    except Exception as e:
        print(f"Error in response collection: {e}")
        return
    
    # 4. Test data processing
    print("\n4. Testing data processing...")
    try:
        processed_data = process_survey_data(survey, responses)
        print(f"Processed data for {processed_data.get('total_respondents', 0)} respondents")
    except Exception as e:
        print(f"Error in data processing: {e}")
        return
    
    # 5. Test analysis
    print("\n5. Testing analysis...")
    try:
        analysis = analyze_survey_data(topic, research_objectives, target_audience, survey, processed_data)
        print(f"Analysis generated with {len(analysis.get('analysis_text', ''))} characters")
    except Exception as e:
        print(f"Error in analysis: {e}")
        return
    
    # 6. Test visualization generation (if analysis succeeded)
    print("\n6. Testing visualization generation...")
    try:
        test_output_dir = "test_output"
        os.makedirs(test_output_dir, exist_ok=True)
        viz_files = generate_visualizations(survey, processed_data, analysis, test_output_dir)
        print(f"Generated {len(viz_files)} visualization files")
    except Exception as e:
        print(f"Error in visualization generation: {e}")
        return
    
    print("\n=== All tests completed! ===")

if __name__ == "__main__":
    asyncio.run(test_functions()) 