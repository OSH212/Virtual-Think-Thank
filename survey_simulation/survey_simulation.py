import os
import asyncio
import re
import json
import argparse
import sys # Import sys for exit
import datetime
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from agents import Agent, Runner
from typing import List, Dict, Any, Tuple, Optional, Union 
import csv
from collections import Counter
import random

# Load environment variables from .env file
load_dotenv()

# --- Agent Definitions ---

class SurveyDesignerAgent(Agent):
    """
    Expert survey designer who creates survey questions based on research objectives.
    """
    def __init__(self, topic: str, research_objectives: str, target_audience: str):
        super().__init__(
            name="Survey Designer",
            instructions="""You are a world-class expert survey methodologist and researcher specializing in rigorous, in-depth survey design.
Your CRITICAL mission is to create a **highly comprehensive, exhaustive, and detailed** survey JSON based *strictly* on the provided topic, research objectives, and target audience. The survey MUST be suitable for gathering nuanced, scientific-quality data.

**EXPECTATION:** Generate a survey with **substantial depth**, typically comprising **at least 15-25 diverse questions** logically organized into **3-5 or more meaningful sections**. Do NOT produce short or superficial surveys.

**MANDATORY JSON STRUCTURE:** Your output MUST be a single, valid JSON object following this PRECISE structure. Absolutely NO deviations, extra fields, or missing fields.

```json
{
  "survey_title": "String: Title reflecting the survey topic accurately",
  "survey_introduction": "String: Clear, concise introduction explaining the survey purpose, estimated time, confidentiality, and aimed at the target audience",
  "sections": [ // Array of Section objects (MINIMUM 3 sections expected)
    {
      "section_title": "String: Clear title for this group of related questions",
      "section_description": "String: Optional brief description clarifying the focus of this section",
      "questions": [ // Array of Question objects (Ensure sufficient questions per section to cover sub-themes)
        // --- Example Question Types ---
        {
          "question_id": "String: Unique alphanumeric ID (e.g., 'INTRO_01', 'ATTITUDE_03a')",
          "question_text": "String: The precise, unambiguous, and unbiased question text",
          "question_type": "String: MUST be one of 'multiple_choice', 'checkbox', 'likert_scale', 'ranking', 'open_ended'",
          // --- Options (Required for non-open_ended types) ---
          "options": [
             // Array of strings for 'multiple_choice', 'checkbox', 'ranking'. Mutually exclusive & exhaustive where applicable.
             // NOT used for 'likert_scale' or 'open_ended'.
          ],
          // --- Likert Scale Specifics (Required for 'likert_scale') ---
          "scale_type": "String: Context for likert scale (e.g., 'Agreement', 'Satisfaction', 'Frequency', 'Importance', 'Likelihood')",
          "scale_points": "Number: Typically 5 or 7 for likert scale",
          "scale_labels": [
            // Array of strings for likert scale points (e.g., ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]).
            // MUST match 'scale_points' count. Used instead of 'options' for likert.
          ],
          // --- End Likert Scale Specifics ---
          "required": "Boolean: true or false" // Indicate if a response is mandatory
        },
        // ... more questions within this section ...
        {
          "question_id": "DEMO_01",
          "question_text": "What is your age range?",
          "question_type": "multiple_choice",
          "options": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
          "required": true
        }
      ]
    },
    // --- Example of another section ---
    {
       "section_title": "Attitudes Towards Topic X",
       "section_description": "Please indicate your views on the following aspects.",
       "questions": [
          // ... questions specifically about attitudes ...
       ]
    }
    // ... MINIMUM 1-3 MORE sections with relevant questions ...
  ]
}
```

**CRITICAL DESIGN PRINCIPLES TO FOLLOW:**
1.  **DEPTH & COMPREHENSIVENESS (MANDATORY):** Thoroughly analyze the research objectives. Break them down into constituent sub-themes or concepts. Ensure **each sub-theme is explored adequately** with multiple, well-designed questions. Aim for the target range of 15-25+ questions across 3-5+ sections.
2.  **Relevance:** ALL questions MUST directly map back to the specific topic and research objectives provided. No irrelevant questions.
3.  **Clarity & Specificity:** Questions must be crystal clear, unambiguous, and easily understood by the specified target audience. Avoid jargon.
4.  **Question Variety:** Employ a strategic mix of the allowed `question_type` values ('multiple_choice', 'checkbox', 'likert_scale', 'ranking', 'open_ended') appropriate for the data needed. Don't overuse one type.
5.  **Logical Flow:** Structure the survey logically. Start with broader engagement questions, group related topics into sections, and typically place sensitive or demographic questions towards the end (unless needed for screening). Sections should follow a coherent order.
6.  **Avoid Bias:** Construct questions with neutral language. Avoid leading, loaded, or double-barreled questions AT ALL COSTS.
7.  **Response Options:** Ensure `options` (for MC/Checkbox/Ranking) are comprehensive, mutually exclusive (where appropriate), balanced, and logically ordered. Ensure `scale_labels` (for Likert) are balanced, clear, and directly correspond to the `scale_points`.
8.  **Appropriate Demographics:** Include only demographic questions that are essential for analyzing the results against the research objectives and target audience segmentation.
9.  **BE EXHAUSTIVE: DEPTH, QUANDITY AND QUALITY ARE KEY. DON'T TRY TO SYNTHESIZE - YOU ARE A WORLD CLASS SURVEY SCIENTIST AND YOU NEVER TRY TO DO THE MINIMUM AMOUNT OF WORK. YOU HAVE TO GET THE MOST OUT OF THE RESPONDENTS.**

**Output Requirement:** Return **ONLY** the single, valid, complete JSON object representing the detailed survey structure. NO additional text, explanations, apologies, or commentary before or after the JSON block.""",
            model="o3-mini" 
        )
        self.topic = topic
        self.research_objectives = research_objectives
        self.target_audience = target_audience

class RespondentGeneratorAgent(Agent):
    """
    Generates diverse, realistic respondent profiles based on the target audience.
    """
    def __init__(self, target_audience: str, num_respondents: int):
        super().__init__(
            name="Respondent Generator",
            instructions=f"""You are an expert market researcher specializing in respondent profiling and persona development.
Your task is to generate {num_respondents} diverse, realistic respondent profiles for a survey targeting: '{target_audience}'.

**MANDATORY JSON STRUCTURE:** Each profile MUST be a JSON object within a JSON array. Each object MUST contain ALL the fields listed below, using the specified types. Use realistic, diverse data appropriate for the target audience.

```json
[ // Start of Array
  {{ // Start of Respondent Object
    "respondent_id": "String: Unique ID (e.g., 'R001')",
    "demographics": {{ // Nested Object
      "age": "Number: e.g., 35",
      "gender": "String: e.g., 'Female', 'Male', 'Non-binary'",
      "ethnicity": "String: e.g., 'Hispanic/Latino', 'White/Caucasian', 'Asian'",
      "location_city": "String: e.g., 'Austin'",
      "location_state": "String: e.g., 'TX'",
      "location_country": "String: e.g., 'USA'",
      "education_level": "String: e.g., 'Bachelor's Degree'",
      "education_field": "String: e.g., 'Marketing', 'Computer Science'",
      "occupation_title": "String: e.g., 'Product Manager', 'Nurse Practitioner'",
      "occupation_industry": "String: e.g., 'Technology', 'Healthcare'",
      "income_annual_usd": "Number: e.g., 85000, 45000", // Numeric annual income
      "marital_status": "String: e.g., 'Married', 'Single', 'Divorced'",
      "household_composition": "String: e.g., 'Living with partner and 2 children', 'Living alone'",
      "homeownership": "String: e.g., 'Homeowner', 'Renter'"
    }},
    "psychographics": {{ // Nested Object
      "personality_traits": ["String", "String", "..."], // Array of 3-5 descriptive traits
      "values": ["String", "String", "..."], // Array of 2-4 core values
      "interests_hobbies": ["String", "String", "..."], // Array of 3-6 interests/hobbies
      "lifestyle_notes": "String: Brief description of lifestyle (e.g., 'Active, travels often', 'Homebody, focuses on family')",
      "media_consumption_primary": ["String", "String", "..."], // Array of 2-4 primary media sources/platforms
      "technology_adoption": "String: e.g., 'Early Adopter', 'Mainstream', 'Late Adopter'"
    }},
    "behaviors": {{ // Nested Object
      "shopping_preferences": ["String", "String", "..."], // Array: e.g., 'Prefers online', 'Value-driven', 'Brand loyal'
      "decision_making_style": "String: e.g., 'Analytical', 'Impulsive', 'Seeks recommendations'",
      "brand_affinities_relevant": ["String", "String", "..."] // Array: Brands relevant to the target audience/topic
    }},
    "response_style": {{ // Nested Object
      "approach": "String: One of 'Thoughtful', 'Rushed', 'Acquiescent', 'Extreme', 'Neutral', 'Distracted'",
      "potential_bias": "String: e.g., 'Slight positive bias towards known brands', 'Tends to rush open-ended questions'",
      "consistency": "String: e.g., 'Generally Consistent', 'Slightly Inconsistent'"
    }}
  }} // End of Respondent Object
  // ... more respondent objects ...
] // End of Array
```

**PROFILE GENERATION GUIDELINES:**
-   **Diversity:** Ensure variation across all demographic, psychographic, and behavioral attributes appropriate for the `{target_audience}`.
-   **Realism & Consistency:** Profiles must be internally coherent. An 'Early Adopter' shouldn't list only outdated media. A 'Value-driven' shopper shouldn't list only luxury brand affinities unless explained (e.g., aspirational).
-   **Detail:** Provide specific job titles, locations, income numbers, etc. Avoid vague terms.
-   **Response Style:** Assign varied response styles to simulate realistic data collection challenges.

**Output Requirement:** Return ONLY the single, valid JSON array containing exactly {num_respondents} respondent profile objects. No explanations or other text.""",
            model="o3-mini" 
        )
        self.target_audience = target_audience
        self.num_respondents = num_respondents


class SurveyResponseAgent(Agent):
    """
    Simulates realistic survey responses based on a detailed respondent profile and response style.
    """
    def __init__(self, respondent_data: Dict[str, Any], survey_data: Dict[str, Any], topic: str):
        
        # --- Create a highly detailed respondent summary for the prompt ---
        respondent_id = respondent_data.get("respondent_id", "Unknown")
        
        demo = respondent_data.get("demographics", {})
        psych = respondent_data.get("psychographics", {})
        behav = respondent_data.get("behaviors", {})
        style = respondent_data.get("response_style", {})

        respondent_summary = f"""
--- YOUR DETAILED PROFILE (ID: {respondent_id}) ---

**Demographics:**
*   Age: {demo.get('age', 'N/A')}
*   Gender: {demo.get('gender', 'N/A')}
*   Ethnicity: {demo.get('ethnicity', 'N/A')}
*   Location: {demo.get('location_city', 'N/A')}, {demo.get('location_state', 'N/A')}, {demo.get('location_country', 'N/A')}
*   Education: {demo.get('education_level', 'N/A')} ({demo.get('education_field', 'N/A')})
*   Occupation: {demo.get('occupation_title', 'N/A')} in {demo.get('occupation_industry', 'N/A')} industry
*   Annual Income (USD): {demo.get('income_annual_usd', 'N/A')}
*   Marital Status: {demo.get('marital_status', 'N/A')}
*   Household: {demo.get('household_composition', 'N/A')}
*   Housing: {demo.get('homeownership', 'N/A')}

**Psychographics:**
*   Personality: {', '.join(psych.get('personality_traits', []))}
*   Values: {', '.join(psych.get('values', []))}
*   Interests/Hobbies: {', '.join(psych.get('interests_hobbies', []))}
*   Lifestyle: {psych.get('lifestyle_notes', 'N/A')}
*   Media Habits: {', '.join(psych.get('media_consumption_primary', []))}
*   Tech Adoption: {psych.get('technology_adoption', 'N/A')}

**Behaviors & Preferences:**
*   Shopping Style: {', '.join(behav.get('shopping_preferences', []))}
*   Decision Making: {behav.get('decision_making_style', 'N/A')}
*   Relevant Brand Affinities: {', '.join(behav.get('brand_affinities_relevant', []))}

**Your Assigned Survey Response Style:**
*   Approach: **{style.get('approach', 'Thoughtful')}** 
*   Potential Bias: {style.get('potential_bias', 'None specified')}
*   Consistency: {style.get('consistency', 'Generally Consistent')}
---
"""
        super().__init__(
            name=f"Respondent_{respondent_id}",
            instructions=f"""You are simulating respondent {respondent_id} completing a survey about '{topic}'.
Your detailed profile and assigned response style are below. You MUST embody this persona *completely* when answering.

{respondent_summary}

**RESPONSE BEHAVIOR MANDATES:**
1.  **Full Embodiment:** Answer EVERY question STRICTLY from the perspective of YOUR profile. Reference your demographics, income, values, interests, behaviors, brand affinities, etc., where relevant.
2.  **Deep Consistency:** Ensure your answers across different questions are logically consistent with your *entire* profile. A high-income, tech-savvy early adopter shouldn't struggle with online concepts. Someone valuing sustainability shouldn't exclusively prefer environmentally unfriendly options without explaining the conflict (e.g., cost).
3.  **Response Style Adherence:** CRITICALLY IMPORTANT - You MUST adopt the assigned 'Approach', 'Potential Bias', and 'Consistency' described in your profile's 'Response Style'.
    *   *If 'Thoughtful':* Provide detailed, nuanced answers, especially for open-ended questions. Explain your reasoning if possible.
    *   *If 'Rushed':* Be brief. Might select first plausible option. Short/skipped open-ended answers.
    *   *If 'Acquiescent':* Tend towards positive/agreeable options. Avoid strong disagreement.
    *   *If 'Extreme':* Prefer polar ends of scales (Strongly Agree/Disagree, Very Satisfied/Dissatisfied).
    *   *If 'Neutral'/'Central Tendency':* Prefer middle scale options. Avoid strong opinions.
    *   *If 'Distracted':* Might give slightly inconsistent answers or miss details. Short open-ended.
    *   *Bias/Consistency:* Actively incorporate your specified bias (e.g., brand bias, price sensitivity) and consistency level.
4.  **Realistic Open-Ended Answers:** Write open-ended responses in a natural voice matching your persona's likely education/background. Length should match your 'Approach' (Thoughtful = longer, Rushed/Distracted = shorter). Reference specific profile details. Skip non-required ones if 'Rushed' or 'Distracted'.
5.  **Selection Logic:** Choose multiple-choice/Likert/ranking options that genuinely reflect your persona's combined profile elements and response style.

**Output Format:**
For each question presented, output a JSON object for your answer. Combine all answer objects into a single JSON array.

*   **For single-select (multiple_choice, likert_scale):**
```json
    {{ "question_id": "Q1", "response": "Selected Option Text" }}
    ```
*   **For multi-select (checkbox):**
```json
    {{ "question_id": "Q2", "response": ["Selected Option 1", "Selected Option 3"] }} 
    ```
    (Include all selected options in the array. If none selected and not required, use an empty array `[]`).
*   **For ranking:**
    ```json
    {{ "question_id": "Q3", "response": ["First Ranked Item", "Second Ranked Item", "..."] }} 
    ```
    (Order the provided options according to your persona's preference).
*   **For open_ended:**
    ```json
    {{ "question_id": "Q4", "response": "Your detailed text answer here, reflecting persona and style." }} 
    ```
    (If skipping an optional question due to style, use an empty string `""` or `null`).

**Task:** Process the entire survey provided in the prompt and return ONE single JSON array containing your answer objects for ALL questions. Ensure valid JSON format.""",
            model="gpt-4o" 
        )
        self.survey_data = survey_data
        self.respondent_data = respondent_data
        self.topic = topic

class SurveyAnalystAgent(Agent):
    """
    Analyzes survey response data and generates comprehensive insights.
    """
    def __init__(self, topic: str, research_objectives: str, target_audience: str):
        super().__init__(
            name="Survey Analyst",
            instructions=f"""You are a senior quantitative research analyst specializing in survey data analysis.
You have been provided with the results of a survey on the topic: '{topic}' 
conducted to address the research objectives: '{research_objectives}' 
with the target audience: '{target_audience}'.

Your task is to analyze the survey data and generate a comprehensive report.

The report should include:
1. **Executive Summary**: A concise overview of the key findings (approx. 200 words).
2. **Research Background**: Brief context on the study purpose, objectives, and target audience.
3. **Methodology**: Overview of the survey approach and respondent demographics.
4. **Key Findings**: Detailed analysis of the survey responses, including:
   - Frequency distributions for each question (provide percentages).
   - Meaningful cross-tabulations (e.g., satisfaction by age group, preference by income). Identify key segments.
   - Statistical significance notes where applicable (e.g., mention if a difference between groups appears notable, even if formal tests aren't run).
   - Thematic analysis summary of open-ended responses, including illustrative quotes.
5. **Data Visualizations Description**: Describe the key charts generated (interpret what they show).
6. **Conclusions & Recommendations**: Strategic insights derived from the findings and actionable recommendations addressing the research objectives.
7. **Limitations**: Methodological limitations of the simulated data or survey design.

Focus on findings that directly address the research objectives. Highlight unexpected or counterintuitive results.
Present a balanced view of the data, avoiding confirmation bias or overinterpretation.
Note any interesting demographic or psychographic differences in responses.

AFTER YOUR ANALYSIS, include a JSON-formatted section with structured data for visualization:

```json
{{
  "key_metrics": [ // Top-level summary numbers
    {{"metric": "Overall Satisfaction (Avg. Score)", "value": 3.8, "scale": 5}},
    {{"metric": "Top Box Purchase Intent (%)", "value": 45.0}} 
    // Add 2-4 key summary metrics
  ],
  "top_findings_summary": [ // Brief text summaries of 3-5 major insights
    "Younger respondents (18-34) show significantly higher interest in Feature X.",
    "Cost remains the primary barrier for adoption among lower-income groups.",
    "Open-ended feedback highlights strong desire for improved customer support."
  ],
  "segment_highlights": [ // Specific insights about key demographic/behavioral groups
    {{"segment": "Females 25-34", "insight": "This group is most likely to recommend the product but desires Feature Y.", "supporting_questions": ["Q5", "Q12"]}},
    {{"segment": "High Income / Early Adopter", "insight": "Willing to pay premium for advanced features but concerned about privacy.", "supporting_questions": ["Q8", "Q15"]}}
    // Add 2-4 segment insights
  ],
  "visualization_summary": [ // Reference the generated charts
    {{"filename": "age_distribution.png", "description": "Shows a peak in the 35-44 age group."}},
    {{"filename": "q3_responses.png", "description": "Reveals moderate overall satisfaction, with 'Neutral' being the most common response."}},
    {{"filename": "key_themes_wordcloud.png", "description": "Highlights 'Price', 'Support', and 'Easy' as frequent terms in open-ended feedback."}}
     // Add summaries for key generated charts
  ]
}}
```
This JSON should be placed at the end of your analysis, clearly separated from the narrative report.""",
            model="gpt-4o" 
        )
        self.topic = topic
        self.research_objectives = research_objectives
        self.target_audience = target_audience

# --- Core Functions ---

# Modified: Added simulation_id, updated save path
async def design_survey(topic: str, research_objectives: str, target_audience: str, simulation_id: str) -> Dict[str, Any]:
    """Uses SurveyDesignerAgent to create a survey and saves it."""
    print(f"STREAM: Designing survey for topic: {topic}", flush=True)
    
    survey_designer = SurveyDesignerAgent(
        topic=topic, research_objectives=research_objectives, target_audience=target_audience
    )
    
    prompt = f"""Design a comprehensive survey JSON object for topic '{topic}', objectives '{research_objectives}', audience '{target_audience}'. Follow the mandatory JSON structure and design principles precisely."""
    
    result = await Runner.run(survey_designer, prompt)
    response_text = result.final_output
    
    # --- Define output paths ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    survey_file_path = os.path.join(base_output_dir, "survey_questions.json")
    # --- End Define output paths ---

    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        survey_json = json_match.group(1) if json_match else response_text
        survey_data = json.loads(survey_json)
        
        if not isinstance(survey_data, dict) or "sections" not in survey_data:
            raise ValueError("Invalid survey structure generated")
            
        # --- Save survey design ---
        with open(survey_file_path, "w") as f:
            json.dump(survey_data, f, indent=2)
        
        total_questions = sum(len(s.get("questions", [])) for s in survey_data.get("sections", []))
        print(f"STREAM: Survey design complete. {len(survey_data.get('sections', []))} sections, {total_questions} questions. Saved to {survey_file_path}", flush=True)
        return survey_data
        
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        print(f"STREAM: Error designing survey: {e}. Creating fallback survey.", flush=True)
        # Fallback survey (keep simple)
        fallback_survey = {
            "survey_title": f"{topic} Survey (Fallback)",
            "survey_introduction": "Basic survey due to design error.",
            "sections": [{"section_title": "Basic Questions", "questions": [
                {"question_id": "Q1", "question_text": "Age?", "question_type": "multiple_choice", "options": ["<25", "25-44", "45+"], "required": True},
                {"question_id": "Q2", "question_text": f"Opinion on {topic}?", "question_type": "open_ended", "required": False}
            ]}]
        }
        # --- Save fallback survey ---
        with open(survey_file_path, "w") as f:
            json.dump(fallback_survey, f, indent=2)
        print(f"STREAM: Fallback survey saved to {survey_file_path}", flush=True)
        return fallback_survey

async def generate_respondents(target_audience: str, num_respondents: int, simulation_id: str) -> List[Dict[str, Any]]:
    """Generates respondent profiles and saves them."""
    print(f"STREAM: Generating {num_respondents} respondent profiles...", flush=True)
    respondent_generator = RespondentGeneratorAgent(target_audience=target_audience, num_respondents=num_respondents)
    
    prompt = f"""Generate exactly {num_respondents} diverse, realistic respondent profile JSON objects for target audience: '{target_audience}'. Follow the mandatory JSON structure strictly, including all nested objects and fields."""
    
    result = await Runner.run(respondent_generator, prompt)
    
    # --- Define output paths ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    respondents_file_path = os.path.join(base_output_dir, "respondents.json")
    # --- End Define output paths ---

    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', result.final_output, re.DOTALL)
        respondents_json = json_match.group(1) if json_match else result.final_output
        respondents_data = json.loads(respondents_json)
        
        if not isinstance(respondents_data, list) or len(respondents_data) == 0:
             raise ValueError("Generated data is not a list or is empty")
        
        # Ensure correct number of respondents, clone if necessary
        if len(respondents_data) < num_respondents:
            print(f"STREAM: Warning: Generated {len(respondents_data)} profiles, cloning to reach {num_respondents}.", flush=True)
            while len(respondents_data) < num_respondents:
                clone = random.choice(respondents_data).copy()
                new_id_num = len(respondents_data) + 1
                clone["respondent_id"] = f"R{new_id_num:03d}" # Ensure unique ID
                # Add minor variations if needed
                if "demographics" in clone and "age" in clone["demographics"]:
                   clone["demographics"]["age"] = max(18, clone["demographics"]["age"] + random.randint(-2, 2))
                respondents_data.append(clone)
        
        respondents_data = respondents_data[:num_respondents] # Ensure exactly num_respondents

        # --- Save respondents ---
        with open(respondents_file_path, "w") as f:
            json.dump(respondents_data, f, indent=2)
        print(f"STREAM: Generated {len(respondents_data)} profiles. Saved to {respondents_file_path}", flush=True)
        return respondents_data
        
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        print(f"STREAM: Error generating respondents: {e}. Creating generic profiles.", flush=True)
        # Fallback generation (keep simple, but remove hardcoded age pattern)
        generic_respondents = []
        for i in range(1, num_respondents + 1):
             # Generate a random age within a plausible range for the fallback
             random_age = random.randint(18, 75) 
             generic_respondents.append({
                "respondent_id": f"R{i:03d}",
                # Use the randomized age here
                "demographics": {"age": random_age, "gender": "Unknown", "location_city": "N/A"}, 
                "psychographics": {"personality_traits": ["Generic"], "values": [], "interests_hobbies": []},
                "behaviors": {},
                "response_style": {"approach": "Neutral", "potential_bias": "None", "consistency": "Consistent"}
            })
        # --- Save generic respondents ---
        with open(respondents_file_path, "w") as f:
            json.dump(generic_respondents, f, indent=2)
        print(f"STREAM: Saved {num_respondents} generic profiles to {respondents_file_path}", flush=True)
        return generic_respondents

# Modified: Added simulation_id, updated save path, added streaming prints
async def collect_survey_responses(
    survey_questions: Dict[str, Any], 
    respondents: List[Dict[str, Any]], 
    simulation_id: str
) -> List[Dict[str, Any]]:
    """Collects responses from all respondents and saves raw responses."""
    print(f"STREAM: Collecting survey responses from {len(respondents)} respondents...", flush=True)
    all_responses_data = []
    
    # --- Define output path ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    raw_responses_file_path = os.path.join(base_output_dir, "survey_responses_raw.json")
    # --- End Define output path ---

    # Format survey text once
    survey_text = f"# {survey_questions.get('survey_title', 'Survey')}\n\n{survey_questions.get('survey_introduction', '')}\n\n"
    for section in survey_questions.get("sections", []):
        survey_text += f"## {section.get('section_title', 'Section')}\n{section.get('section_description', '')}\n\n"
        for q in section.get("questions", []):
            q_id = q.get('question_id', 'N/A')
            q_text = q.get('question_text', 'N/A')
            q_type = q.get('question_type', 'N/A')
            required = "Required" if q.get('required', False) else "Optional"
            options = q.get('options', [])
            scale_labels = q.get('scale_labels', [])
            
            options_str = ""
            if q_type in ['multiple_choice', 'checkbox', 'ranking']:
                options_str = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
            elif q_type == 'likert_scale':
                labels = scale_labels if scale_labels else options
                points = q.get('scale_points', len(labels))
                options_str = f"\nScale ({points}-point):\n" + "\n".join([f"- {lbl}" for lbl in labels])

            survey_text += f"QUESTION ID: {q_id}\nTYPE: {q_type} ({required})\nTEXT: {q_text}{options_str}\n\n"

    tasks = []
    for i, respondent in enumerate(respondents):
        respondent_id = respondent.get("respondent_id", f"R{i+1:03d}")
        print(f"STREAM: Preparing respondent {i+1}/{len(respondents)} ({respondent_id})...", flush=True)
        response_agent = SurveyResponseAgent(
            respondent_data=respondent,
            survey_data=survey_questions,
            topic=survey_questions.get("survey_title", "Survey")
        )
        prompt = f"""Here is the survey structure. Please complete it based *strictly* on your assigned persona and response style. Output ONLY the JSON array of your answers.

{survey_text}"""
        # Create a task for each respondent
        tasks.append(asyncio.create_task(Runner.run(response_agent, prompt), name=respondent_id))

    # Gather responses concurrently
    print(f"STREAM: Sending survey to {len(tasks)} respondents concurrently...", flush=True)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("STREAM: All respondents have replied.", flush=True)

    for i, result in enumerate(results):
        respondent_id = respondents[i].get("respondent_id", f"R{i+1:03d}")
        if isinstance(result, Exception):
            print(f"STREAM: Error collecting response from {respondent_id}: {result}", flush=True)
            response_data = [{"question_id": "ERROR", "response": f"Agent failed: {result}"}]
        else:
            response_text = result.final_output
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                response_json = json_match.group(1) if json_match else response_text
                response_data = json.loads(response_json)
                if not isinstance(response_data, list):
                    print(f"STREAM: Warning - Response from {respondent_id} not a list, wrapping.", flush=True)
                    response_data = [{"question_id": "FORMAT_WARNING", "response": response_data}]
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"STREAM: Error decoding JSON from {respondent_id}: {e}. Raw output: {response_text[:200]}...", flush=True)
                response_data = [{"question_id": "JSON_ERROR", "response": f"Invalid JSON: {e}"}]

        # Process the response
        response_set = respondents[i].copy() # Start with profile
        response_set["responses"] = response_data # Add the collected answers
        all_responses_data.append(response_set)
        
    # --- Save raw responses ---
    try:
        with open(raw_responses_file_path, "w") as f:
            json.dump(all_responses_data, f, indent=2)
        print(f"STREAM: Collected and saved raw responses for {len(all_responses_data)} respondents to {raw_responses_file_path}", flush=True)
    except Exception as e:
        print(f"STREAM: Error saving raw responses: {e}", flush=True)

    return all_responses_data


async def process_survey_data(
    survey_questions: Dict[str, Any], 
    responses: List[Dict[str, Any]], 
    simulation_id: str
) -> Dict[str, Any]:
    """Processes raw survey responses into a structured format and saves it."""
    print(f"STREAM: Processing raw responses from {len(responses)} respondents...", flush=True)
    
    # --- Define output path ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    processed_file_path = os.path.join(base_output_dir, "survey_data_processed.json")
    # --- End Define output path ---
    
    # Get all question IDs from the survey
    all_questions: Dict[str, Dict[str, Any]] = {}
    for section in survey_questions.get("sections", []):
        for question in section.get("questions", []):
            q_id = question.get("question_id")
            if q_id:  # Ensure question has an ID
                all_questions[q_id] = {
                    "text": question.get("question_text", ""),
                    "type": question.get("question_type", ""),
                    "options": question.get("options", []),  # Use options for checkbox/mc/ranking
                    "scale_labels": question.get("scale_labels", []),  # Use scale_labels for likert
                    "responses": []  # Initialize responses list
                }
    
    # Extract demographic information (simplified for processing)
    demographics = []
    for resp in responses:
        demo_data = resp.get("demographics", {})
        demo = {"respondent_id": resp.get("respondent_id", "Unknown")}
        demo.update(demo_data) # Add all demographic fields
        demographics.append(demo)
    
    # Process all responses and link to questions
    respondent_response_map: Dict[str, List[Dict[str, Any]]] = {}
    for resp in responses:
        respondent_id = resp.get("respondent_id", "Unknown")
        resp_answers = resp.get("responses", [])
        respondent_response_map[respondent_id] = resp_answers # Store for potential later use
        
        for answer in resp_answers:
            q_id = answer.get("question_id")
            response_value = answer.get("response")
            
            if q_id and q_id in all_questions:
                response_entry: Dict[str, Any] = {
                    "respondent_id": respondent_id,
                    "response": response_value
                }
                # Optionally add comment if present
                if "response_comment" in answer:
                    response_entry["comment"] = answer.get("response_comment")
                    
                all_questions[q_id]["responses"].append(response_entry)
            # else: # Log if an answer doesn't match a known question ID?
            #     print(f"Warning: Response found for unknown question ID '{q_id}' from respondent '{respondent_id}'")

    
    # Create the processed data structure
    processed_data = {
        "survey_title": survey_questions.get("survey_title", "Survey"),
        "num_respondents": len(responses),
        "demographics_summary": demographics, # Include full demographics list
        "questions_aggregated": all_questions # Aggregated responses per question
        # respondent_response_map could be included if needed downstream
    }
    
    # --- Save processed data ---
    try:
        with open(processed_file_path, "w") as f:
            json.dump(processed_data, f, indent=2)
        print(f"STREAM: Processed data saved to {processed_file_path}", flush=True)
    except Exception as e:
        print(f"STREAM: Error saving processed data: {e}", flush=True)

    return processed_data


# Modified: Added simulation_id, updated save path
async def analyze_survey_data(
    topic: str, research_objectives: str, target_audience: str,
    survey_questions: Dict[str, Any], processed_data: Dict[str, Any], 
    simulation_id: str
) -> Dict[str, Any]:
    """Analyzes processed data using SurveyAnalystAgent and saves results."""
    print("STREAM: Analyzing processed survey data...", flush=True)
    
    # --- Define output path ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    analysis_file_path = os.path.join(base_output_dir, "survey_analysis.json")
    # --- End Define output path ---

    # Create a concise summary for the analyst agent
    data_summary_for_agent = {
        "survey_info": {
            "title": survey_questions.get("survey_title", f"Survey on {topic}"),
            "topic": topic, "research_objectives": research_objectives, "target_audience": target_audience
        },
        "respondent_count": processed_data.get("num_respondents", 0),
        "question_summary": {},
        "demographic_overview": {} # Add summary of key demographics
    }
    
    # Summarize demographics
    demos = processed_data.get("demographics_summary", [])
    if demos:
        ages = [d.get('age') for d in demos if isinstance(d.get('age'), int)]
        genders = [d.get('gender', 'Unknown') for d in demos]
        incomes = [d.get('income_annual_usd') for d in demos if isinstance(d.get('income_annual_usd'), (int, float))]
        data_summary_for_agent["demographic_overview"] = {
            "count": len(demos),
            "age_avg": round(np.mean(ages), 1) if ages else 'N/A',
            "age_range": f"{min(ages)}-{max(ages)}" if ages else 'N/A',
            "gender_distribution": dict(Counter(genders)),
            "income_avg": round(np.mean(incomes)) if incomes else 'N/A'
        }

    # Summarize question responses
    for q_id, q_data in processed_data.get("questions_aggregated", {}).items():
        responses = q_data.get("responses", [])
        resp_values = [r.get("response") for r in responses]
        q_type = q_data.get("type", "")
        summary = {"text": q_data.get("text", ""), "type": q_type, "response_count": len(responses)}
        
        if q_type in ["multiple_choice", "likert_scale"]:
            counts = Counter(str(r) for r in resp_values if r is not None) # Count frequencies
            summary["response_distribution"] = dict(counts)
        elif q_type == "checkbox":
             # Flatten list of lists and count
             all_selected = [item for sublist in resp_values if isinstance(sublist, list) for item in sublist]
             summary["response_distribution"] = dict(Counter(all_selected))
        elif q_type == "open_ended":
            # Provide a few examples
             summary["example_responses"] = [r for r in resp_values if isinstance(r, str) and r][:5] # First 5 non-empty

        data_summary_for_agent["question_summary"][q_id] = summary

    # Create analyst agent and run analysis
    analyst = SurveyAnalystAgent(topic=topic, research_objectives=research_objectives, target_audience=target_audience)
    prompt = f"""Analyze the following survey data summary regarding '{topic}'. Focus on addressing the research objectives: '{research_objectives}'. Provide key findings, segment highlights, and actionable recommendations.

**Data Summary:**
{json.dumps(data_summary_for_agent, indent=2)}

**Output Requirements:**
1.  Provide your narrative analysis covering Executive Summary, Findings, Conclusions, etc.
2.  At the VERY END, include the MANDATORY JSON block with 'key_metrics', 'top_findings_summary', 'segment_highlights', and 'visualization_summary' as specified in your instructions."""

    result = await Runner.run(analyst, prompt)
    analysis_text = result.final_output.strip()
    
    # Extract JSON block
    visualization_data = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', analysis_text, re.DOTALL | re.IGNORECASE) # Case-insensitive search
    narrative_analysis = analysis_text # Default to full text

    if json_match:
        json_string = json_match.group(1)
        try:
            visualization_data = json.loads(json_string)
            # Remove JSON block from narrative
            narrative_analysis = analysis_text.replace(json_match.group(0), '').strip()
            print("STREAM: Extracted analysis JSON block.", flush=True)
        except json.JSONDecodeError as e:
            print(f"STREAM: Warning - Failed to parse analysis JSON: {e}. JSON block kept in narrative.", flush=True)
            # Keep JSON block in narrative if parsing fails
    else:
        print("STREAM: Warning - Analysis JSON block not found in analyst output.", flush=True)
        # Populate with defaults if analyst fails
        visualization_data = {
             "key_metrics": [{"metric": "NPS (Estimated)", "value": 10}],
             "top_findings_summary": ["Analysis incomplete."],
             "segment_highlights": [],
             "visualization_summary": []
        }


    analysis_results = {
        "narrative_report": narrative_analysis,
        "visualization_data": visualization_data,
        # Optionally include the summary sent to the agent for reference
        # "data_summary_for_agent": data_summary_for_agent 
    }

    # --- Save analysis results ---
    try:
        with open(analysis_file_path, "w") as f:
            json.dump(analysis_results, f, indent=2)
        print(f"STREAM: Analysis results saved to {analysis_file_path}", flush=True)
    except Exception as e:
        print(f"STREAM: Error saving analysis results: {e}", flush=True)

    return analysis_results

# Modified: Added simulation_id, updated paths
async def generate_visualizations(
    survey_questions: Dict[str, Any],
    processed_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
    simulation_id: str
) -> List[str]:
    """Generates visualizations based on survey data and saves them."""
    print("STREAM: Generating visualizations...", flush=True)
    
    # --- Define output paths ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    viz_dir = os.path.join(base_output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    viz_rel_path_prefix = f"/simulations/survey/{simulation_id}/visualizations"
    # --- End Define output paths ---
    
    generated_files_relative: List[str] = [] # Store relative paths for the report function

    # --- Robust plotting functions ---
    def save_plot(filename: str, title: str = "Plot"):
        try:
            filepath = os.path.join(viz_dir, filename)
            plt.title(title, fontsize=14, pad=20) # Add padding to title
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.savefig(filepath, dpi=150) # Use slightly lower DPI for web
            plt.close() # Close figure to free memory
            generated_files_relative.append(f"{viz_rel_path_prefix}/{filename}")
            print(f"   - Saved: {filename}")
        except Exception as e:
            print(f"   - Error saving plot {filename}: {e}")
            plt.close() # Ensure figure is closed even on error
            
    # 1. Demographic Visualizations
    demographics = processed_data.get("demographics_summary", [])
    if demographics:
        print("STREAM:   Generating demographic charts...", flush=True)
        # Age Distribution (Histogram)
        ages = [d.get('age') for d in demographics if isinstance(d.get('age'), int)]
        if ages:
            plt.figure(figsize=(8, 5))
            sns.histplot(ages, bins=10, kde=True, color='skyblue')
            save_plot("age_distribution.png", "Age Distribution")
            
        # Gender Distribution (Bar)
        genders = [d.get('gender', 'Unknown') for d in demographics]
        if genders:
            gender_counts = Counter(genders)
        if gender_counts:
                plt.figure(figsize=(7, 5))
                sns.barplot(x=list(gender_counts.keys()), y=list(gender_counts.values()), palette="pastel")
                save_plot("gender_distribution.png", "Gender Distribution")

        # Education Level (Bar) - Horizontal for readability
        educations = [d.get('education_level', 'Unknown') for d in demographics]
        if educations:
            education_counts = Counter(educations)
            if education_counts:
                sorted_edu = sorted(education_counts.items(), key=lambda item: item[1], reverse=True)
                plt.figure(figsize=(8, max(5, len(sorted_edu)*0.5))) # Dynamic height
                sns.barplot(y=[item[0] for item in sorted_edu], x=[item[1] for item in sorted_edu], orient='h', palette="viridis")
                plt.xlabel("Count")
                plt.ylabel("Education Level")
                save_plot("education_distribution.png", "Education Level Distribution")

    # 2. Question Response Visualizations
    questions = processed_data.get("questions_aggregated", {})
    print(f"STREAM:   Generating charts for {len(questions)} questions...", flush=True)
    for q_id, q_data in questions.items():
        q_type = q_data.get("type", "")
        q_text = q_data.get("text", f"Question {q_id}")
        responses = q_data.get("responses", [])
        if not responses: continue
        
        filename = f"q_{q_id.lower().replace(' ', '_')}_responses.png"
        plot_title = f"Responses: {q_text[:60]}{'...' if len(q_text)>60 else ''} (ID: {q_id})"
            
        try:
            if q_type in ["multiple_choice", "likert_scale"]:
                response_values = [r.get("response") for r in responses]
                # Filter out potential None values before counting
                valid_responses = [str(r) for r in response_values if r is not None]
                if not valid_responses: continue
                
                value_counts = Counter(valid_responses)
                sorted_items = sorted(value_counts.items(), key=lambda item: item[1], reverse=True)
                
                # Try to maintain Likert order if possible
                if q_type == "likert_scale" and q_data.get("scale_labels"):
                    ordered_keys = [lbl for lbl in q_data["scale_labels"] if lbl in value_counts]
                    if len(ordered_keys) == len(value_counts): # If all response values match scale labels
                        sorted_items = [(k, value_counts[k]) for k in ordered_keys]

                labels = [item[0] for item in sorted_items]
                values = [item[1] for item in sorted_items]
                
                # Use horizontal bar for many options or long labels
                if len(labels) > 6 or any(len(l) > 20 for l in labels):
                    plt.figure(figsize=(8, max(5, len(labels)*0.4))) # Dynamic height
                    sns.barplot(y=labels, x=values, orient='h', palette="magma")
                    plt.xlabel("Count")
                    plt.ylabel("Response")
                else:
                    plt.figure(figsize=(max(8, len(labels)*1.2), 5)) # Dynamic width
                    sns.barplot(x=labels, y=values, palette="magma")
                    plt.xlabel("Response")
                    plt.ylabel("Count")
                    if len(labels) > 4: plt.xticks(rotation=15, ha='right')

                save_plot(filename, plot_title)
                
            elif q_type == "checkbox":
                all_selected = [item for r in responses for item in r.get("response", []) if isinstance(r.get("response"), list)]
                if not all_selected: continue
                
                option_counts = Counter(all_selected)
                sorted_items = sorted(option_counts.items(), key=lambda item: item[1], reverse=True)
                labels = [item[0] for item in sorted_items]
                values = [item[1] for item in sorted_items]

                # Use horizontal bar for many options or long labels
                if len(labels) > 6 or any(len(l) > 20 for l in labels):
                    plt.figure(figsize=(8, max(5, len(labels)*0.4))) # Dynamic height
                    sns.barplot(y=labels, x=values, orient='h', palette="crest")
                    plt.xlabel("Number of Selections")
                    plt.ylabel("Option")
                else:
                    plt.figure(figsize=(max(8, len(labels)*1.2), 5)) # Dynamic width
                    sns.barplot(x=labels, y=values, palette="crest")
                    plt.xlabel("Option")
                    plt.ylabel("Number of Selections")
                    if len(labels) > 4: plt.xticks(rotation=15, ha='right')
                
                save_plot(filename, plot_title)

            # Could add basic word cloud for open_ended if needed later
            # elif q_type == "open_ended": ...
                
        except Exception as e:
            print(f"   - Error generating chart for Q{q_id}: {e}")
            plt.close() # Ensure figure is closed

    # 3. Custom Visualizations from Analysis (Placeholder)
    viz_summary = analysis_results.get("visualization_data", {}).get("visualization_summary", [])
    if viz_summary:
        print("STREAM:   Generating custom analysis charts (placeholders)...", flush=True)
        for i, viz_info in enumerate(viz_summary):
            filename = viz_info.get("filename", f"custom_analysis_{i+1}.png")
            description = viz_info.get("description", "Custom Analysis Chart")
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, f"Placeholder for:\n{description}", ha='center', va='center', fontsize=12, wrap=True)
            plt.axis('off')
            save_plot(filename, description)

    print(f"STREAM: Generated {len(generated_files_relative)} visualizations.", flush=True)
    # Return relative paths for the report function
    return generated_files_relative

def generate_final_report(
    topic: str, research_objectives: str, target_audience: str, num_respondents: int,
    analysis_results: Dict[str, Any],
    visualization_files: List[str], # Expecting relative paths now
    simulation_id: str
):
    """Generates and saves the final markdown report with improved error handling."""
    print("STREAM: Generating final analysis report...", flush=True)

    # --- Define output path ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    report_file_path = os.path.join(base_output_dir, "report.md")
    # --- End Define output path ---

    report_content = f"""# Survey Simulation Report: {topic}

**Simulation ID:** {simulation_id}
**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Research Overview
*   **Topic:** {topic}
*   **Research Objectives:** {research_objectives}
*   **Target Audience:** {target_audience}
*   **Number of Simulated Respondents:** {num_respondents}

## 2. Narrative Analysis
{analysis_results.get("narrative_report", "Analysis narrative not available.")}

## 3. Key Visualizations
"""
    # Embed relative paths to visualizations
    if visualization_files:
        for viz_path in visualization_files:
            filename = os.path.basename(viz_path)
            # Extract title from filename or use default
            title = filename.replace('.png', '').replace('_', ' ').title()
            report_content += f"### {title}\n"
            report_content += f"![{title}]({viz_path})\n\n" # Use relative path directly
    else:
        report_content += "*No visualizations were generated.*\n"

    # Serialize visualization_data with specific error handling
    viz_data_json_str = ""
    try:
        viz_data = analysis_results.get("visualization_data", {})
        # Attempt to serialize the potentially problematic structure
        viz_data_json_str = json.dumps(viz_data, indent=2)
    except TypeError as te:
        # Catch the specific "unhashable type" error if it occurs here
        print(f"STREAM: Error serializing visualization_data for report: {te}", flush=True)
        # Include the raw data (or a representation) in the report for debugging
        viz_data_json_str = f"Error serializing visualization_data: {te}\nRaw Data:\n{viz_data}"
    except Exception as e:
        # Catch any other serialization errors
        print(f"STREAM: Unexpected error serializing visualization_data: {e}", flush=True)
        viz_data_json_str = f"Unexpected error serializing visualization_data: {e}\nRaw Data:\n{analysis_results.get('visualization_data', {})}"

    # Add analysis JSON output (or error message) to the report
    report_content += f"""
<details>
<summary>Analysis JSON Output (for reference)</summary>

```json
{viz_data_json_str}
```
</details>
"""

    # --- Save Report ---
    try:
        with open(report_file_path, "w", encoding='utf-8') as f:
            f.write(report_content)
        print(f"STREAM: Final report saved to {report_file_path}", flush=True)
    except Exception as e:
        # Catch errors during file writing
        print(f"STREAM: Error saving final report file: {e}", flush=True)

# --- Main Execution Logic ---

async def run_survey_simulation(
    topic: str,
    research_objectives: str,
    target_audience: str,
    num_respondents: int,
    simulation_id: str # Added simulation_id
) -> None: # Changed return type to None as results are saved
    """
    Runs the complete survey simulation pipeline and saves all artifacts.
    """
    start_time = datetime.now()
    print(f"STREAM: Starting survey simulation {simulation_id} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"STREAM: Parameters - Topic='{topic}', Objectives='{research_objectives}', Audience='{target_audience}', N={num_respondents}", flush=True)

    # Define base output directory structure (used by sub-functions now)
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # --- Pipeline Stages ---
    try:
        # 1. Design Survey (saves survey_questions.json)
        survey_questions = await design_survey(topic, research_objectives, target_audience, simulation_id)
        
        # 2. Generate Respondents (saves respondents.json)
        respondents = await generate_respondents(target_audience, num_respondents, simulation_id)
        
        # 3. Collect Responses (saves survey_responses_raw.json)
        responses = await collect_survey_responses(survey_questions, respondents, simulation_id)
        
        # 4. Process Data (saves survey_data_processed.json)
        processed_data = await process_survey_data(survey_questions, responses, simulation_id)
        
        # 5. Analyze Results (saves survey_analysis.json)
        analysis_results = await analyze_survey_data(
            topic, research_objectives, target_audience, survey_questions, processed_data, simulation_id
        )
        
        # 6. Generate Visualizations (saves PNGs into visualizations/ subdir)
        visualization_files = await generate_visualizations(
            survey_questions, processed_data, analysis_results, simulation_id # Pass simulation_id here
        )
        
        # 7. Generate Final Report (saves report.md)
        generate_final_report( # Doesn't need to be async if just writing string
            topic, research_objectives, target_audience, len(respondents), # Use actual respondent count
            analysis_results, visualization_files, simulation_id
        )

        # Save parameters.json (similar to other simulations)
        params_path = os.path.join(base_output_dir, "parameters.json")
        with open(params_path, "w") as f:
            json.dump({
                "id": simulation_id,
        "topic": topic,
        "research_objectives": research_objectives,
        "target_audience": target_audience,
                "num_respondents": len(respondents),
                "status": "completed", # Mark as completed here
                "startTime": start_time.isoformat(),
                "endTime": datetime.now().isoformat()
            }, f, indent=2)
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"STREAM: Survey simulation {simulation_id} completed successfully at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration})", flush=True)

    except Exception as e:
        print(f"STREAM: **** Survey simulation {simulation_id} failed! ****", flush=True)
        print(f"STREAM: Error: {e}", flush=True)
        # Optionally save error status to parameters.json
        params_path = os.path.join(base_output_dir, "parameters.json")
        try:
            with open(params_path, "w") as f:
                json.dump({
                    "id": simulation_id, "topic": topic, "research_objectives": research_objectives, 
                    "target_audience": target_audience, "num_respondents": num_respondents,
                    "status": "failed", "startTime": start_time.isoformat(), "error": str(e)
                }, f, indent=2)
        except Exception as write_err:
            print(f"STREAM: Could not write failure status to parameters.json: {write_err}", flush=True)
        # Re-raise exception so the calling process knows it failed
        raise e


async def main():
    """Main entry point for command-line execution."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run a simulated survey with AI-generated respondents")
    parser.add_argument("--topic", type=str, required=True, help="The survey topic")
    parser.add_argument("--objectives", type=str, required=True, help="Research objectives for the survey")
    parser.add_argument("--audience", type=str, required=True, help="Description of the target audience")
    parser.add_argument("--respondents", type=int, default=50, help="Number of respondents to simulate")
    # Added simulation_id argument
    parser.add_argument("--simulation_id", type=str, required=True, help="Unique ID for this simulation run.")
    
    args = parser.parse_args()
    
    try:
        await run_survey_simulation(
            topic=args.topic,
            research_objectives=args.objectives,
            target_audience=args.audience,
            num_respondents=args.respondents,
            simulation_id=args.simulation_id # Pass the ID
        )
    except Exception as e:
        # Error already printed in run_survey_simulation
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    asyncio.run(main()) 

