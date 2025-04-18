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
7.  **Response Options:** Ensure `options` (for MC/Checkbox/Ranking) are comprehensive, mutually exclusive (where applicable), balanced, and logically ordered. Ensure `scale_labels` (for Likert) are balanced, clear, and directly correspond to the `scale_points`.
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
Your task is to generate exactly {num_respondents} diverse, realistic respondent profiles for a survey targeting: '{target_audience}'.

**CRITICAL: Adhere STRICTLY to the MANDATORY JSON STRUCTURE below. Pay meticulous attention to nested objects and data types. Generate highly diverse and realistic profiles.**

**MANDATORY JSON STRUCTURE:** Your output MUST be a single JSON array containing {num_respondents} respondent objects. Each object MUST conform EXACTLY to this structure, including ALL specified nested objects and fields.

```json
[ // Start of Array
  {{ // Start of Respondent Object
    "respondent_id": "String: Unique ID (e.g., 'R001')",
    "demographics": {{ // Nested JSON Object - REQUIRED
      "age": "Number: e.g., 35",
      "gender": "String: e.g., 'Female', 'Male', 'Non-binary', 'Prefer not to say'",
      "ethnicity": "String: e.g., 'Hispanic/Latino', 'White/Caucasian', 'Black/African American', 'Asian', 'Other'",
      "location_city": "String: e.g., 'Austin'",
      "location_state": "String: e.g., 'TX'",
      "location_country": "String: e.g., 'USA'",
      "education_level": "String: e.g., 'High School', 'Some College', 'Associate Degree', 'Bachelor's Degree', 'Master's Degree', 'Doctorate'",
      "education_field": "String: e.g., 'Marketing', 'Computer Science', 'Undecided', 'N/A'",
      "occupation_title": "String: e.g., 'Product Manager', 'Nurse Practitioner', 'Student', 'Unemployed', 'Retired'",
      "occupation_industry": "String: e.g., 'Technology', 'Healthcare', 'Education', 'Retail', 'N/A'",
      "income_annual_usd": "Number: e.g., 85000, 45000, 25000, 150000", // Numeric annual income
      "marital_status": "String: e.g., 'Married', 'Single', 'Divorced', 'Widowed', 'Partnered'",
      "household_composition": "String: e.g., 'Living with partner and 2 children', 'Living alone', 'Living with roommates'",
      "homeownership": "String: e.g., 'Homeowner', 'Renter'"
    }}, // End demographics object
    "psychographics": {{ // Nested JSON Object - REQUIRED
      "personality_traits": ["String", "String", "..."], // Array of 3-5 descriptive traits (e.g., "Introverted", "Analytical", "Creative")
      "values": ["String", "String", "..."], // Array of 2-4 core values (e.g., "Family", "Career Growth", "Environmentalism")
      "interests_hobbies": ["String", "String", "..."], // Array of 3-6 interests/hobbies (e.g., "Hiking", "Reading Sci-Fi", "Cooking")
      "lifestyle_notes": "String: Brief description of lifestyle (e.g., 'Active, travels often for work', 'Homebody, focuses on family and local community')",
      "media_consumption_primary": ["String", "String", "..."], // Array of 2-4 primary media sources/platforms (e.g., "LinkedIn", "NY Times", "Specific Podcasts", "YouTube Channels")
      "technology_adoption": "String: e.g., 'Innovator', 'Early Adopter', 'Early Majority', 'Late Majority', 'Laggard'"
    }}, // End psychographics object
    "behaviors": {{ // Nested JSON Object - REQUIRED
      "shopping_preferences": ["String", "String", "..."], // Array: e.g., 'Prefers online research before buying in-store', 'Value-driven', 'Brand loyal for specific categories', 'Influenced by reviews'
      "decision_making_style": "String: e.g., 'Analytical and slow', 'Impulsive', 'Seeks recommendations from peers', 'Relies on expert opinions'",
      "brand_affinities_relevant": ["String", "String", "..."] // Array: Specific brands relevant to the target audience/topic (can be empty if none strongly apply)
    }}, // End behaviors object
    "response_style": {{ // Nested JSON Object - REQUIRED
      "approach": "String: One of 'Thoughtful & Detailed', 'Concise & To-the-point', 'Slightly Rushed', 'Acquiescent/Agreeable', 'Skeptical/Critical', 'Neutral/Middle-ground', 'Distracted/Inconsistent'",
      "potential_bias": "String: e.g., 'Slight positive bias towards known brands', 'Tendency towards socially desirable answers', 'Price sensitive perspective', 'Skeptical of new tech'",
      "consistency": "String: e.g., 'Highly Consistent', 'Generally Consistent', 'Slightly Inconsistent at times'"
    }} // End response_style object
  }}, // End of Respondent Object
  // ... more respondent objects ({num_respondents} total) ...
] // End of Array
```

**PROFILE GENERATION GUIDELINES:**
-   **DIVERSITY (CRITICAL):** Ensure significant variation across ALL demographic, psychographic, and behavioral attributes. Avoid repetition. Reflect the nuances of the `{target_audience}`.
-   **REALISM & COHERENCE (CRITICAL):** Profiles MUST be internally consistent and plausible. An 'Innovator' in tech adoption should align with relevant media consumption and shopping habits. A 'Laggard' shouldn't list cutting-edge tech hobbies. Income should generally align with occupation/industry/location.
-   **SPECIFICITY:** Provide concrete details (specific job titles, income *numbers*, city/state, specific interests). Avoid vague terms like "Mid-level manager" or "Average income".
-   **RESPONSE STYLE VARIETY:** Assign diverse response styles to realistically simulate different survey-taking behaviors.

**Output Requirement:** Return ONLY the single, valid JSON array containing exactly {num_respondents} respondent profile objects adhering STRICTLY to the specified structure. NO explanations or other text.""",
            model="gpt-4.1" 
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
*   Approach: **{style.get('approach', 'Thoughtful & Detailed')}**
*   Potential Bias: {style.get('potential_bias', 'None specified')}
*   Consistency: {style.get('consistency', 'Generally Consistent')}
---

**CRITICAL INSTRUCTIONS: You MUST embody this persona COMPLETELY.**

**RESPONSE BEHAVIOR MANDATES:**
1.  **Full Embodiment (MANDATORY):** Answer EVERY question STRICTLY from the perspective of YOUR **ENTIRE** profile. Your age, income, location, occupation, education, values, interests, personality, tech adoption, lifestyle, shopping style, decision making, biases – ALL these factors MUST collectively influence your answers. **DO NOT give generic responses.**
2.  **Deep Consistency (MANDATORY):** Ensure your answers across different questions are logically coherent with your *entire* profile. An 'Innovator' with high income shouldn't express views typical of a 'Laggard' with low income, unless there's a specific reason explained by other profile elements (e.g., specific values overriding tech enthusiasm). A 'Value-driven' shopper should reflect this in choices, even if they have high income.
3.  **Response Style Adherence (MANDATORY):** CRITICALLY IMPORTANT - You MUST adopt the assigned 'Approach', 'Potential Bias', and 'Consistency' described in your 'Response Style'. This dictates the *manner* of your response (length, tone, certainty, selection patterns).
    *   *Approach Variations:* 'Thoughtful & Detailed' means longer, reasoned open-ends, careful selections. 'Concise' means brief but accurate. 'Rushed' means quick, possibly less considered choices, short/skipped open-ends. 'Acquiescent' means agreeing more often. 'Skeptical' means questioning or disagreeing more. 'Neutral' means middle options. 'Distracted' might mean slight inconsistencies or missed details.
    *   *Bias/Consistency:* Actively incorporate your specified bias (e.g., brand preference, price sensitivity, social desirability) and consistency level into your choices and wording.
4.  **Realistic Open-Ended Answers:** Write open-ended responses in a natural voice matching your persona's likely background (education, occupation, personality). Length and detail MUST match your 'Approach' (e.g., 'Thoughtful' = longer, 'Rushed' = shorter). Explicitly reference specific profile details where relevant (e.g., "As someone living in [City] with [Household Comp], this matters because..." or "My interest in [Hobby] makes me think..."). Skip non-required questions only if your style is 'Rushed' or 'Distracted'.
5.  **Selection Logic:** Choose multiple-choice/Likert/ranking options that genuinely reflect your persona's *combined* profile elements AND response style. If your persona is 'Analytical' and 'Thoughtful', your selections should reflect careful consideration. If 'Impulsive' and 'Rushed', choices might be quicker first impressions.

**Output Format:**
For each question presented, output a JSON object for your answer. Combine ALL answer objects into a single JSON array.

*   **For single-select (multiple_choice, likert_scale):**
```json
    {{ "question_id": "Q1", "response": "Selected Option Text" }}
    ```
*   **For multi-select (checkbox):**
```json
    {{ "question_id": "Q2", "response": ["Selected Option 1", "Selected Option 3"] }} 
    ```
    (Use an empty array `[]` if none selected and not required).
*   **For ranking:**
    ```json
    {{ "question_id": "Q3", "response": ["First Ranked Item", "Second Ranked Item", "..."] }} 
    ```
    (Order ALL provided options according to your persona's preference).
*   **For open_ended:**
    ```json
    {{ "question_id": "Q4", "response": "Your detailed text answer here, reflecting persona and style. Reference your profile details." }}
    ```
    (Use an empty string `""` or `null` only if skipping an optional question due to 'Rushed' or 'Distracted' style).

**Task:** Process the entire survey provided in the prompt and return ONE single JSON array containing your answer objects for ALL questions. Ensure valid JSON format. DO NOT add any commentary before or after the JSON array."""
        
        super().__init__(
            name="Survey Response",
            instructions=respondent_summary,
            model="gpt-4.1" 
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
            instructions=f"""You are a senior quantitative research analyst specializing in rigorous survey data analysis and interpretation.
You have been provided with the results of a survey on the topic: '{topic}' 
conducted to address the research objectives: '{research_objectives}' 
with the target audience: '{target_audience}'.

**CRITICAL TASK:** Analyze the provided survey data meticulously. Your primary goal is to derive **accurate, data-driven insights** and present them clearly. You MUST perform calculations based *only* on the data provided. Avoid making assumptions or using generic patterns.

**ANALYSIS REQUIREMENTS:**
1.  **Calculate Distributions:** For each categorical question (multiple_choice, likert_scale, checkbox), calculate the precise frequency distribution (counts and percentages) for each option based *strictly* on the provided response data.
2.  **Summarize Open-Ended:** For open-ended questions, perform a thematic analysis. Identify key recurring themes, quantify their approximate frequency (e.g., mentioned by X% of respondents), and provide 2-3 diverse, illustrative quotes *with respondent IDs* if available.
3.  **Cross-Tabulations:** Identify 2-3 meaningful cross-tabulations based on the research objectives and available demographic data (e.g., satisfaction score by age group, purchase intent by income level). Calculate and report the results for these specific segments. Highlight any statistically suggestive differences (even without formal tests).
4.  **Sentiment Analysis (If Applicable):** If the survey includes questions directly gauging sentiment (e.g., satisfaction Likert scales, open-ended questions about feelings), analyze and report the sentiment expressed. Quantify the distribution (positive/neutral/negative percentages) based *specifically* on responses to those questions.
5.  **Synthesize Findings:** Connect the findings from different questions to build a coherent narrative addressing the core research objectives.

**REPORT STRUCTURE:**
Generate a comprehensive report with the following sections:
1.  **Executive Summary:** Concise overview of the most critical findings and key takeaways directly addressing the research objectives (approx. 200 words). Focus on calculated metrics and significant differences.
2.  **Research Background:** Briefly restate the study purpose, objectives, and target audience.
3.  **Methodology:** Briefly describe the survey approach (e.g., number of respondents, key question types used). Mention the source of the data (simulated respondents).
4.  **Respondent Demographics:** Summarize the key demographic characteristics of the respondent pool based *only* on the provided data (e.g., age range/average, gender distribution, income average/distribution).
5.  **Detailed Findings:** Present the analysis question-by-question or grouped by themes derived from the objectives.
    *   For each question, report the calculated frequency distributions (percentages are crucial).
    *   Present results of your cross-tabulations clearly.
    *   Summarize thematic analysis for open-ended questions with quotes.
6.  **Data Visualizations Description:** Describe the key charts generated (interpret what they show based *only* on the data). *You will be provided with filenames later; describe the insights the charts SHOULD convey based on your analysis.*
7.  **Conclusions & Recommendations:** Draw strategic conclusions based *strictly* on your analysis. Provide actionable recommendations directly linked to the findings and research objectives.
8.  **Limitations:** Briefly mention the limitations of simulated data (e.g., lack of real-world nuance, potential persona inconsistencies).

**MANDATORY JSON OUTPUT:**
AFTER your narrative analysis, include a JSON-formatted section containing structured data derived *directly and accurately* from YOUR analysis of the provided input data. **DO NOT use placeholder values or generic distributions.** Calculate these values.

```json
{{
  "key_metrics": [ // Calculated summary metrics
    // Example: {{"metric": "Overall Satisfaction (Avg. Likert Score)", "value": 3.8, "scale": 5, "question_id": "Q5"}}, // Calculate if applicable
    // Example: {{"metric": "Top Box Purchase Intent ('Very Likely' %)", "value": 28.0, "question_id": "Q12"}}, // Calculate if applicable
    // Add 2-4 KEY calculated metrics relevant to objectives
  ],
  "top_findings_summary": [ // 3-5 major quantitative insights from your analysis
    // Example: "Respondents aged 18-34 reported significantly higher usage frequency (Avg: 4.5/5) compared to 55+ (Avg: 2.8/5) for Q7.", // Based on calculated cross-tab
    // Example: "Cost ('Price' theme) was the most frequently cited barrier in open-ended question Q15, mentioned by 42% of respondents.", // Based on calculated theme frequency
    // Add precise findings based on YOUR calculations.
  ],
  "segment_highlights": [ // Specific calculated insights about key demographic/behavioral groups
    // Example: {{"segment": "High Income (>$100k)", "insight": "Showed 15% higher agreement with 'Willingness to pay premium' (Q8) than lower-income segments.", "supporting_questions": ["Q8", "DEMO_Income"]}}, // Based on calculated cross-tab
    // Add 2-4 specific, calculated segment insights.
  ],
  "demographic_summary_stats": {{ // Calculated demographic statistics
    "respondent_count": <calculated_total_respondents>,
    "age_average": <calculated_average_age>,
    "age_median": <calculated_median_age>,
    "gender_distribution_percent": {{ // Calculate percentages
      // "Female": 52.0, "Male": 46.0, "Non-binary": 1.0, "Prefer not to say": 1.0
    }},
    "income_average_usd": <calculated_average_income> // If available
    // Add other key calculated demographic stats as available
  }},
  "visualization_summary": [ // Describe insights for charts based on YOUR analysis
     // Example: {{"filename": "age_distribution.png", "description": "The calculated age distribution shows a peak in the 35-44 age group (35% of respondents), with fewer respondents in the 18-24 (15%) and 65+ (10%) brackets."}},
     // Example: {{"filename": "q3_satisfaction.png", "description": "Analysis of Q3 reveals moderate overall satisfaction (Average score: 3.2/5). The most frequent response was 'Neutral' (40%), followed by 'Satisfied' (30%). 'Very Dissatisfied' was least common (5%)."}},
     // Add descriptions for key charts reflecting calculated results.
  ]
}}
```

Adhere strictly to these instructions. Accuracy and data fidelity are paramount.""",
            model="gpt-4.1" 
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
    
    prompt = f"""Design a comprehensive survey JSON object for topic '{topic}', objectives '{research_objectives}', audience '{target_audience}'. Follow the mandatory JSON structure and design principles precisely. Aim for depth (15-25+ questions, 3-5+ sections).""" # Keep prompt concise
    
    result = await Runner.run(survey_designer, prompt)
    response_text = result.final_output
    
    # --- Define output paths ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    survey_file_path = os.path.join(base_output_dir, "survey_questions.json")
    # --- End Define output paths ---

    survey_data = None
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            survey_json = json_match.group(1).strip()
        else:
            # Attempt to parse the whole string if no code block found
            survey_json = response_text.strip()
            # Basic check: does it start like JSON?
            if not (survey_json.startswith('{') and survey_json.endswith('}')):
                 raise ValueError("Response does not appear to be valid JSON.")

        survey_data = json.loads(survey_json)
        
        # --- START Data Validation ---
        if not isinstance(survey_data, dict):
            raise ValueError("Parsed JSON is not a dictionary.")
        if "survey_title" not in survey_data or not isinstance(survey_data["survey_title"], str):
             raise ValueError("Missing or invalid 'survey_title'.")
        if "sections" not in survey_data or not isinstance(survey_data["sections"], list) or not survey_data["sections"]:
             raise ValueError("Missing, invalid, or empty 'sections' array.")
        total_questions = 0
        for i, section in enumerate(survey_data["sections"]):
            if not isinstance(section, dict):
                 raise ValueError(f"Section {i} is not a dictionary.")
            if "section_title" not in section or not isinstance(section["section_title"], str):
                 raise ValueError(f"Section {i} missing or invalid 'section_title'.")
            if "questions" not in section or not isinstance(section["questions"], list):
                 raise ValueError(f"Section {i} missing or invalid 'questions' array.")
            for j, q in enumerate(section["questions"]):
                if not isinstance(q, dict):
                     raise ValueError(f"Section {i}, Question {j} is not a dictionary.")
                if not all(k in q for k in ["question_id", "question_text", "question_type", "required"]):
                     raise ValueError(f"Section {i}, Question {j} missing required keys (id, text, type, required).")
                # Add more checks for options based on type if needed
                total_questions += 1
        print(f"STREAM: Survey JSON basic validation passed. Found {len(survey_data['sections'])} sections, {total_questions} questions.")
        # --- END Data Validation ---
            
        # --- Save survey design ---
        with open(survey_file_path, "w") as f:
            json.dump(survey_data, f, indent=2)
        print(f"STREAM: Survey design complete and saved to {survey_file_path}", flush=True)
        return survey_data
        
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        print(f"STREAM: Error processing/validating survey design: {e}", flush=True)
        print(f"STREAM: Raw response snippet: {response_text[:500]}...", flush=True) # Log snippet for debugging
        print(f"STREAM: Creating and saving fallback survey.", flush=True)
        # Fallback survey
        fallback_survey = {
            "survey_title": f"{topic} Survey (Fallback)",
            "survey_introduction": "Basic survey due to design error.",
            "sections": [{"section_title": "Basic Questions", "section_description":"", "questions": [
                {"question_id": "FALLBACK_Q1", "question_text": "Age?", "question_type": "multiple_choice", "options": ["<25", "25-44", "45+"], "required": True},
                {"question_id": "FALLBACK_Q2", "question_text": f"Overall opinion on {topic}?", "question_type": "open_ended", "required": False}
            ]}]
        }
        # --- Save fallback survey ---
        try:
            with open(survey_file_path, "w") as f:
                json.dump(fallback_survey, f, indent=2)
            print(f"STREAM: Fallback survey saved to {survey_file_path}", flush=True)
        except Exception as save_e:
            print(f"STREAM: CRITICAL ERROR - Could not save fallback survey: {save_e}", flush=True)
        return fallback_survey


async def generate_respondents(target_audience: str, num_respondents: int, simulation_id: str) -> List[Dict[str, Any]]:
    """Generates respondent profiles and saves them."""
    print(f"STREAM: Generating {num_respondents} respondent profiles using gpt-4.1...", flush=True)
    respondent_generator = RespondentGeneratorAgent(target_audience=target_audience, num_respondents=num_respondents)
    
    prompt = f"""Generate exactly {num_respondents} diverse, realistic respondent profile JSON objects for target audience: '{target_audience}'. Follow the mandatory JSON structure strictly, including all nested objects and fields. Ensure high diversity and realism.""" # Concise prompt
    
    result = await Runner.run(respondent_generator, prompt)
    response_text = result.final_output
    
    # --- Define output paths ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    respondents_file_path = os.path.join(base_output_dir, "respondents.json")
    # --- End Define output paths ---

    respondents_data = None
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
             respondents_json = json_match.group(1)
        else:
            respondents_json = response_text.strip()
            if not (respondents_json.startswith('[') and respondents_json.endswith(']')):
                 raise ValueError("Response does not appear to be valid JSON array.")

        respondents_data = json.loads(respondents_json)
        
        # --- START Data Validation ---
        if not isinstance(respondents_data, list):
            raise ValueError("Generated data is not a list.")
        if len(respondents_data) == 0:
             raise ValueError("Generated list is empty.")

        print(f"STREAM: Generated {len(respondents_data)} profiles initially.", flush=True)

        validated_respondents = []
        required_keys = ["respondent_id", "demographics", "psychographics", "behaviors", "response_style"]
        required_nested_keys = {
            "demographics": ["age", "gender", "location_city", "occupation_title", "income_annual_usd"],
            "psychographics": ["personality_traits", "values", "interests_hobbies", "technology_adoption"],
            "behaviors": ["shopping_preferences", "decision_making_style"],
            "response_style": ["approach", "potential_bias", "consistency"]
        }

        for i, profile in enumerate(respondents_data):
            if not isinstance(profile, dict):
                 print(f"STREAM: Warning - Profile {i} is not a dictionary, skipping.")
                 continue
            if not all(key in profile for key in required_keys):
                 print(f"STREAM: Warning - Profile {i} (ID: {profile.get('respondent_id', 'N/A')}) missing one or more top-level keys ({required_keys}), skipping.")
                 continue

            valid_nested = True
            for nested_key, sub_keys in required_nested_keys.items():
                if not isinstance(profile.get(nested_key), dict):
                    print(f"STREAM: Warning - Profile {i} (ID: {profile.get('respondent_id', 'N/A')}) '{nested_key}' is not a dictionary, skipping.")
                    valid_nested = False
                    break
                if not all(sub_key in profile[nested_key] for sub_key in sub_keys):
                     print(f"STREAM: Warning - Profile {i} (ID: {profile.get('respondent_id', 'N/A')}) '{nested_key}' is missing required sub-keys, skipping.")
                     valid_nested = False
                     break
            if valid_nested:
                 validated_respondents.append(profile)

        respondents_data = validated_respondents
        print(f"STREAM: {len(respondents_data)} profiles passed validation.", flush=True)

        if not respondents_data:
             raise ValueError("No valid respondent profiles were generated after validation.")

        # Ensure correct number of respondents, clone if necessary from validated ones
        if len(respondents_data) < num_respondents:
            print(f"STREAM: Warning: Validated {len(respondents_data)} profiles, cloning to reach {num_respondents}.", flush=True)
            cloned_count = 0
            while len(respondents_data) < num_respondents:
                clone_source = random.choice(validated_respondents) # Clone only from validated ones
                clone = json.loads(json.dumps(clone_source)) # Deep copy
                new_id_num = len(validated_respondents) + cloned_count + 1 # Ensure unique ID logic is sound
                clone["respondent_id"] = f"R{new_id_num:03d}_clone" # Mark clones
                # Add minor variations
                if "demographics" in clone and isinstance(clone.get("demographics"), dict) and "age" in clone["demographics"]:
                    original_age = clone["demographics"]["age"]
                    if isinstance(original_age, (int, float)):
                         clone["demographics"]["age"] = max(18, int(original_age + random.randint(-3, 3)))
                    else: # Handle case where age might not be numeric initially
                        clone["demographics"]["age"] = random.randint(18,75)

                respondents_data.append(clone)
                cloned_count += 1
        
        respondents_data = respondents_data[:num_respondents] # Ensure exactly num_respondents
        # --- END Data Validation ---

        # --- Save respondents ---
        with open(respondents_file_path, "w") as f:
            json.dump(respondents_data, f, indent=2)
        print(f"STREAM: Generated and validated {len(respondents_data)} profiles. Saved to {respondents_file_path}", flush=True)
        return respondents_data
        
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        print(f"STREAM: Error generating/validating respondents: {e}", flush=True)
        print(f"STREAM: Raw response snippet: {response_text[:500]}...", flush=True)
        print(f"STREAM: Creating {num_respondents} generic profiles as fallback.", flush=True)
        # Fallback generation
        generic_respondents = []
        for i in range(1, num_respondents + 1):
             random_age = random.randint(18, 75) 
             generic_respondents.append({
                "respondent_id": f"R{i:03d}_fallback",
                "demographics": {"age": random_age, "gender": "Unknown", "location_city": "N/A", "location_state": "N/A", "location_country": "N/A", "education_level": "N/A", "education_field": "N/A", "occupation_title": "N/A", "occupation_industry": "N/A", "income_annual_usd": 50000, "marital_status": "N/A", "household_composition": "N/A", "homeownership": "N/A"},
                "psychographics": {"personality_traits": ["Neutral"], "values": [], "interests_hobbies": [], "lifestyle_notes": "N/A", "media_consumption_primary": [], "technology_adoption": "Mainstream"},
                "behaviors": {"shopping_preferences": [], "decision_making_style": "Moderate", "brand_affinities_relevant": []},
                "response_style": {"approach": "Neutral/Middle-ground", "potential_bias": "None", "consistency": "Generally Consistent"}
            })
        # --- Save generic respondents ---
        try:
            with open(respondents_file_path, "w") as f:
                json.dump(generic_respondents, f, indent=2)
            print(f"STREAM: Saved {num_respondents} generic profiles to {respondents_file_path}", flush=True)
        except Exception as save_e:
            print(f"STREAM: CRITICAL ERROR - Could not save fallback respondents: {save_e}", flush=True)
        return generic_respondents

# Modified: Added simulation_id, updated save path, added streaming prints
async def collect_survey_responses(
    survey_questions: Dict[str, Any], 
    respondents: List[Dict[str, Any]], 
    topic: str, # Added topic explicitly
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

    # Format survey text once (ensure structure is clear for the agent)
    survey_text = f"# Survey: {survey_questions.get('survey_title', topic)}\n\n"
    survey_text += f"Introduction: {survey_questions.get('survey_introduction', '')}\n\n---\n"
    for section in survey_questions.get("sections", []):
        survey_text += f"## Section: {section.get('section_title', 'Section')}\n"
        if section.get('section_description'):
             survey_text += f"Description: {section.get('section_description', '')}\n"
        survey_text += "\n"
        for q_num, q in enumerate(section.get("questions", [])):
            q_id = q.get('question_id', f'S{section.get("section_title","")}_Q{q_num+1}') # Fallback ID
            q_text = q.get('question_text', 'N/A')
            q_type = q.get('question_type', 'N/A')
            required_str = "**Required**" if q.get('required', False) else "Optional"
            options = q.get('options', [])
            scale_labels = q.get('scale_labels', []) # Prefer scale_labels for Likert

            survey_text += f"**Question ID:** `{q_id}` ({required_str})\n"
            survey_text += f"**Type:** {q_type}\n"
            survey_text += f"**Text:** {q_text}\n"

            # Format options clearly
            if q_type in ['multiple_choice', 'checkbox', 'ranking']:
                if options:
                     survey_text += "**Options:**\n" + "\n".join([f"- `{opt}`" for opt in options]) + "\n"
            elif q_type == 'likert_scale':
                labels_to_use = scale_labels if scale_labels else options # Use options as fallback
                points = q.get('scale_points', len(labels_to_use))
                if labels_to_use:
                    scale_type_info = f" ({q.get('scale_type', 'Scale')})" if q.get('scale_type') else ""
                    survey_text += f"**Scale{scale_type_info} ({points}-point):**\n" + "\n".join([f"- `{lbl}`" for lbl in labels_to_use]) + "\n"
            elif q_type == 'open_ended':
                 survey_text += "**Response:** (Provide your text answer)\n"
            else:
                 survey_text += "(Unknown Question Type Format)\n"

            survey_text += "\n" # Space between questions
        survey_text += "---\n" # Space between sections


    tasks = []
    for i, respondent in enumerate(respondents):
        respondent_id = respondent.get("respondent_id", f"R{i+1:03d}")
        print(f"STREAM: Preparing respondent {i+1}/{len(respondents)} ({respondent_id})...", flush=True)
        try:
            response_agent = SurveyResponseAgent(
                respondent_data=respondent,
                survey_data=survey_questions, # Pass full survey data if needed by agent init
                topic=topic # Pass topic from main function
            )
            # Construct the prompt for the agent
            prompt = f"""You are respondent {respondent_id}. Please complete the following survey based STRICTLY on your assigned persona and response style, detailed in your instructions. Output ONLY the JSON array of your answers, following the specified format precisely.

{survey_text}

**Reminder:** Your entire profile (demographics, psychographics, behaviors) and response style (approach, bias, consistency) MUST dictate your answers. Provide realistic, consistent, and persona-driven responses. Refer back to your detailed instructions if needed."""

            # Create a task for each respondent
            tasks.append(asyncio.create_task(Runner.run(response_agent, prompt), name=respondent_id))
        except Exception as agent_init_e:
            print(f"STREAM: Error initializing agent for {respondent_id}: {agent_init_e}", flush=True)
            # Add a placeholder error task result if initialization fails
            async def error_task():
                return Exception(f"Agent initialization failed for {respondent_id}: {agent_init_e}")
            tasks.append(asyncio.create_task(error_task(), name=respondent_id))


    # Gather responses concurrently
    if tasks:
        print(f"STREAM: Sending survey to {len(tasks)} respondents concurrently...", flush=True)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print("STREAM: All respondents have replied.", flush=True)
    else:
        print("STREAM: No respondent tasks were created. Skipping response collection.")
        results = []


    for i, result in enumerate(results):
        # Use the task name attribute to get the respondent ID reliably
        task_name = tasks[i].get_name() if i < len(tasks) else f"Unknown_Task_{i}"
        respondent_id = task_name # Use task name as ID

        # Find the original respondent data based on the ID from the task name
        # This assumes task names were set correctly using respondent IDs during task creation
        original_respondent_data = next((r for r in respondents if r.get("respondent_id") == respondent_id), None)
        if original_respondent_data is None:
             print(f"STREAM: Warning - Could not find original data for respondent ID '{respondent_id}' from task name. Using index {i} as fallback.", flush=True)
             original_respondent_data = respondents[i] if i < len(respondents) else {"respondent_id": respondent_id, "demographics": {}, "psychographics": {}, "behaviors": {}, "response_style": {}} # Fallback structure


        response_set = original_respondent_data.copy() # Start with profile
        response_set["responses"] = [] # Default to empty list

        if isinstance(result, Exception):
            print(f"STREAM: Error collecting response from {respondent_id}: {result}", flush=True)
            response_set["responses"] = [{"question_id": "AGENT_ERROR", "response": f"Agent failed: {result}"}]
        else:
            response_text = result.final_output.strip() # Strip whitespace
            response_data = None
            try:
                # Look for JSON code block first
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_json = json_match.group(1).strip()
                else:
                    # If no block, assume entire response is JSON, but validate start/end chars
                    if response_text.startswith('[') and response_text.endswith(']'):
                        response_json = response_text
                    else:
                         raise ValueError("Response is not a JSON array and no JSON code block found.")

                response_data = json.loads(response_json)

                # --- Response Validation ---
                if not isinstance(response_data, list):
                     print(f"STREAM: Warning - Response from {respondent_id} is not a list, wrapping. Raw: {response_json[:100]}...", flush=True)
                     response_set["responses"] = [{"question_id": "FORMAT_WARNING_NOT_LIST", "response": response_data}]
                else:
                    # Check individual answer format (basic)
                    valid_answers = []
                    all_q_ids_in_survey = {q['question_id'] for s in survey_questions.get('sections', []) for q in s.get('questions', [])}
                    provided_q_ids = set()
                    for answer_num, answer in enumerate(response_data):
                        if isinstance(answer, dict) and "question_id" in answer and "response" in answer:
                            valid_answers.append(answer)
                            provided_q_ids.add(answer["question_id"])
                        else:
                            print(f"STREAM: Warning - Invalid answer format from {respondent_id} at index {answer_num}: {str(answer)[:100]}...", flush=True)
                    response_set["responses"] = valid_answers # Store only valid answers

                    # Check if all required questions were answered
                    missing_required = []
                    for s in survey_questions.get('sections', []):
                        for q in s.get('questions', []):
                            if q.get('required') and q['question_id'] not in provided_q_ids:
                                missing_required.append(q['question_id'])
                    if missing_required:
                        print(f"STREAM: Warning - Respondent {respondent_id} missed required questions: {missing_required}", flush=True)
                        # Optionally add a note about missing required questions
                        response_set["responses"].append({"question_id": "MISSING_REQUIRED", "response": missing_required})

            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                print(f"STREAM: Error decoding/validating JSON from {respondent_id}: {e}. Raw output: {response_text[:200]}...", flush=True)
                response_set["responses"] = [{"question_id": "JSON_ERROR", "response": f"Invalid JSON: {e}"}]

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
    
    # Get all question details from the survey for reference
    all_questions_details: Dict[str, Dict[str, Any]] = {}
    for section in survey_questions.get("sections", []):
        for question in section.get("questions", []):
            q_id = question.get("question_id")
            if q_id:  # Ensure question has an ID
                all_questions_details[q_id] = {
                    "text": question.get("question_text", ""),
                    "type": question.get("question_type", ""),
                    "required": question.get("required", False),
                    "options": question.get("options", []),
                    "scale_labels": question.get("scale_labels", []),
                    "scale_points": question.get("scale_points"),
                    "scale_type": question.get("scale_type"),
                    "responses": []  # Initialize responses list
                }
    
    # Extract demographic and profile information cleanly
    profiles_summary = []
    for resp_data in responses:
        # Ensure base keys exist
        profile = {
            "respondent_id": resp_data.get("respondent_id", "Unknown"),
            "demographics": resp_data.get("demographics", {}) if isinstance(resp_data.get("demographics"), dict) else {},
            "psychographics": resp_data.get("psychographics", {}) if isinstance(resp_data.get("psychographics"), dict) else {},
            "behaviors": resp_data.get("behaviors", {}) if isinstance(resp_data.get("behaviors"), dict) else {},
            "response_style": resp_data.get("response_style", {}) if isinstance(resp_data.get("response_style"), dict) else {}
        }
        profiles_summary.append(profile)

    # Process all responses and link to questions, handle potential errors
    respondent_errors = Counter()
    for resp_data in responses:
        respondent_id = resp_data.get("respondent_id", "Unknown")
        resp_answers = resp_data.get("responses", []) # This should be a list of answer dicts

        if not isinstance(resp_answers, list):
             print(f"STREAM: Warning - Responses for {respondent_id} is not a list, skipping processing. Type: {type(resp_answers)}")
             respondent_errors[respondent_id] += 1
             continue
        
        for answer in resp_answers:
            if not isinstance(answer, dict):
                print(f"STREAM: Warning - Answer item for {respondent_id} is not a dictionary, skipping: {str(answer)[:50]}...")
                respondent_errors[respondent_id] += 1
                continue

            q_id = answer.get("question_id")
            response_value = answer.get("response") # Can be string, list, null, etc.

            # Handle agent errors explicitly
            if q_id in ["AGENT_ERROR", "JSON_ERROR", "FORMAT_WARNING_NOT_LIST", "MISSING_REQUIRED"]:
                 print(f"STREAM: Note - Found agent-generated error/note for {respondent_id}: {q_id} = {response_value}")
                 respondent_errors[respondent_id] += 1
                 continue # Don't add these to question responses

            if q_id and q_id in all_questions_details:
                # Basic type validation based on question type
                q_type = all_questions_details[q_id].get("type")
                is_valid_response = True
                if q_type == "checkbox" and not isinstance(response_value, list):
                     # Allow empty string/null if optional? For now, assume list unless empty
                     if response_value is not None and response_value != "" :
                         print(f"STREAM: Warning - Type mismatch for {respondent_id}, Q:{q_id} (checkbox). Expected list, got {type(response_value)}. Value: {str(response_value)[:50]}")
                         # Try to recover if it's a single string that might be an option?
                         # response_value = [str(response_value)] # Example recovery - careful!
                         is_valid_response = False # Mark as invalid for now
                elif q_type == "ranking" and not isinstance(response_value, list):
                     if response_value is not None and response_value != "" :
                         print(f"STREAM: Warning - Type mismatch for {respondent_id}, Q:{q_id} (ranking). Expected list, got {type(response_value)}. Value: {str(response_value)[:50]}")
                         is_valid_response = False
                elif q_type == "open_ended" and not isinstance(response_value, (str, type(None))):
                     print(f"STREAM: Warning - Type mismatch for {respondent_id}, Q:{q_id} (open_ended). Expected string/null, got {type(response_value)}. Value: {str(response_value)[:50]}")
                     response_value = str(response_value) # Force to string
                elif q_type in ["multiple_choice", "likert_scale"] and not isinstance(response_value, (str, type(None))):
                      if response_value is not None and response_value != "" :
                         print(f"STREAM: Warning - Type mismatch for {respondent_id}, Q:{q_id} ({q_type}). Expected string/null, got {type(response_value)}. Value: {str(response_value)[:50]}")
                         response_value = str(response_value) # Force to string? Risky if it was meant to be list etc. Mark invalid?

                if is_valid_response:
                    response_entry: Dict[str, Any] = {
                        "respondent_id": respondent_id,
                        "response": response_value
                    }
                    all_questions_details[q_id]["responses"].append(response_entry)
                else:
                    respondent_errors[respondent_id] += 1
            elif q_id:  # q_id exists but not in survey?
                print(f"STREAM: Warning - Response found for unknown question ID '{q_id}' from respondent '{respondent_id}'. Discarding.")
                respondent_errors[respondent_id] += 1

    
    # Create the processed data structure
    processed_data = {
        "survey_metadata": {
        "survey_title": survey_questions.get("survey_title", "Survey"),
            "simulation_id": simulation_id,
            "num_respondents_processed": len(responses),
            "respondent_error_counts": dict(respondent_errors) # Include error counts per respondent
        },
        "respondent_profiles": profiles_summary, # Include full profiles list
        "questions_aggregated": all_questions_details # Aggregated responses per question
    }
    
    # --- Save processed data ---
    try:
        with open(processed_file_path, "w") as f:
            # Use default=str for potential datetime objects, though unlikely here
            json.dump(processed_data, f, indent=2, default=str)
        print(f"STREAM: Processed data saved to {processed_file_path}", flush=True)
    except Exception as e:
        print(f"STREAM: Error saving processed data: {e}", flush=True)

    return processed_data


async def analyze_survey_data(
    topic: str, research_objectives: str, target_audience: str,
    survey_questions: Dict[str, Any], # Keep original questions for context
    processed_data: Dict[str, Any],
    simulation_id: str
) -> Dict[str, Any]:
    """Analyzes processed data using SurveyAnalystAgent and saves results."""
    print("STREAM: Analyzing processed survey data using gpt-4.1...", flush=True)
    
    # --- Define output path ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "survey", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    analysis_file_path = os.path.join(base_output_dir, "survey_analysis.json")
    # --- End Define output path ---

    # --- Create a DETAILED summary for the analyst agent ---
    # Include survey structure AND aggregated responses for accurate calculation
    data_summary_for_agent = {
        "analysis_instructions": "Analyze the following survey data meticulously. Calculate all metrics and distributions *directly* from the provided 'questions_aggregated' data. Do not use generic patterns. Provide the narrative report AND the structured JSON output as requested.",
        "survey_metadata": processed_data.get("survey_metadata", {}),
        "research_context": {
            "topic": topic, "research_objectives": research_objectives, "target_audience": target_audience
        },
         "respondent_profiles_summary": [], # Add key profile stats below
        "survey_structure_and_aggregated_responses": {} # Combine structure & responses
    }

    # Summarize respondent profiles (more detail for analyst context)
    profiles = processed_data.get("respondent_profiles", [])
    data_summary_for_agent["survey_metadata"]["num_respondents_analyzed"] = len(profiles) # Actual count analyzed
    if profiles:
        # Calculate actual stats from profiles data
        ages = [p['demographics'].get('age') for p in profiles if isinstance(p.get('demographics', {}).get('age'), (int, float))]
        genders = [p['demographics'].get('gender', 'Unknown') for p in profiles]
        incomes = [p['demographics'].get('income_annual_usd') for p in profiles if isinstance(p.get('demographics', {}).get('income_annual_usd'), (int, float))]
        tech_adoption = [p['psychographics'].get('technology_adoption', 'Unknown') for p in profiles]

        demo_stats = {
            "count": len(profiles),
            "age_avg": round(np.mean(ages), 1) if ages else None,
            "age_median": int(np.median(ages)) if ages else None,
            "age_min": min(ages) if ages else None,
            "age_max": max(ages) if ages else None,
            "gender_distribution_percent": dict(Counter(genders)),
            "income_avg": int(round(np.mean(incomes))) if incomes else None,
            "income_median": int(np.median(incomes)) if incomes else None,
            "income_min": min(incomes) if incomes else None,
            "income_max": max(incomes) if incomes else None,
            "technology_adoption_distribution": dict(Counter(tech_adoption)),
             # Add more stats as needed (e.g., location distribution)
        }
        data_summary_for_agent["respondent_profiles_summary"] = demo_stats
    else:
        data_summary_for_agent["respondent_profiles_summary"] = {"count": 0}


    # Add detailed question structure and *all* aggregated responses
    questions_agg = processed_data.get("questions_aggregated", {})
    for q_id, q_data in questions_agg.items():
        # Prepare data for analyst: structure + all individual responses
        # This gives the LLM the raw data to perform calculations on.
        data_summary_for_agent["survey_structure_and_aggregated_responses"][q_id] = {
            "question_text": q_data.get("text", ""),
            "question_type": q_data.get("type", ""),
            "options": q_data.get("options", []), # Include options/labels for context
            "scale_labels": q_data.get("scale_labels", []),
            "scale_points": q_data.get("scale_points"),
            "scale_type": q_data.get("scale_type"),
            "required": q_data.get("required"),
            "responses_data": q_data.get("responses", []) # Provide the list of {"respondent_id": ..., "response": ...}
        }
    # --- End detailed summary preparation ---

    # Create analyst agent and run analysis
    analyst = SurveyAnalystAgent(topic=topic, research_objectives=research_objectives, target_audience=target_audience)

    # Construct the prompt clearly
    prompt = f"""Please perform a detailed analysis of the following survey data according to your instructions. Calculate all required metrics and distributions based *only* on the provided data.

**Survey Data:**
```json
{json.dumps(data_summary_for_agent, indent=2, default=str)}
```

**Output Requirements:**
1.  Provide your full narrative analysis report (Executive Summary, Findings, Conclusions, etc.).
2.  At the VERY END of your response, include the MANDATORY JSON block containing the accurately calculated metrics ('key_metrics', 'top_findings_summary', 'segment_highlights', 'demographic_summary_stats', 'visualization_summary') as specified in your instructions. Ensure all values in the JSON are calculated from the input data."""

    result = await Runner.run(analyst, prompt)
    analysis_text = result.final_output.strip()
    
    # --- Extract and Validate JSON block ---
    visualization_data = None
    narrative_analysis = analysis_text # Default to full text
    analysis_json_parsed = {} # To store the parsed JSON content

    # Regex to find the JSON block (case-insensitive)
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_text, re.DOTALL | re.IGNORECASE)

    if json_match:
        json_string = json_match.group(1).strip()
        try:
            analysis_json_parsed = json.loads(json_string)
            # Basic structural validation of the parsed JSON
            required_json_keys = ["key_metrics", "top_findings_summary", "segment_highlights", "demographic_summary_stats", "visualization_summary"]
            if all(key in analysis_json_parsed for key in required_json_keys):
                # Remove JSON block from narrative
                narrative_analysis = analysis_text.replace(json_match.group(0), '').strip()
                visualization_data = analysis_json_parsed # Use the parsed data
                print("STREAM: Extracted and validated analysis JSON block structure.", flush=True)
            else:
                missing_keys = [key for key in required_json_keys if key not in analysis_json_parsed]
                print(f"STREAM: Warning - Parsed analysis JSON is missing required keys: {missing_keys}. JSON block kept in narrative.", flush=True)
                # Keep JSON block in narrative if structure is wrong
                analysis_json_parsed = {"error": "JSON structure validation failed", "missing_keys": missing_keys}
        except json.JSONDecodeError as e:
            print(f"STREAM: Warning - Failed to parse analysis JSON: {e}. JSON block kept in narrative.", flush=True)
            analysis_json_parsed = {"error": f"JSON parsing failed: {e}", "raw_json_string": json_string}
            # Keep JSON block in narrative if parsing fails
    else:
        print("STREAM: Warning - Analysis JSON block not found in analyst output. Using defaults.", flush=True)
        # Populate with defaults if analyst fails completely
        analysis_json_parsed = {
            "error": "No JSON block found in output",
            "key_metrics": [{"metric": "Analysis Incomplete", "value": None}],
            "top_findings_summary": ["Analysis JSON block generation failed."],
            "segment_highlights": [],
            "demographic_summary_stats": data_summary_for_agent.get("respondent_profiles_summary", {}), # Use calculated stats if possible
            "visualization_summary": []
        }
        visualization_data = analysis_json_parsed # Use the default structure

    analysis_results = {
        "narrative_report": narrative_analysis,
        "visualization_data": visualization_data, # This now holds the parsed or default JSON
        # Optionally include the summary sent to the agent for debugging/reference
        # "data_summary_for_agent": data_summary_for_agent 
    }
    # --- End Extract and Validate JSON block ---


    # --- Save analysis results ---
    try:
        with open(analysis_file_path, "w") as f:
            # Use default=str for potential non-serializable types (like numpy types if analyst calculates them directly)
            json.dump(analysis_results, f, indent=2, default=str)
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
        responses = await collect_survey_responses(survey_questions, respondents, topic, simulation_id)
        
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

