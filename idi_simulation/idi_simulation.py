import os
import asyncio
import re
import json
import argparse
import sys
import matplotlib.pyplot as plt
import datetime
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from agents import Agent, Runner
from typing import List, Dict, Any, Tuple
import csv
from collections import Counter

# Load environment variables from .env file
load_dotenv()

# --- Agent Definitions ---

class InterviewerAgent(Agent):
    """
    Expert interviewer who conducts the in-depth interview,
    asking questions and follow-ups to gain deep insights.
    """
    def __init__(self, topic: str, respondent_profile: str, num_questions: int = 8):
        super().__init__(
            name="IDI Interviewer",
            instructions=f"""You are an expert qualitative researcher and interviewer with over 15 years of experience in in-depth interviews.
Your goal is to conduct an insightful and productive in-depth interview on the topic: '{topic}'.
You're interviewing a respondent with the following profile: {respondent_profile}.

INTERVIEW APPROACH:
- Start with a warm welcome, brief introduction of yourself, and establish rapport
- Explain the purpose of the interview and confidentiality
- Begin with broad, open-ended questions before moving to more specific ones
- Use probing techniques to explore responses in depth (e.g., "Tell me more about that", "How did that make you feel?")
- Practice active listening and reflect back what you hear
- Ask follow-up questions based on the respondent's answers
- Avoid leading questions or imposing your own views
- Cover approximately {num_questions} main questions throughout the interview
- Explore unexpected insights that emerge during the conversation
- Use silence strategically to allow the respondent to elaborate
- Maintain a conversational, non-judgmental tone
- End with a summative question and thank the respondent

Ensure your questions are thought-provoking and designed to elicit rich, detailed responses.
Adapt your questioning strategy based on how the interview unfolds.
Your output should be ONLY your dialogue as the interviewer.""",
            model="o3-mini" 
        )
        self.topic = topic
        self.respondent_profile = respondent_profile
        self.num_questions = num_questions

class RespondentAgent(Agent):
    """
    Represents the interview respondent, answering based on their assigned persona.
    """
    def __init__(self, persona_data: Dict[str, Any], topic: str):
        # --- Create a highly detailed and structured persona description ---
        # Safely extract data, ensuring nested structures are dicts
        name = persona_data.get("name", "Unknown")
        age = persona_data.get("age", "N/A")
        gender = persona_data.get("gender", "N/A")
        occupation = persona_data.get("occupation", "N/A")
        education = persona_data.get("education", "N/A")
        location = persona_data.get("location", "N/A")
        income_bracket = persona_data.get("income_bracket", "N/A")

        demographics = persona_data.get("demographics", {}) if isinstance(persona_data.get("demographics"), dict) else {}
        psychographics = persona_data.get("psychographics", {}) if isinstance(persona_data.get("psychographics"), dict) else {}
        behaviors = persona_data.get("behaviors", {}) if isinstance(persona_data.get("behaviors"), dict) else {}
        attitudes = persona_data.get("attitudes", {}) if isinstance(persona_data.get("attitudes"), dict) else {}

        # Extract nested fields safely
        ethnicity = demographics.get("ethnicity", "N/A")
        marital_status = demographics.get("marital_status", "N/A")
        household = demographics.get("household_composition", "N/A")
        values = psychographics.get("values", [])
        interests = psychographics.get("interests", [])
        lifestyle = psychographics.get("lifestyle_details", "N/A")
        personality = psychographics.get("personality_traits", [])
        consumption = behaviors.get("consumption_patterns", "N/A")
        usage = behaviors.get("usage_habits", "N/A")
        opinions = attitudes.get("opinions", "N/A")
        beliefs = attitudes.get("beliefs", "N/A")
        sentiment_topic = attitudes.get("sentiments_towards_topic", "Neutral")

        media_consumption = persona_data.get("media_consumption", [])
        motivations = persona_data.get("motivations", "N/A")
        challenges = persona_data.get("challenges", "N/A")
        topic_experience = persona_data.get("topic_experience", "N/A")
        brand_affinities = persona_data.get("brand_affinities", [])
        communication_style = persona_data.get("communication_style", "natural and authentic")

        # Format lists cleanly
        values_str = ', '.join(values) if values else 'N/A'
        interests_str = ', '.join(interests) if interests else 'N/A'
        personality_str = ', '.join(personality) if personality else 'N/A'
        media_str = ', '.join(media_consumption) if media_consumption else 'N/A'
        brands_str = ', '.join(brand_affinities) if brand_affinities else 'N/A'

        persona_description = f"""
--- YOUR DETAILED PERSONA PROFILE ---
**Identity:** {name}, {age}, {gender}
**Background:** Living in {location}, Occupation: {occupation}, Education: {education}, Income Bracket: {income_bracket}
**Demographics:** Ethnicity: {ethnicity}, Marital Status: {marital_status}, Household: {household}
**Psychographics:**
    - Personality: {personality_str}
    - Values: {values_str}
    - Interests/Hobbies: {interests_str}
    - Lifestyle Notes: {lifestyle}
**Behaviors:**
    - Consumption Patterns (Topic Relevant): {consumption}
    - Usage Habits (Topic Relevant): {usage}
**Attitudes & Beliefs (Regarding '{topic}'):**
    - Overall Sentiment: {sentiment_topic}
    - Specific Opinions: {opinions}
    - Underlying Beliefs: {beliefs}
**Other Key Info:**
    - Motivations: {motivations}
    - Challenges/Pain Points: {challenges}
    - Prior Experience with '{topic}': {topic_experience}
    - Media Habits: {media_str}
    - Brand Affinities: {brands_str}
    - Typical Communication Style: **{communication_style}**
---

**CRITICAL INSTRUCTIONS: You MUST embody this persona COMPLETELY and CONSISTENTLY.**

**BEHAVIORAL MANDATES:**

1.  **Full Embodiment (MANDATORY):** Respond *exclusively* from the perspective of **{name}**. Your answers MUST reflect the combined influence of your *entire* profile: age, occupation, income, location, education, values, interests, personality, lifestyle, behaviors, attitudes, motivations, challenges, experiences, etc. **DO NOT give generic AI answers.**
2.  **Integrate Specific Details (MANDATORY):** Actively weave details from your profile into your responses naturally. Don't just state opinions; explain *why* **{name}** holds them, linking back to specific profile elements. Examples:
    *   Discussing cost? "Well, given my income bracket as a {occupation}, I have to be mindful..."
    *   Discussing features? "As someone interested in {interests_str}, I find that..." or "My {personality_str} side makes me appreciate..."
    *   Discussing ease of use? "With my {challenges}, simplicity is really key..." or "Based on my experience with {topic_experience}, I..."
3.  **Show, Don't Just Tell:** Express your persona through your language and reasoning. Reflect your `communication_style` in your tone, vocabulary, and sentence structure.
4.  **Maintain Deep Consistency:** Ensure your views remain coherent with your *entire* profile. If your profile has conflicting elements (e.g., values vs. behavior), portray that internal conflict realistically. Your `sentiment_towards_topic` should guide your overall tone.
5.  **Realistic Nuance & Emotion:** Provide detailed, thoughtful answers appropriate for your persona. Express emotions (excitement, frustration, indifference, etc.) authentically when discussing relevant points, guided by your profile (values, challenges, experiences). Use natural language, including occasional pauses or fillers ("umm," "well," "you know...") sparingly, fitting your `communication_style`.
6.  **Authentic Engagement:** Listen to the interviewer. Respond directly to their questions. Acknowledge or react to their probes and reflections based on how **{name}** would perceive them. If asked about something outside your direct experience, relate it to something similar or state that honestly based on your profile.
7.  **Output Format:** Your response must be *ONLY* your dialogue as **{name}**. No extra text, labels, or explanations.

Think: How would **{name}**, with all their specific characteristics and experiences, genuinely react and contribute to *this specific point* in the conversation?"""

        super().__init__(
            name=name, # Use actual name for the agent instance
            instructions=persona_description, # Pass the detailed prompt here
            model="o3-mini" 
        )
        self.topic = topic
        self.persona_data = persona_data # Store for potential later use

class AnalystAgent(Agent):
    """
    Analyzes the interview transcript to extract key insights, themes, sentiment,
    and provide a structured, professional report.
    """
    def __init__(self, topic: str, target_audience: str):
        super().__init__(
            name="IDI Analyst",
            instructions=f"""You are a senior qualitative research analyst with deep expertise in analyzing in-depth interview (IDI) transcripts.
You have been provided with an IDI transcript on '{topic}' with a respondent from the target audience: '{target_audience}'.
**CRITICAL TASK:** Perform a rigorous, data-driven analysis of this transcript. Your goal is to identify accurate insights based *only* on the provided text.

**ANALYSIS REQUIREMENTS:**
1.  **Sentiment Analysis (Transcript-Wide & Per Turn):**
    *   Analyze the respondent's dialogue turn-by-turn. Classify the sentiment (Positive, Neutral, Negative) expressed in *each response* concerning the topic being discussed at that point.
    *   **Calculate** the overall sentiment distribution (percentage Positive, Neutral, Negative) for the *entire interview* by aggregating the turn-by-turn classifications. Base this calculation strictly on the respondent's utterances in the transcript.
2.  **Thematic Analysis:** Identify the major themes discussed by the respondent regarding '{topic}'. For each theme:
    *   Describe the theme clearly.
    *   Estimate its prominence (e.g., discussed frequently, mentioned briefly).
    *   Provide 1-2 direct, representative quotes from the respondent *exactly* as they appear in the transcript to illustrate the theme.
    *   Note the dominant sentiment associated with the theme based on your turn-by-turn analysis.
3.  **Insight Extraction:** Synthesize findings from themes and sentiment analysis to uncover key insights about the respondent's perspective, motivations, pain points, and attitudes related to '{topic}'.
4.  **Response Pattern Analysis:** Briefly comment on the respondent's communication style as observed in the transcript (e.g., detailed, concise, hesitant, confident), referencing their persona's `communication_style` if provided. Note any significant non-verbal cues implied (e.g., "seemed enthusiastic when discussing X").

**REPORT STRUCTURE:**
Generate a professional report with the following sections:
1.  **Executive Summary:** Concise overview of the most critical findings, focusing on the respondent's core attitudes, motivations, and key takeaways regarding '{topic}' (approx. 150 words). Include the calculated overall sentiment breakdown.
2.  **Research Background:** Briefly restate the study topic and target audience.
3.  **Methodology:** Note it was a simulated 1-on-1 in-depth interview.
4.  **Respondent Profile Summary:** Briefly summarize the key characteristics of the interviewed persona provided to you.
5.  **Key Themes:** Detail the identified themes with descriptions, illustrative quotes (exact wording), and associated sentiment.
6.  **Sentiment Analysis Findings:** Report the **calculated** overall sentiment distribution (%). Discuss how sentiment evolved or related to specific sub-topics based on your turn-by-turn analysis.
7.  **Response Patterns:** Briefly describe the observed communication style and any implied non-verbal cues.
8.  **Actionable Insights & Recommendations:** Translate the findings into 2-3 strategic insights or potential actions relevant to '{topic}'.
9.  **Limitations:** Acknowledge the limitations of a single simulated interview.
10. **Suggestions for Further Research:** Recommend 1-2 follow-up questions or areas.

**MANDATORY JSON OUTPUT:**
AFTER your narrative analysis, include a JSON-formatted section containing structured data derived *directly and accurately* from YOUR analysis of the transcript. **DO NOT use placeholder values. Calculate these values based on the transcript.**

```json
{{
  "sentiment_breakdown": {{ // CALCULATED overall sentiment distribution for the respondent
    "positive": <calculated_positive_percentage>, // e.g., 45.5
    "neutral": <calculated_neutral_percentage>, // e.g., 30.0
    "negative": <calculated_negative_percentage> // e.g., 24.5
  }},
  "key_themes": [ // Identified themes from your analysis
    // Example: {{"theme": "Cost Concerns", "prominence": "High", "sentiment": "Negative", "description": "Respondent frequently expressed concerns about the affordability...", "example_quote": "..."}},
    // Add 3-5 key themes identified and analyzed
  ],
  "response_metrics": {{ // CALCULATED metrics based on transcript
    "avg_response_length_words": <calculated_avg_respondent_response_word_count>,
    "total_respondent_word_count": <calculated_total_respondent_word_count>,
    "estimated_hesitation_markers": <count_of_markers_like_um_uh> // Count actual markers if possible
  }},
  "sentiment_by_turn": [ // Your calculated sentiment for each respondent turn
    // Example: {{"turn_number": 1, "sentiment": "Neutral", "topic_discussed": "Initial greeting/topic intro"}},
    // Example: {{"turn_number": 2, "sentiment": "Positive", "topic_discussed": "Positive experience with feature X"}},
    // Add entry for EACH respondent turn analyzed
  ]
}}
```

Adhere strictly to these instructions. Calculation accuracy and fidelity to the transcript are paramount.""",
            model="gpt-4.1" 
        )
        self.topic = topic
        self.target_audience = target_audience


# --- Core Functions ---

async def generate_respondent_persona(target_audience: str, topic: str, simulation_id: str) -> Dict[str, Any]:
    """
    Generates a detailed respondent persona using gpt-4.1 and validates the structure.
    Saves the persona to the simulation-specific directory.
    """
    persona_generator_agent = Agent(
        name="Persona Generator",
        instructions=f"""You are an expert research methodologist specializing in qualitative research recruitment and persona development.
Based on the target audience description: '{target_audience}', generate a single, highly detailed, realistic, and internally coherent respondent persona suitable for an in-depth interview on the topic: '{topic}'.

**CRITICAL: Adhere STRICTLY to the MANDATORY JSON STRUCTURE below. Pay meticulous attention to nested objects (MUST be actual JSON objects, not strings) and data types. Ensure depth and realism.**

**MANDATORY JSON STRUCTURE:** The persona MUST be a SINGLE JSON object with the exact fields specified below.

```json
{{
  "name": "String: Full name (e.g., 'Alex Chen', 'Maria Garcia')",
  "age": "Number: Numeric age (e.g., 38)",
  "gender": "String: Gender identity (e.g., 'Female', 'Male', 'Non-binary')",
  "occupation": "String: Specific current job title/professional role (e.g., 'Senior Software Engineer', 'Registered Nurse', 'Freelance Graphic Designer')",
  "education": "String: Highest educational attainment (e.g., 'Master's Degree in Public Health', 'Bachelor of Arts in History')",
  "location": "String: Reasonably specific location (e.g., 'Chicago, IL, USA', 'Suburban area near London, UK')",
  "income_bracket": "String: Income level category or range (e.g., '$70k-$90k USD', '£40k-£55k GBP', 'Upper-middle income')",
  "demographics": {{ // Nested JSON Object - REQUIRED
    "ethnicity": "String: e.g., 'East Asian', 'White European', 'Black/African Descent', 'Hispanic/Latino'",
    "marital_status": "String: e.g., 'Married', 'Single', 'Divorced', 'Partnered'",
    "household_composition": "String: e.g., 'Living with partner and one young child', 'Living alone with a pet', 'Multigenerational household'",
    "other_relevant_demographics": "String: Any other key demographic detail (e.g., 'First-generation immigrant', 'Military veteran', or null)"
  }},
  "psychographics": {{ // Nested JSON Object - REQUIRED
    "values": ["String", "String", "..."], // Array of 2-4 core personal values (e.g., "Community involvement", "Personal growth", "Financial security")
    "interests": ["String", "String", "..."], // Array of 3-5 specific interests/hobbies (e.g., "Playing competitive chess", "Urban gardening", "Following indie music scene")
    "lifestyle_details": "String: Description of daily life/routine/activities (e.g., 'Commutes by bike, active in local environmental groups, enjoys cooking at home')",
    "personality_traits": ["String", "String", "..."] // Array of 3-5 descriptive traits (e.g., "Expressive", "Collaborative", "Cautious", "Optimistic")
  }},
  "behaviors": {{ // Nested JSON Object - REQUIRED
    "consumption_patterns": "String: Behaviors related to the topic (e.g., 'Rarely uses mobile banking, prefers desktop', 'Shops for sustainable products weekly')",
    "usage_habits": "String: Specific habits related to the topic/technology (e.g., 'Follows sustainability influencers online', 'Rarely considers environmental impact when shopping')",
    "other_relevant_behaviors": "String: Other related behaviors (e.g., 'Actively participates in online forums about [topic]', or null)"
  }},
  "attitudes": {{ // Nested JSON Object - REQUIRED
    "opinions": "String: Specific opinions on the interview topic '{topic}' (e.g., 'Feels sustainable footwear lacks style options', 'Thinks brands greenwash too much')",
    "beliefs": "String: Underlying beliefs related to the topic (e.g., 'Believes individual actions matter for sustainability', 'Thinks convenience often outweighs ethical concerns')",
    "sentiments_towards_topic": "String: Overall feeling about the topic (e.g., 'Passionate Advocate', 'Interested but Confused', 'Slightly Skeptical', 'Apathetic')"
  }},
  "media_consumption": ["String", "String", "..."], // Array of 2-4 specific media sources/types (e.g., "Instagram", "Specific environmental blogs", "BBC News")
  "motivations": "String: Primary drivers or goals in life/work (e.g., 'Expressing personal style authentically', 'Reducing personal environmental footprint', 'Finding affordable solutions')",
  "challenges": "String: Key current challenges or pain points (e.g., 'Finding trustworthy information about brand sustainability', 'Balancing budget with ethical purchasing', 'Lack of time for research')",
  "topic_experience": "String: Specific past experiences related to '{topic}' (e.g., 'Used competitor App X for 2 years, switched due to fees', 'Tried sustainable brand Y, found quality poor', 'No direct experience but researched extensively')",
  "brand_affinities": ["String", "String", "..."], // Array of 1-3 specific brands they like or use (relevant if possible, e.g., "Allbirds", "Zara", "Nike")
  "communication_style": "String: How they tend to communicate (e.g., 'Thoughtful and detailed', 'Concise and direct', 'Warm and conversational', 'Slightly hesitant initially', 'Data-driven and analytical')"
}}
```

**Output Requirement:** Return your response ONLY as a single, valid JSON object adhering strictly to the structure defined above. Ensure all specified fields, including nested objects and their sub-fields, are present and correctly typed. NO commentary before or after the JSON object.""",
        model="gpt-4.1" # Use gpt-4.1
    )

    print(f"STREAM: Generating respondent persona for topic '{topic}' using gpt-4.1...", flush=True)
    result = await Runner.run(persona_generator_agent, f"Generate a detailed persona JSON object for an interview on '{topic}' for target audience '{target_audience}'.")
    response_text = result.final_output
    
    # --- Define output directory ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "idi", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    persona_file_path = os.path.join(base_output_dir, "persona.json")
    # --- End Define output directory ---
    
    persona = None
    try:
        # Extract JSON from agent output - properly indented
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            persona_json = json_match.group(1).strip()
        else:
            persona_json = response_text.strip()
            if not (persona_json.startswith('{') and persona_json.endswith('}')):
                raise ValueError("Response does not appear to be a valid JSON object.")

        persona = json.loads(persona_json)

        # --- START Data Validation ---
        if not isinstance(persona, dict):
            raise ValueError("Parsed JSON is not a dictionary.")

        required_keys = [ # From mandatory structure
            "name", "age", "gender", "occupation", "education", "location", "income_bracket",
            "demographics", "psychographics", "behaviors", "attitudes",
            "media_consumption", "motivations", "challenges", "topic_experience",
            "brand_affinities", "communication_style"
        ]
        missing_keys = [key for key in required_keys if key not in persona]
        if missing_keys:
            raise ValueError(f"Persona missing required top-level keys: {missing_keys}")

        # Validate nested objects
        required_nested_keys = {
            "demographics": ["ethnicity", "marital_status", "household_composition"],
            "psychographics": ["values", "interests", "lifestyle_details", "personality_traits"],
            "behaviors": ["consumption_patterns", "usage_habits"],
            "attitudes": ["opinions", "beliefs", "sentiments_towards_topic"]
        }
        for nested_key, sub_keys in required_nested_keys.items():
             if not isinstance(persona.get(nested_key), dict):
                 raise ValueError(f"Persona key '{nested_key}' is not a JSON object.")
             missing_sub_keys = [sk for sk in sub_keys if sk not in persona[nested_key]]
             if missing_sub_keys:
                  raise ValueError(f"Persona nested object '{nested_key}' missing required keys: {missing_sub_keys}")

        # Validate specific types (basic examples)
        if not isinstance(persona["age"], (int, float)): raise ValueError("'age' must be a number.")
        if not isinstance(persona["demographics"], dict): raise ValueError("'demographics' must be an object.")
        if not isinstance(persona["psychographics"]["values"], list): raise ValueError("'psychographics.values' must be an array.")
        if not isinstance(persona["media_consumption"], list): raise ValueError("'media_consumption' must be an array.")

        print("STREAM: Persona JSON validation passed.", flush=True)
        # --- END Data Validation ---
        
        # --- Save persona to file ---
        with open(persona_file_path, "w") as f:
            json.dump(persona, f, indent=2)
        print(f"STREAM: Generated detailed persona saved to {persona_file_path}", flush=True)
        return persona

    except Exception as e:
        # Exception handling - properly indented
        print(f"Error: {e}")
        print(f"STREAM: Raw response snippet: {response_text[:500]}...", flush=True)
        print("STREAM: Creating and saving generic persona as fallback.", flush=True)
        # Create generic persona if parsing fails
        generic_persona = {
            "name": "Generic Person", "age": 35, "gender": "Unknown", "occupation": "Professional",
            "education": "N/A", "location": "N/A", "income_bracket": "N/A",
            "demographics": {"ethnicity": "N/A", "marital_status": "N/A", "household_composition": "N/A", "other_relevant_demographics": None},
            "psychographics": {"values": [], "interests": [], "lifestyle_details": "N/A", "personality_traits": ["Neutral"]},
            "behaviors": {"consumption_patterns": "N/A", "usage_habits": "N/A", "other_relevant_behaviors": None},
            "attitudes": {"opinions": "Neutral opinion", "beliefs": "Neutral beliefs", "sentiments_towards_topic": "Neutral"},
            "media_consumption": [], "motivations": "N/A", "challenges": "N/A",
            "topic_experience": "Some familiarity", "brand_affinities": [], "communication_style": "Moderate"
        }
        # --- Save generic persona ---
        try:
            with open(persona_file_path, "w") as f:
                json.dump(generic_persona, f, indent=2)
            print(f"STREAM: Saved generic persona to {persona_file_path}", flush=True)
        except Exception as save_e:
            print(f"STREAM: CRITICAL ERROR - Could not save fallback persona: {save_e}", flush=True)
        return generic_persona


async def run_interview(interviewer: InterviewerAgent, respondent: RespondentAgent, num_questions: int) -> List[Tuple[str, str]]:
    """Runs the in-depth interview simulation with names in stream."""
    print("\n--- Starting In-Depth Interview Simulation ---")
    transcript = []
    # Use agent names directly now
    interviewer_name = interviewer.name # Should be "IDI Interviewer" or similar
    respondent_name = respondent.name # Should be the generated persona name
    
    current_context = f"The interview topic is: {interviewer.topic}\nInterviewer: {interviewer_name}\nRespondent: {respondent_name} ({respondent.persona_data.get('age')} y/o {respondent.persona_data.get('occupation')})"
    
    question_count = 0
    
    # Introduction phase
    print(f"{interviewer_name} is preparing introduction...") # Use name
    intro_prompt = f"""Current context:
{current_context}

Start the interview. Introduce yourself ({interviewer_name}), explain the purpose about '{interviewer.topic}', establish rapport with {respondent_name}, and ask your first question. Make the respondent feel comfortable."""
    
    intro_result = await Runner.run(interviewer, intro_prompt)
    intro_dialogue = intro_result.final_output
    print(f"STREAM: {interviewer_name}: {intro_dialogue}", flush=True) # Use name
    transcript.append((interviewer_name, intro_dialogue))
    current_context += f"""\n{interviewer_name}: {intro_dialogue}"""

    print(f"{respondent_name} is thinking...") # Use name
    respondent_prompt = f"""Current context:
{current_context}

You are {respondent_name}. Respond to the interviewer's introduction and first question based *strictly* on your detailed persona profile and communication style provided in your instructions. Be authentic."""
    
    resp_result = await Runner.run(respondent, respondent_prompt)
    respondent_dialogue = resp_result.final_output
    print(f"STREAM: {respondent_name}: {respondent_dialogue}", flush=True) # Use name
    transcript.append((respondent_name, respondent_dialogue))
    current_context += f"""\n{respondent_name}: {respondent_dialogue}"""
    
    question_count += 1
    
    # Main interview loop
    while question_count < num_questions:
        print(f"\n--- Question {question_count + 1} of {num_questions} ---")
        
        # Interviewer's turn
        print(f"{interviewer_name} is thinking...") # Use name
        interviewer_prompt = f"""Current context:
{current_context}

You are {interviewer_name}. Ask your next question or make a follow-up comment to {respondent_name}. Use probing techniques, active listening, and aim to explore the topic '{interviewer.topic}' deeply based on their previous responses."""
        
        int_result = await Runner.run(interviewer, interviewer_prompt)
        interviewer_dialogue = int_result.final_output
        print(f"STREAM: {interviewer_name}: {interviewer_dialogue}", flush=True) # Use name
        transcript.append((interviewer_name, interviewer_dialogue))
        current_context += f"""\n{interviewer_name}: {interviewer_dialogue}"""
        
        # Respondent's turn
        print(f"{respondent_name} is thinking...") # Use name
        respondent_prompt = f"""Current context:
{current_context}

You are {respondent_name}. Respond to the interviewer's latest question/comment based *strictly* on your detailed persona profile (including your background, values, experiences, attitudes, motivations, challenges, communication style etc.). Provide a thoughtful, authentic, and consistent response."""
        
        resp_result = await Runner.run(respondent, respondent_prompt)
        respondent_dialogue = resp_result.final_output
        print(f"STREAM: {respondent_name}: {respondent_dialogue}", flush=True) # Use name
        transcript.append((respondent_name, respondent_dialogue))
        current_context += f"""\n{respondent_name}: {respondent_dialogue}"""
        
        question_count += 1
        await asyncio.sleep(0.2) # Shorter delay okay?

    
    # Conclusion phase
    print("\n--- Concluding Interview ---")
    print(f"{interviewer_name} is thinking...") # Use name
    conclusion_prompt = f"""Current context:
{current_context}

This is the final part of the interview. Ask {respondent_name} a concluding question to summarize their views on '{interviewer.topic}' or get final thoughts. Then, thank them sincerely for their time and input."""
    
    concl_result = await Runner.run(interviewer, conclusion_prompt)
    conclusion_dialogue = concl_result.final_output
    print(f"STREAM: {interviewer_name}: {conclusion_dialogue}", flush=True) # Use name
    transcript.append((interviewer_name, conclusion_dialogue))
    current_context += f"""\n{interviewer_name}: {conclusion_dialogue}"""

    print(f"{respondent_name} is thinking...") # Use name
    final_prompt = f"""Current context:
{current_context}

You are {respondent_name}. This is your final response. Share any concluding thoughts on '{interviewer.topic}' based on your persona, and respond to the interviewer's thank you."""
    
    final_result = await Runner.run(respondent, final_prompt)
    final_dialogue = final_result.final_output
    print(f"STREAM: {respondent_name}: {final_dialogue}", flush=True) # Use name
    transcript.append((respondent_name, final_dialogue))
    
    print("\n--- Interview Finished ---")
    return transcript

async def analyze_interview(analyst: AnalystAgent, transcript: List[Tuple[str, str]], respondent_data: Dict[str, Any]) -> Tuple[str, Dict]:
    """Analyzes the interview transcript using the AnalystAgent, ensuring robust JSON extraction."""
    print("\n--- Analyzing Interview Transcript ---")
    
    # Calculate transcript metrics for fallback generation
    respondent_turn_counter = 0
    respondent_word_count = 0
    hesitation_markers = 0
    
    # Get interview participants and topic from analyst
    interviewer_name = "IDI Interviewer"
    respondent_name = respondent_data.get('name', 'Respondent')
    topic = analyst.topic
    target_audience = analyst.target_audience
    
    # Process transcript metrics
    formatted_transcript = []
    for speaker, dialogue in transcript:
        formatted_transcript.append(f"{speaker}: {dialogue}")
        
        # Calculate metrics if this is a respondent turn
        if respondent_name in speaker:
            respondent_turn_counter += 1
            words = dialogue.split()
            respondent_word_count += len(words)
            
            # Count hesitation markers (um, uh, hmm, well, etc.)
            hesitation_patterns = ['um', 'uh', 'hmm', 'er', 'ah', 'like', 'you know']
            for pattern in hesitation_patterns:
                hesitation_markers += sum(1 for word in words if word.lower() == pattern)
    
    full_transcript_str = "\n".join(formatted_transcript)
    
    # Create context on respondent for the analysis
    respondent_context = f"**Respondent Information:**\n```json\n{json.dumps(respondent_data, indent=2)}\n```\n---" # Format as JSON for clarity

    analysis_prompt = f"""Please perform a detailed analysis of the following in-depth interview transcript regarding '{topic}' with target audience '{target_audience}'. Follow your instructions meticulously. Calculate all metrics and sentiment distributions based *only* on the provided transcript text.

{respondent_context}

**TRANSCRIPT:**
```
{full_transcript_str}
```

**Output Requirements:**
1.  Provide your full narrative analysis report (Executive Summary, Themes, Sentiment, Insights, etc.).
2.  At the VERY END of your response, include the MANDATORY JSON block containing the accurately calculated metrics ('sentiment_breakdown', 'key_themes', 'response_metrics', 'sentiment_by_turn') as specified in your instructions. Ensure all values in the JSON are calculated *directly* from the transcript."""
    
    result = await Runner.run(analyst, analysis_prompt)
    analysis_text = result.final_output.strip()

    # --- Extract and Validate JSON block ---
    narrative_analysis = analysis_text # Default to full text
    analysis_json_parsed = None

    json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_text, re.DOTALL | re.IGNORECASE)

    if json_match:
        json_string = json_match.group(1).strip()
        try:
            analysis_json_parsed = json.loads(json_string)
            # Basic structural validation of the parsed JSON
            required_json_keys = ["sentiment_breakdown", "key_themes", "response_metrics", "sentiment_by_turn"]
            if all(key in analysis_json_parsed for key in required_json_keys):
                 # Remove JSON block from narrative
                 narrative_analysis = analysis_text.replace(json_match.group(0), '').strip()
                 print("STREAM: Extracted and validated analysis JSON block structure.", flush=True)
            else:
                 missing_keys = [key for key in required_json_keys if key not in analysis_json_parsed]
                 print(f"STREAM: Warning - Parsed analysis JSON is missing required keys: {missing_keys}. JSON block kept in narrative.", flush=True)
                 analysis_json_parsed = {"error": "JSON structure validation failed", "missing_keys": missing_keys}

        except json.JSONDecodeError as e:
            print(f"STREAM: Warning - Failed to parse analysis JSON: {e}. JSON block kept in narrative.", flush=True)
            analysis_json_parsed = {"error": f"JSON parsing failed: {e}", "raw_json_string": json_string}
    else:
        print("STREAM: Warning - Analysis JSON block not found in analyst output. Using defaults.", flush=True)

    # If JSON parsing/validation failed, create a default structure with calculated metrics
    if analysis_json_parsed is None or "error" in analysis_json_parsed:
         print("STREAM: Populating analysis JSON with calculated defaults due to extraction/validation issues.", flush=True)
         analysis_json_parsed = {
             "sentiment_breakdown": {"positive": 0, "neutral": 100, "negative": 0}, # Default neutral if not calculated
             "key_themes": [{"theme": "Analysis Incomplete", "prominence": 50, "sentiment": "Neutral", "description": "Analyst failed to extract themes.", "example_quote": ""}],
             "response_metrics": {
                 "avg_response_length_words": int(respondent_word_count / max(1, respondent_turn_counter)) if respondent_turn_counter > 0 else 0,
                 "total_respondent_word_count": respondent_word_count,
                 "estimated_hesitation_markers": hesitation_markers
             },
             "sentiment_by_turn": [{"turn_number": i+1, "sentiment": "Unknown", "topic_discussed": "Analysis Incomplete"} for i in range(respondent_turn_counter)]
         }

    print("STREAM: Analysis Complete.", flush=True)
    return narrative_analysis, analysis_json_parsed # Return parsed/default JSON


# ... existing code ...

def generate_report(analysis: str, parameters: Dict[str, Any], transcript: List[Tuple[str, str]], 
                   visualization_data: Dict[str, Any], respondent_data: Dict[str, Any], simulation_id: str):
    """Generates and saves the final report with visualizations to the simulation-specific directory."""
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- Generating Report for IDI Simulation ID: {simulation_id} ---")
    
    # --- Define Paths ---
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "idi", simulation_id)
    viz_dir = os.path.join(base_output_dir, "visualizations")
    viz_rel_path_prefix = f"/simulations/idi/{simulation_id}/visualizations" # Relative path for markdown links
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Output directory ensured: {base_output_dir}")
    # --- End Define Paths ---

    report_content = f"""# In-Depth Interview Report

## 1. Study Parameters
- **Topic:** {parameters['topic']}
- **Target Audience:** {parameters['target_audience']}
- **Number of Questions:** {parameters['num_questions']}
- **Simulation ID:** {simulation_id}
- **Date/Time:** {current_timestamp}

## 2. Respondent Profile
"""
    # Add respondent profile to the report
    report_content += f"### {respondent_data.get('name', 'Unknown')}\n"
    report_content += f"- **Age:** {respondent_data.get('age', 'Unknown')}\n"
    report_content += f"- **Occupation:** {respondent_data.get('occupation', 'Unknown')}\n"
    report_content += f"- **Education:** {respondent_data.get('education', 'Unknown')}\n"
    report_content += f"- **Location:** {respondent_data.get('location', 'Unknown')}\n"
    # Use json.dumps for potentially nested dicts
    report_content += f"- **Demographics:** {json.dumps(respondent_data.get('demographics', {}))}\n" 
    report_content += f"- **Psychographics:** {json.dumps(respondent_data.get('psychographics', {}))}\n"
    report_content += f"- **Relevant Behaviors:** {json.dumps(respondent_data.get('behaviors', {}))}\n"
    report_content += f"- **Relevant Attitudes:** {json.dumps(respondent_data.get('attitudes', {}))}\n"
    report_content += f"- **Topic Experience:** {respondent_data.get('topic_experience', 'N/A')}\n\n"
    
    report_content += f"""
## 3. Analysis Results
{analysis}

## 4. Visualizations
*See attached visualization files referenced below*

"""
    
    # Generate visualizations if data is available
    visualization_files = []
    if visualization_data:
        print(f"Generating visualizations in {viz_dir}...")
        
        # 1. Sentiment Breakdown Pie Chart
        if "sentiment_breakdown" in visualization_data:
            try: # Add try-except for robustness
                sentiment_data = visualization_data["sentiment_breakdown"]
                # Ensure data has values before plotting
                if sentiment_data and any(sentiment_data.values()):
                    plt.figure(figsize=(10, 7))
                    labels = list(sentiment_data.keys())
                    sizes = [float(v) for v in sentiment_data.values()] # Ensure float
                    colors = ['#66b3ff', '#99ff99', '#ff9999'] # Neutral, Positive, Negative order? Adjust if needed
                    # Find positive index for explode, handle missing keys
                    pos_index = labels.index('positive') if 'positive' in labels else -1
                    explode = tuple(0.1 if i == pos_index else 0 for i in range(len(labels))) 
            
                    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                            autopct='%1.1f%%', shadow=True, startangle=140)
                    plt.axis('equal') 
                    plt.title('Overall Sentiment Distribution', fontsize=16)
            
                    pie_file = os.path.join(viz_dir, "sentiment_pie_chart.png")
                    plt.savefig(pie_file, dpi=300)
                    plt.close()
                    visualization_files.append(os.path.basename(pie_file))
                    report_content += f"![Sentiment Distribution]({viz_rel_path_prefix}/sentiment_pie_chart.png)\n\n"
                else:
                    print("Skipping sentiment pie chart: No data or all values are zero.")
            except Exception as e:
                print(f"Error generating sentiment pie chart: {e}")
        
        # 2. Key Themes Bar Chart
        if "key_themes" in visualization_data and visualization_data["key_themes"]:
            try:
                themes_data = visualization_data["key_themes"]
                # Extract themes that have at least a theme name
                valid_themes = [item["theme"] for item in themes_data if "theme" in item]
                
                # Get metrics and sentiments with fallbacks
                metrics = []
                sentiments = []
                
                for item in themes_data:
                    if "theme" not in item:
                        continue
                        
                    # Get sentiment with fallback
                    sentiment = item.get("sentiment", "Neutral")
                    sentiments.append(sentiment)
                    
                    # Try different metric keys with fallbacks
                    if "frequency" in item and isinstance(item["frequency"], (int, float)):
                        metrics.append(item["frequency"])
                    elif "prominence" in item:
                        # Try to extract numeric value from prominence
                        prominence = item["prominence"]
                        if isinstance(prominence, (int, float)):
                            metrics.append(prominence)
                        else:
                            # Try to extract percentage
                            match = re.search(r'(\d+)%', str(prominence))
                            if match:
                                metrics.append(int(match.group(1)))
                            # Extract textual prominence levels
                            elif "high" in str(prominence).lower():
                                metrics.append(75)
                            elif "medium" in str(prominence).lower():
                                metrics.append(50)
                            elif "low" in str(prominence).lower():
                                metrics.append(25)
                            else:
                                metrics.append(50)  # Default value
                    else:
                        # Default value
                        metrics.append(50)
                
                # Only proceed if we have valid data
                if valid_themes and len(valid_themes) == len(metrics) == len(sentiments):
                    plt.figure(figsize=(12, 8))
                    
                    colors = []
                    for sentiment in sentiments:
                        s_lower = str(sentiment).lower()
                        if "positive" in s_lower: colors.append('#99ff99')
                        elif "negative" in s_lower: colors.append('#ff9999')
                        else: colors.append('#66b3ff') # Default to neutral
                    
                    bars = plt.bar(valid_themes, metrics, color=colors)
                    
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom')
            
                    plt.xlabel('Themes', fontsize=14)
                    plt.ylabel('Frequency/Prominence', fontsize=14)
                    plt.title('Key Themes with Sentiment', fontsize=16)
                    plt.xticks(rotation=45, ha='right', fontsize=12)
                    
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#99ff99', label='Positive'),
                        Patch(facecolor='#66b3ff', label='Neutral'),
                        Patch(facecolor='#ff9999', label='Negative')]
                    plt.legend(handles=legend_elements, title="Sentiment")
                    
                    plt.tight_layout()
                    
                    themes_file = os.path.join(viz_dir, "key_themes_chart.png")
                    plt.savefig(themes_file, dpi=300)
                    plt.close()
                    visualization_files.append(os.path.basename(themes_file))
                    report_content += f"![Key Themes]({viz_rel_path_prefix}/key_themes_chart.png)\n\n"
                else:
                    print(f"Skipping key themes chart: Mismatched data lengths - themes: {len(valid_themes)}, metrics: {len(metrics)}, sentiments: {len(sentiments)}")
            except Exception as e:
                print(f"Error generating key themes chart: {e}")
                import traceback
                traceback.print_exc() # More detailed error information
        
        # 3. Question Type Breakdown
        if "response_metrics" in visualization_data and "question_types" in visualization_data["response_metrics"]:
            try:
                question_types = visualization_data["response_metrics"]["question_types"]
                if question_types and any(question_types.values()):
                    plt.figure(figsize=(10, 7))
                    labels = list(question_types.keys())
                    sizes = [float(v) for v in question_types.values()]
                    colors = ['#ffcc99','#66b3ff','#99ff99'] # Open, Closed, Probing? Adjust as needed
                
                    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                    plt.title('Interviewer Question Type Distribution', fontsize=16)
                
                    question_file = os.path.join(viz_dir, "question_types.png")
                    plt.savefig(question_file, dpi=300)
                    plt.close()
                    visualization_files.append(os.path.basename(question_file))
                    report_content += f"![Question Types]({viz_rel_path_prefix}/question_types.png)\n\n"
                else:
                    print("Skipping question types chart: No data or all values zero.")
            except Exception as e:
                print(f"Error generating question types chart: {e}")
        
        # 4. Sentiment Flow Through Interview
        if "sentiment_by_turn" in visualization_data and visualization_data["sentiment_by_turn"]:
            try:
                sentiment_flow = visualization_data["sentiment_by_turn"]
                sentiment_flow = sorted(sentiment_flow, key=lambda x: x["turn_number"])
                
                q_numbers = [item["turn_number"] for item in sentiment_flow]
                sentiments = [item["sentiment"] for item in sentiment_flow]
                topics = [item.get("topic", f"Q{item['turn_number']}") for item in sentiment_flow] # Use topic if available
                
                sentiment_values = []
                for sentiment in sentiments:
                    s_lower = str(sentiment).lower()
                    if "positive" in s_lower: sentiment_values.append(1)
                    elif "negative" in s_lower: sentiment_values.append(-1)
                    else: sentiment_values.append(0) # Default neutral
                
                if q_numbers: # Only plot if there's data
                    plt.figure(figsize=(14, 8))
                    plt.plot(q_numbers, sentiment_values, marker='o', linestyle='-', linewidth=2, markersize=8)
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                
                    for i, topic in enumerate(topics):
                        plt.annotate(topic, (q_numbers[i], sentiment_values[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
                
                    plt.xlabel('Question Turn', fontsize=14)
                    plt.ylabel('Sentiment Score', fontsize=14)
                    plt.title('Sentiment Flow Through Interview', fontsize=16)
                    plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
                    plt.xticks(np.arange(min(q_numbers), max(q_numbers)+1, 1)) # Ensure integer ticks
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                
                    flow_file = os.path.join(viz_dir, "sentiment_flow.png")
                    plt.savefig(flow_file, dpi=300)
                    plt.close()
                    visualization_files.append(os.path.basename(flow_file))
                    report_content += f"![Sentiment Flow]({viz_rel_path_prefix}/sentiment_flow.png)\n\n"
                else:
                    print("Skipping sentiment flow chart: No question data.")
            except Exception as e:
                print(f"Error generating sentiment flow chart: {e}")
    
    report_content += """
## 5. Full Transcript
"""
    for speaker, dialogue in transcript:
        report_content += f"- **{speaker}:** {dialogue}\n\n"

    # --- Save Report and Transcript ---
    report_filename = os.path.join(base_output_dir, "report.md")
    with open(report_filename, "w", encoding='utf-8') as f:
        f.write(report_content)
    print(f"Report saved successfully as: {report_filename}")
    
    transcript_filename = os.path.join(base_output_dir, "transcript.csv")
    with open(transcript_filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Dialogue"])
        for speaker, dialogue in transcript:
            writer.writerow([speaker, dialogue])
    print(f"Transcript saved successfully as: {transcript_filename}")
    # --- End Save Report and Transcript ---
    
    if visualization_files:
        print(f"Generated {len(visualization_files)} visualization files in {viz_dir}")


# --- Main Execution ---

async def main():
    """Main function to run the in-depth interview simulation."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1) # Exit if key is missing

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a simulated in-depth interview.")
    parser.add_argument(
        "--topic", type=str, default="Mobile banking app user experience",
        help="The topic for the in-depth interview."
    )
    parser.add_argument(
        "--target_audience", type=str, default="Urban professionals aged 25-40",
        help="Description of the target audience/personas."
    )
    parser.add_argument(
        "--num_questions", type=int, default=8,
        help="Number of main questions to include in the interview."
    )
    # --- Added simulation_id argument ---
    parser.add_argument(
        "--simulation_id", type=str, required=True, 
        help="Unique ID for this simulation run."
    )
    # --- End Added argument ---
    args = parser.parse_args()

    params = {
        "topic": args.topic,
        "target_audience": args.target_audience,
        "num_questions": args.num_questions
    }
    simulation_id = args.simulation_id # Get the ID

    print("\n--- IDI Parameters ---")
    print(f"Simulation ID: {simulation_id}")
    print(f"Topic: {params['topic']}")
    print(f"Target Audience: {params['target_audience']}")
    print(f"Number of Questions: {params['num_questions']}")
    print("-----------------------------")

    # Pass simulation_id to persona generation
    persona = await generate_respondent_persona(params["target_audience"], params["topic"], simulation_id)
    
    # Create respondent summary for interviewer
    psychographics = persona.get('psychographics', {})
    # Ensure psychographics is a dict before attempting to stringify
    psychographics_brief = json.dumps(psychographics)[:150] if isinstance(psychographics, dict) else str(psychographics)[:150]
    persona_summary = f"{persona.get('name', 'Unknown')}, {persona.get('age', 'Unknown')} year old {persona.get('occupation', 'professional')}. {json.dumps(persona.get('demographics', {}))}. Psychographics snippet: {psychographics_brief}..."
    
    # Create agents
    interviewer = InterviewerAgent(
        topic=params["topic"], respondent_profile=persona_summary, num_questions=params["num_questions"]
    )
    respondent = RespondentAgent(persona_data=persona, topic=params["topic"])
    analyst = AnalystAgent(topic=params["topic"], target_audience=params["target_audience"])

    # Run the interview
    transcript = await run_interview(interviewer, respondent, params["num_questions"])

    if transcript:
        # Analyze the transcript
        analysis, visualization_data = await analyze_interview(analyst, transcript, persona)
        
        # Generate the report, passing simulation_id
        generate_report(analysis, params, transcript, visualization_data, persona, simulation_id)
    else:
        print("Interview did not produce a transcript. Analysis skipped.")

if __name__ == "__main__":
    asyncio.run(main()) 
