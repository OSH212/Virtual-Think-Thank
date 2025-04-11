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
        # Extract ALL relevant persona components for the prompt
        name = persona_data.get("name", "Unknown")
        age = persona_data.get("age", "Unknown")
        gender = persona_data.get("gender", "Unknown")
        occupation = persona_data.get("occupation", "Unknown")
        education = persona_data.get("education", "Unknown")
        location = persona_data.get("location", "Unknown")
        income_bracket = persona_data.get("income_bracket", "Unknown")

        # --- Robust Handling for Nested Objects ---
        demographics_raw = persona_data.get("demographics", {})
        demographics = demographics_raw if isinstance(demographics_raw, dict) else {}

        psychographics_raw = persona_data.get("psychographics", {})
        psychographics = psychographics_raw if isinstance(psychographics_raw, dict) else {}

        behaviors_raw = persona_data.get("behaviors", {})
        behaviors = behaviors_raw if isinstance(behaviors_raw, dict) else {}

        attitudes_raw = persona_data.get("attitudes", {})
        attitudes = attitudes_raw if isinstance(attitudes_raw, dict) else {}
        # --- End Robust Handling ---
        
        # Safely access nested fields using the guaranteed dicts
        ethnicity = demographics.get("ethnicity", "Unknown")
        marital_status = demographics.get("marital_status", "Unknown")
        values = psychographics.get("values", [])
        interests = psychographics.get("interests", [])
        consumption_patterns = behaviors.get("consumption_patterns", "Unknown")
        opinions = attitudes.get("opinions", "Unknown")
        beliefs = attitudes.get("beliefs", "Unknown")

        # Continue extracting other fields
        media_consumption = persona_data.get("media_consumption", [])
        motivations = persona_data.get("motivations", "Unknown")
        challenges = persona_data.get("challenges", "Unknown")
        topic_experience = persona_data.get("topic_experience", "Unknown")
        brand_affinities = persona_data.get("brand_affinities", [])
        communication_style = persona_data.get("communication_style", "natural and authentic")


        # Create a more comprehensive persona description for the prompt
        persona_description = f"""
--- YOUR PERSONA PROFILE ---
Name: {name}, Age: {age}, Gender: {gender}
Location: {location}
Occupation: {occupation}
Education: {education}
Income Bracket: {income_bracket}
Ethnicity: {ethnicity}, Marital Status: {marital_status}
---
Values: {', '.join(values) if isinstance(values, list) and values else 'N/A'}
Interests: {', '.join(interests) if isinstance(interests, list) and interests else 'N/A'}
---
Attitudes/Opinions/Beliefs related to '{topic}': {opinions}; {beliefs}
Relevant Behaviors: {consumption_patterns}
Experience with '{topic}': {topic_experience}
---
Motivations: {motivations}
Challenges/Pain Points: {challenges}
Media Habits: {', '.join(media_consumption) if isinstance(media_consumption, list) and media_consumption else 'N/A'}
Brand Affinities: {', '.join(brand_affinities) if isinstance(brand_affinities, list) and brand_affinities else 'N/A'}
Typical Communication Style: {communication_style}
---
"""

        super().__init__(
            name="Respondent",
            instructions=f"""You are being interviewed in an in-depth interview (IDI) about: '{topic}'.
Your assigned persona profile is detailed below. Your primary goal is to consistently and authentically portray this *entire* persona throughout the interview.

{persona_description}

**BEHAVIORAL MANDATES:**

1.  **Embody Fully:** Respond *exclusively* from the perspective of this persona. Do NOT act as a generic AI or respondent.
2.  **Integrate Details:** In your responses, actively draw upon your persona's *specific* motivations, challenges, background (occupation, location, demographics, income, education), values, interests, topic experience, and even brand affinities or media habits when relevant to the interviewer's questions. Don't just state opinions; explain *why* your persona holds them based on their life and experiences.
3.  **Show, Don't Just Tell:** Instead of saying "As a [occupation], I think...", try phrasing responses naturally reflecting that background. For example, discussing cost? Relate it to your income bracket or financial challenges. Discussing convenience? Relate it to your lifestyle or challenges. Discussing past experiences? Refer to your specific `topic_experience`.
4.  **Maintain Consistency:** Keep your opinions, attitudes, and behaviors consistent with the profile throughout the discussion. If the profile indicates ambivalence or conflicting views, portray that nuance realistically.
5.  **Realistic Nuance & Depth:** Provide detailed, thoughtful responses. Use language, tone, and sentence structure appropriate for your persona's background and `communication_style`. Include occasional natural speech patterns (pauses, "umm," etc.) but don't overdo it. Express emotions naturally when discussing relevant experiences or strong opinions. Be conversational.
6.  **Authentic Engagement:** Listen to the interviewer's questions carefully. If a question touches on something sensitive for your persona (based on challenges, values, etc.), show appropriate hesitation or thoughtfulness. If you lack direct experience, relate it to something similar or state that honestly.
7.  **Output Format:** Your response must be *ONLY* your dialogue as this respondent. Do not include any other text, labels, or explanations.

Think: How would *this specific person* (with all their listed traits and experiences) react and respond thoughtfully and in detail to the interviewer's question? """,
            model="o3-mini" 
        )
        self.topic = topic
        self.persona_data = persona_data

class AnalystAgent(Agent):
    """
    Analyzes the interview transcript to extract key insights, themes, sentiment,
    and provide a structured, professional report.
    """
    def __init__(self, topic: str, target_audience: str):
        super().__init__(
            name="IDI Analyst",
            instructions=f"""You are a senior qualitative research analyst with expertise in analyzing in-depth interviews.
You have been provided with a transcript of an in-depth interview on '{topic}' with a respondent from this target audience: '{target_audience}'.
Your task is to perform a deep analysis of this transcript and generate a comprehensive report that meets professional standards.

**Perform a thorough sentiment analysis of the entire transcript.** Calculate the overall distribution of positive, neutral, and negative sentiment expressed by the respondent regarding the main topic and related sub-topics discussed. Identify key themes and associate them with the dominant sentiment expressed when discussing them.

The report should include:
1.  **Executive Summary:** A concise overview of the key findings (approx. 150 words).
2.  **Research Background:** Brief context on the study purpose, topic, and target audience.
3.  **Methodology:** Brief description of the in-depth interview approach.
4.  **Respondent Profile:** Summary of the respondent's persona and relevance to the target audience.
5.  **Key Themes:** Identify and elaborate on the major themes, ideas, and opinions expressed.
    - Provide theme frequency/prominence based on your analysis.
    - Include representative direct quotes for each theme.
    - Note areas of particular conviction or ambivalence.
6.  **Sentiment Analysis:** Assess the overall sentiment towards the topic and specific aspects discussed, based *directly on your analysis of the transcript*.
    - Quantify the calculated sentiment distribution (e.g., percentage positive, negative, neutral).
    - Identify emotional responses and their triggers.
    - Note any shifts in sentiment during the interview.
7.  **Response Patterns:** Analyze how the respondent structured their thoughts, notable linguistic patterns, and non-verbal cues implied in the text (hesitations, enthusiasm, etc.).
8.  **Actionable Insights & Recommendations:** Translate the findings into strategic insights and actionable recommendations relevant to the interview topic.
9.  **Potential Biases/Limitations:** Acknowledge any potential biases observed or limitations of the simulation.
10. **Suggestions for Further Research:** Recommend additional areas or questions to explore based on the findings.

Structure your output clearly using markdown headings. Ensure the analysis is objective, insightful, and presented professionally, as expected from a top-tier research firm.

AFTER YOUR ANALYSIS, include a JSON-formatted section with structured data derived *from your analysis* for visualization:
```json
{{
  "sentiment_breakdown": {{
    "positive": <calculated_positive_percentage>,
    "neutral": <calculated_neutral_percentage>,
    "negative": <calculated_negative_percentage>
  }},
  "key_themes": [
    {{"theme": "<Identified Theme 1>", "frequency": <calculated_frequency_1>, "sentiment": "<dominant_sentiment_for_theme_1>", "description": "<Brief theme description>"}},
    {{"theme": "<Identified Theme 2>", "frequency": <calculated_frequency_2>, "sentiment": "<dominant_sentiment_for_theme_2>", "description": "<Brief theme description>"}}
    // Add more themes as identified
  ],
  "response_metrics": {{
    "avg_response_length": <calculated_avg_response_word_count>,
    "total_word_count": <calculated_total_respondent_word_count>,
    "unique_words": <estimated_unique_words>,
    "hesitation_count": <estimated_hesitation_count>,
    "question_types": {{ // Based on interviewer questions in transcript
      "open_ended": <count_open_ended>,
      "closed": <count_closed>,
      "probing": <count_probing>
    }}
  }},
  "sentiment_by_question": [ // Map sentiment to interviewer question turn
    {{"question_number": 1, "sentiment": "<sentiment_following_q1>", "topic": "<topic_of_q1>"}},
    {{"question_number": 2, "sentiment": "<sentiment_following_q2>", "topic": "<topic_of_q2>"}}
    // Add entry for each major question/response pair analyzed
  ]
}}
```

**Crucially, ensure the values in the JSON section accurately reflect the results of YOUR analysis of the provided transcript content. Do not use the placeholder values literally.** This JSON should be placed at the end of your analysis, clearly separated from the narrative report.""",
            model="o3-mini" 
        )
        self.topic = topic
        self.target_audience = target_audience


# --- Core Functions ---

async def generate_respondent_persona(target_audience: str, topic: str, simulation_id: str) -> Dict[str, Any]:
    """
    Generates a detailed respondent persona as structured data for the interview.
    Saves the persona to the simulation-specific directory.
    """
    persona_generator_agent = Agent(
        name="Persona Generator",
        instructions=f"""You are an expert research methodologist specializing in qualitative research recruitment and persona development.
Based on the target audience description: '{target_audience}', generate a single, detailed respondent persona suitable for an in-depth interview on the topic: '{topic}'.

The persona should:
1. Represent a realistic individual within the target audience
2. Have detailed demographic characteristics (name, age, location, etc.)
3. Include comprehensive psychographic details (values, beliefs, interests, etc.)
4. Have relevant behaviors and attitudes toward the topic
5. Include potential biases or predispositions
6. Have a backstory that makes them interesting and authentic for an interview
7. Be nuanced - avoid creating a one-dimensional or stereotypical character

**MANDATORY JSON STRUCTURE:** The persona MUST be a JSON object with the exact fields specified below. Fields like `demographics`, `psychographics`, `behaviors`, and `attitudes` MUST be **nested JSON objects** containing their respective sub-fields as described. DO NOT use simple strings for these nested objects.

**Persona Object Fields:**
- name: Full name (string)
- age: Numeric age (number)
- gender: Gender identity (string)
- occupation: Current job/professional role (string)
- education: Highest educational attainment (string)
- location: City/region/country (string)
- income_bracket: Income level category (e.g., "Low", "Middle", "High") or specific range (string)
- **demographics**: (JSON Object) Contains:
    - ethnicity: (string)
    - marital_status: (string)
    - household_composition: (string, e.g., "Living alone", "With partner and child")
    - other_relevant_demographics: (string, optional)
- **psychographics**: (JSON Object) Contains:
    - values: (Array of strings)
    - interests: (Array of strings)
    - lifestyle_details: (string)
    - personality_traits: (Array of strings)
- **behaviors**: (JSON Object) Contains:
    - consumption_patterns: (string related to topic)
    - usage_habits: (string related to topic/technology)
    - other_relevant_behaviors: (string, optional)
- **attitudes**: (JSON Object) Contains:
    - opinions: (string of opinions on the topic)
    - beliefs: (string of underlying beliefs related to the topic)
    - sentiments_towards_topic: (string, e.g., "Positive", "Skeptical", "Neutral")
- media_consumption: (Array of strings, e.g., ["Social Media Platform", "News Source", "Podcast Genre"])
- motivations: Primary motivations and goals in life (string)
- challenges: Key challenges or pain points they face (string)
- topic_experience: Specific experiences with the topic of the interview (string)
- brand_affinities: (Array of strings, relevant brands)
- communication_style: How they tend to communicate (verbose, concise, formal, casual, etc.) (string)


**Output Requirement:** Return your response ONLY as a JSON object adhering strictly to the structure defined above. Ensure all specified fields are present. Do not include any commentary before or after the JSON object.""",
        model="gpt-4o"
    )
    
    print("\nGenerating respondent persona...")
    result = await Runner.run(persona_generator_agent, f"Generate a detailed persona JSON object for an interview on '{topic}' for target audience '{target_audience}'.")
    
    # --- Define output directory ---
    # Assumes script runs from project root, saving into UI's public folder
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "idi", simulation_id)
    os.makedirs(base_output_dir, exist_ok=True)
    persona_file_path = os.path.join(base_output_dir, "persona.json")
    # --- End Define output directory ---
    
    try:
        # Extract JSON from the agent output
        json_match = re.search(r'```json\s*(.*?)\s*```', result.final_output, re.DOTALL)
        if json_match:
            persona_json = json_match.group(1)
        else:
            persona_json = result.final_output # Assume the whole output is JSON
            
        persona = json.loads(persona_json)
        
        # --- Save persona to file ---
        with open(persona_file_path, "w") as f:
            json.dump(persona, f, indent=2)
        print(f"Generated detailed persona saved to {persona_file_path}")
        return persona
        
    except json.JSONDecodeError as e:
        print(f"Error parsing persona: {e}")
        print("Creating and saving generic persona instead.")
        # Create generic persona if parsing fails
        generic_persona = {
            "name": "Generic Person", 
            "age": 30, 
            "occupation": "Professional",
            "attitudes": {"sentiments_towards_topic": "Neutral"},
            "behaviors": {"consumption_patterns": "Average user"},
            "topic_experience": "Some familiarity",
            "communication_style": "Moderate"
            # Add other required fields with default values if necessary
        }
        # --- Save generic persona ---
        with open(persona_file_path, "w") as f:
            json.dump(generic_persona, f, indent=2)
        print(f"Saved generic persona to {persona_file_path}")
        return generic_persona


async def run_interview(interviewer: InterviewerAgent, respondent: RespondentAgent, num_questions: int) -> List[Tuple[str, str]]:
    """Runs the in-depth interview simulation."""
    print("\n--- Starting In-Depth Interview Simulation ---")
    transcript = []
    current_context = f"The interview topic is: {interviewer.topic}"
    
    # Add respondent intro to the context
    p_data = respondent.persona_data
    intro = f"The respondent is {p_data.get('name', 'Unknown')}, {p_data.get('age', 'Unknown')}, {p_data.get('occupation', 'Unknown')}."
    current_context += f"\n{intro}"
    
    question_count = 0
    
    # Introduction phase
    print("Interviewer is preparing introduction...")
    intro_prompt = f"""Current context:
{current_context}

Start the interview. Introduce yourself as the interviewer, explain the purpose of the interview, establish rapport,
and ask your first question to the respondent. Make the respondent feel comfortable and set the tone for an open conversation."""
    
    intro_result = await Runner.run(interviewer, intro_prompt)
    intro_dialogue = intro_result.final_output
    print(f"STREAM: Interviewer: {intro_dialogue}", flush=True)
    transcript.append(("Interviewer", intro_dialogue))
    current_context += f"""
Interviewer: {intro_dialogue}"""
    
    print("Respondent is thinking...")
    respondent_prompt = f"""Current context:
{current_context}

Respond to the interviewer's introduction and first question from your persona's perspective.
Be authentic and provide a thoughtful, detailed response that reveals something about who you are."""
    
    resp_result = await Runner.run(respondent, respondent_prompt)
    respondent_dialogue = resp_result.final_output
    print(f"STREAM: Respondent: {respondent_dialogue}", flush=True)
    transcript.append(("Respondent", respondent_dialogue))
    current_context += f"""
Respondent: {respondent_dialogue}"""
    
    question_count += 1
    
    # Main interview loop
    while question_count < num_questions:
        print(f"\n--- Question {question_count + 1} of {num_questions} ---")
        
        # Interviewer's turn
        print("Interviewer is thinking...")
        interviewer_prompt = f"""Current context:
{current_context}

Based on the conversation so far, ask your next question or make a follow-up comment.
Remember to use probing techniques, reflect back what you heard, and explore interesting points in depth.
If appropriate, explore a new aspect of the topic that hasn't been discussed yet."""
        
        int_result = await Runner.run(interviewer, interviewer_prompt)
        interviewer_dialogue = int_result.final_output
        print(f"STREAM: Interviewer: {interviewer_dialogue}", flush=True)
        transcript.append(("Interviewer", interviewer_dialogue))
        current_context += f"""
Interviewer: {interviewer_dialogue}"""
        
        # Respondent's turn
        print("Respondent is thinking...")
        # Create a personalized prompt that references the persona details
        p_data = respondent.persona_data
        persona_reminder = f"""
Remember that you are {p_data.get('name', '')}, {p_data.get('age', '')} years old, working as {p_data.get('occupation', '')}.
You have these attitudes: {p_data.get('attitudes', '')}
And these behaviors: {p_data.get('behaviors', '')}
Communication style: {p_data.get('communication_style', 'natural and authentic')}
"""
        
        respondent_prompt = f"""Current context:
{current_context}

{persona_reminder}

Respond to the interviewer's question from your persona's perspective.
Provide a thoughtful, detailed response that reveals your authentic thoughts and feelings.
Draw on your (fictional) personal experiences when relevant."""
        
        resp_result = await Runner.run(respondent, respondent_prompt)
        respondent_dialogue = resp_result.final_output
        print(f"STREAM: Respondent: {respondent_dialogue}", flush=True)
        transcript.append(("Respondent", respondent_dialogue))
        current_context += f"""
Respondent: {respondent_dialogue}"""
        
        question_count += 1
        await asyncio.sleep(0.5) # Small delay
    
    # Conclusion phase
    print("\n--- Concluding Interview ---")
    print("Interviewer is thinking...")
    conclusion_prompt = f"""Current context:
{current_context}

This is the final part of the interview. Ask a concluding question that helps summarize the respondent's overall views 
or gets their final thoughts on the topic. Then thank them for their time and insights."""
    
    concl_result = await Runner.run(interviewer, conclusion_prompt)
    conclusion_dialogue = concl_result.final_output
    print(f"STREAM: Interviewer: {conclusion_dialogue}", flush=True)
    transcript.append(("Interviewer", conclusion_dialogue))
    current_context += f"""
Interviewer: {conclusion_dialogue}"""
    
    print("Respondent is thinking...")
    final_prompt = f"""Current context:
{current_context}

This is your final response in the interview. Share any concluding thoughts and respond to the interviewer's thank you.
You might reflect briefly on the conversation or add anything important you haven't mentioned yet."""
    
    final_result = await Runner.run(respondent, final_prompt)
    final_dialogue = final_result.final_output
    print(f"STREAM: Respondent: {final_dialogue}", flush=True)
    transcript.append(("Respondent", final_dialogue))
    
    print("\n--- Interview Finished ---")
    return transcript

async def analyze_interview(analyst: AnalystAgent, transcript: List[Tuple[str, str]], respondent_data: Dict[str, Any]) -> Tuple[str, Dict]:
    """Analyzes the interview transcript using the AnalystAgent."""
    print("\n--- Analyzing Interview Transcript ---")
    
    # Create a formatted transcript
    formatted_transcript = []
    
    for speaker, dialogue in transcript:
        if speaker == "Respondent":
            name = respondent_data.get("name", "Unknown")
            formatted_line = f"{speaker} ({name}): {dialogue}"
        else:
            formatted_line = f"{speaker}: {dialogue}"
        formatted_transcript.append(formatted_line)
    
    full_transcript = "\n".join(formatted_transcript)
    
    # Create context on respondent for the analysis
    respondent_context = "Respondent Information:\n"
    respondent_context += f"Name: {respondent_data.get('name', 'Unknown')}\n"
    respondent_context += f"Age: {respondent_data.get('age', 'Unknown')}\n"
    respondent_context += f"Occupation: {respondent_data.get('occupation', 'Unknown')}\n"
    respondent_context += f"Education: {respondent_data.get('education', 'Unknown')}\n"
    # Use json.dumps for potentially nested dicts to ensure readability
    respondent_context += f"Demographics: {json.dumps(respondent_data.get('demographics', {}))}\n" 
    respondent_context += f"Psychographics: {json.dumps(respondent_data.get('psychographics', {}))}\n"
    
    analysis_prompt = f"""Analyze the following in-depth interview transcript:

{respondent_context}

TRANSCRIPT:
{full_transcript}

Apply your expert knowledge in qualitative research analysis to identify key themes, sentiments, 
response patterns, and actionable insights from this interview.
Remember to include the structured JSON data for visualization at the end of your analysis.
"""
    
    result = await Runner.run(analyst, analysis_prompt)
    analysis = result.final_output
    
    # Calculate additional metrics for transcript
    word_counts = {}
    response_counts = {}
    
    interviewer_questions = []
    respondent_responses = []
    
    for i, (speaker, dialogue) in enumerate(transcript):
        if speaker not in word_counts:
            word_counts[speaker] = 0
            response_counts[speaker] = 0
        
        word_counts[speaker] += len(dialogue.split())
        response_counts[speaker] += 1
        
        if speaker == "Interviewer":
            interviewer_questions.append(dialogue)
        elif speaker == "Respondent":
            respondent_responses.append(dialogue)
    
    # Extract JSON data for visualization
    print("Extracting visualization data...")
    json_data = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', analysis, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            
            # Add or update response metrics if analyst didn't provide them
            if "response_metrics" not in json_data:
                json_data["response_metrics"] = {}
            
            json_data["response_metrics"]["total_word_count"] = word_counts.get("Respondent", 0)
            json_data["response_metrics"]["avg_response_length"] = int(word_counts.get("Respondent", 0) / max(1, response_counts.get("Respondent", 1)))
            
            # Add estimated unique words / hesitation count if missing
            if "unique_words" not in json_data["response_metrics"]:
                 json_data["response_metrics"]["unique_words"] = int(word_counts.get("Respondent", 0) * 0.3) # Rough estimate
            if "hesitation_count" not in json_data["response_metrics"]:
                 json_data["response_metrics"]["hesitation_count"] = int(word_counts.get("Respondent", 0) / 200) # Rough estimate
            
            # Analyze question types if not present
            if "question_types" not in json_data["response_metrics"]:
                question_types = { "open_ended": 0, "closed": 0, "probing": 0 }
                for q in interviewer_questions:
                    if "?" not in q: continue
                    q_lower = q.lower()
                    if any(p in q_lower for p in ["tell me more", "how did that", "why do you", "could you explain", "what makes you say"]):
                        question_types["probing"] += 1
                    elif any(w in q_lower for w in ["what", "how", "why", "describe", "explain", "share"]):
                        question_types["open_ended"] += 1
                    else:
                        question_types["closed"] += 1
                json_data["response_metrics"]["question_types"] = question_types
            
            # Remove the JSON section from the analysis text
            analysis = analysis.replace(json_match.group(0), '').strip()
            print("Data extraction successful.")
        except json.JSONDecodeError:
            print("Warning: Could not parse JSON data from analyst response.")
            analysis = analysis.strip() # Still strip whitespace
            # Populate with defaults if parsing fails (similar to focus group)
            json_data = { 
                "sentiment_breakdown": {"positive": 60, "neutral": 30, "negative": 10}, 
                "key_themes": [], 
                "response_metrics": {
                    "total_word_count": word_counts.get("Respondent", 0),
                    "avg_response_length": int(word_counts.get("Respondent", 0) / max(1, response_counts.get("Respondent", 1))),
                 },
                 "sentiment_by_question": []
            }
    else:
        print("Warning: No JSON data found in analyst response.")
        analysis = analysis.strip() # Strip whitespace
        # Populate with defaults if no JSON found (similar to focus group)
        json_data = {
            "sentiment_breakdown": {"positive": 60, "neutral": 30, "negative": 10}, 
            "key_themes": [], 
            "response_metrics": {
                "total_word_count": word_counts.get("Respondent", 0),
                "avg_response_length": int(word_counts.get("Respondent", 0) / max(1, response_counts.get("Respondent", 1))),
            },
            "sentiment_by_question": []
        }
    
    print("Analysis Complete.")
    return analysis, json_data

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
                # Filter out themes with potentially missing data
                valid_themes_data = [item for item in themes_data if "theme" in item and "frequency" in item and "sentiment" in item]
                
                if valid_themes_data:
                    themes = [item["theme"] for item in valid_themes_data]
                    frequencies = [item["frequency"] for item in valid_themes_data]
                    sentiments = [item["sentiment"] for item in valid_themes_data]
            
                    plt.figure(figsize=(12, 8))
                    
                    colors = []
                    for sentiment in sentiments:
                        s_lower = str(sentiment).lower()
                        if "positive" in s_lower: colors.append('#99ff99')
                        elif "negative" in s_lower: colors.append('#ff9999')
                        else: colors.append('#66b3ff') # Default to neutral
                    
                    bars = plt.bar(themes, frequencies, color=colors)
                    
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
                    print("Skipping key themes chart: No valid theme data.")
            except Exception as e:
                print(f"Error generating key themes chart: {e}")
        
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
        if "sentiment_by_question" in visualization_data and visualization_data["sentiment_by_question"]:
            try:
                sentiment_flow = visualization_data["sentiment_by_question"]
                sentiment_flow = sorted(sentiment_flow, key=lambda x: x["question_number"])
                
                q_numbers = [item["question_number"] for item in sentiment_flow]
                sentiments = [item["sentiment"] for item in sentiment_flow]
                topics = [item.get("topic", f"Q{item['question_number']}") for item in sentiment_flow] # Use topic if available
                
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
