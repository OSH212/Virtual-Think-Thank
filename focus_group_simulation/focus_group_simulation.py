import os
import asyncio
import re
import json
import argparse
import sys
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from dotenv import load_dotenv
from agents import Agent, Runner
from typing import List, Dict, Any, Tuple
import csv
import numpy as np 

# Load environment variables from .env file
load_dotenv()

# --- Agent Definitions ---

class ModeratorAgent(Agent):
    """
    Guides the focus group discussion, poses questions, manages flow,
    and ensures all participants contribute.
    """
    def __init__(self, topic: str, participant_profiles: List[str], num_rounds: int = 3):
        super().__init__(
            name="Focus Group Moderator",
            instructions=f"""You are an expert focus group moderator with over 15 years of experience.
Your goal is to facilitate a productive discussion on the topic: '{topic}'.
You need to guide the conversation among participants with the following profiles: {participant_profiles}.

MODERATION APPROACH:
- Start with a warm welcome and clear introduction of the topic
- Ask clear, open-ended questions following the funnel technique (general to specific)
- Encourage interaction between participants by specifically calling on them to respond to others
- Use probing questions like "Tell me more about...", "Why do you feel that way?", "How does that compare to..."
- When participants provide brief answers, ask them to elaborate
- Manage dominant speakers tactfully by redirecting questions to others
- Draw out quieter participants with direct but gentle questioning
- Validate contributions while remaining neutral about the topic
- Listen for unexpected insights and follow promising threads
- Synthesize points between rounds to build on previous discussion
- Ensure all key aspects of the topic are covered
- End with final thoughts from each participant and a clear conclusion

Ensure the discussion stays on topic and delve deeper into interesting points.
Manage the flow over approximately {num_rounds} rounds of interaction.
Start by introducing the topic and asking an initial question.
In subsequent rounds, synthesize previous points and pose follow-up questions.
Address participants by their assigned persona names (e.g., Participant_1, Participant_2).
Conclude the session by thanking participants and summarizing key insights.

Your output should be ONLY your dialogue as the moderator.""",
            model= "o3-mini" #gpt-4o" 
        )
        self.topic = topic
        self.participant_profiles = participant_profiles
        self.num_rounds = num_rounds

class ParticipantAgent(Agent):
    """
    Represents a single focus group participant, responding based on their assigned persona.
    """
    def __init__(self, persona_id: str, persona_data: Dict[str, Any], topic: str):
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
        brand_affinities = persona_data.get("brand_affinities", [])

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
Values: {', '.join(values) if isinstance(values, list) else 'N/A'}
Interests: {', '.join(interests) if isinstance(interests, list) else 'N/A'}
---
Attitudes/Opinions/Beliefs related to '{topic}': {opinions}; {beliefs}
Relevant Behaviors: {consumption_patterns}
---
Motivations: {motivations}
Challenges/Pain Points: {challenges}
Media Habits: {', '.join(media_consumption) if isinstance(media_consumption, list) else 'N/A'}
Brand Affinities: {', '.join(brand_affinities) if isinstance(brand_affinities, list) else 'N/A'}
---
"""
        
        super().__init__(
            name=persona_id, # e.g., "Participant_1"
            instructions=f"""You are participating in a focus group discussing: '{topic}'.
Your assigned persona profile is detailed below. Your primary goal is to consistently and authentically portray this *entire* persona throughout the discussion.

{persona_description}

**BEHAVIORAL MANDATES:**

1.  **Embody Fully:** Respond *exclusively* from the perspective of this persona. Do NOT act as a generic AI or participant.
2.  **Integrate Details:** In your responses, actively draw upon your persona's *specific* motivations, challenges, background (occupation, location, demographics, income), values, interests, and even brand affinities or media habits when relevant to the discussion. Don't just state opinions; explain *why* your persona holds them based on their life.
3.  **Show, Don't Just Tell:** Instead of saying "As a [occupation], I think...", try phrasing responses naturally reflecting that background. For example, discussing cost? Relate it to your income bracket or financial challenges. Discussing community? Relate it to your location or values.
4.  **Maintain Consistency:** Keep your opinions, attitudes, and behaviors consistent with the profile throughout the discussion.
5.  **Realistic Nuance:** Use language, tone, and sentence structure appropriate for your persona's background (education, occupation). Include occasional natural speech patterns like "umm," "well," or slight hesitations, but don't overdo it. Show realistic conviction or uncertainty based on the persona's values and attitudes. Acknowledge complexity – sometimes stated values conflict with actual behavior (e.g., valuing sustainability but finding eco-options too expensive or inconvenient – mention this conflict if applicable!).
6.  **Engage Appropriately:** React to the moderator and other participants as your persona naturally would. Build on others' points or offer contrasting views based on your profile.
7.  **Output Format:** Your response must be *ONLY* your dialogue as this participant. Do not include any other text, labels, or explanations.

Think: How would *this specific person* (with all their listed traits) react and respond to the current point in the discussion? """,
            model="o3-mini" 
        )
        self.persona_id = persona_id
        self.topic = topic
        self.persona_data = persona_data

class AnalystAgent(Agent):
    """
    Analyzes the focus group transcript to extract key insights, themes, sentiment,
    and provide a structured, professional report.
    """
    def __init__(self, topic: str, target_audience: str):
        super().__init__(
            name="Focus Group Analyst",
            instructions=f"""You are a senior market research analyst with expertise in qualitative research methodology.
You have been provided with a transcript of a focus group discussion on '{topic}' with target audience: '{target_audience}'.
Your task is to perform a deep analysis of this transcript and generate a comprehensive report that meets professional standards.

**Perform a thorough sentiment analysis of the entire transcript.** Calculate the overall distribution of positive, neutral, and negative sentiment expressed by participants regarding the main topic and related sub-topics discussed. Also, analyze the sentiment expressed by each individual participant.

The report should include:
1.  **Executive Summary:** A concise overview of the key findings (approx. 150 words).
2.  **Research Background:** Brief context on the study purpose, topic, and target audience.
3.  **Methodology:** Brief description of the focus group simulation approach.
4.  **Participant Overview:** Summary of participant profiles/personas and how they represent the target audience.
5.  **Key Themes:** Identify and elaborate on the major recurring themes, ideas, and opinions expressed. 
    - Provide theme frequency/prominence based on your analysis.
    - Include representative direct quotes for each theme (with attribution).
    - Note areas of consensus and disagreement.
6.  **Sentiment Analysis:** Assess the overall sentiment towards the topic and specific aspects discussed, based *directly on your analysis of the transcript*.
    - Quantify the calculated sentiment distribution (e.g., percentage positive, negative, neutral).
    - Analyze how sentiment varies by participant persona characteristics.
    - Identify sentiment trends or shifts during the discussion.
7.  **Participant Dynamics:** Analyze interaction patterns, influence, and engagement levels.
8.  **Actionable Insights & Recommendations:** Translate the findings into strategic insights and actionable recommendations relevant to the focus group's topic.
9.  **Potential Biases/Limitations:** Acknowledge any potential biases observed or limitations of the simulation.
10. **Appendix:** Suggestions for follow-up research.

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
    {{"theme": "<Identified Theme 1>", "frequency": <calculated_frequency_1>, "description": "<Brief theme description>"}},
    {{"theme": "<Identified Theme 2>", "frequency": <calculated_frequency_2>, "description": "<Brief theme description>"}}
    // Add more themes as identified
  ],
  "participant_sentiment": {{
    "Participant_1": {{"positive": <p1_positive_%>, "neutral": <p1_neutral_%>, "negative": <p1_negative_%>, "name": "<Persona 1 Name>"}},
    "Participant_2": {{"positive": <p2_positive_%>, "neutral": <p2_neutral_%>, "negative": <p2_negative_%>, "name": "<Persona 2 Name>"}}
    // Add all participants analyzed
  }},
  "engagement_metrics": {{
    "Participant_1": {{"word_count": <p1_word_count>, "response_count": <p1_response_count>, "interaction_score": <calculated_p1_score>}},
    "Participant_2": {{"word_count": <p2_word_count>, "response_count": <p2_response_count>, "interaction_score": <calculated_p2_score>}}
     // Add all participants analyzed
  }}
}}
```

**Crucially, ensure the values in the JSON section accurately reflect the results of YOUR analysis of the provided transcript content. Do not use the placeholder values literally.** This JSON should be placed at the end of your analysis, clearly separated from the narrative report.""",
            model="gpt-4o" 
        )
        self.topic = topic
        self.target_audience = target_audience

# --- Core Functions ---

async def generate_persona_profiles(target_audience: str, num_participants: int, topic: str) -> List[Dict[str, Any]]:
    """
    Generates detailed persona profiles as structured data for focus group participants.
    """
    persona_generator_agent = Agent(
        name="Persona Generator",
        instructions=f"""You are an expert research methodologist specializing in participant recruitment and persona development.
Based on the target audience description: '{target_audience}', generate {num_participants} distinct, realistic, and detailed participant persona profiles suitable for a focus group on the topic: '{topic}'.

**MANDATORY JSON STRUCTURE:** Each persona MUST be a JSON object with the exact fields specified below. Fields like `demographics`, `psychographics`, `behaviors`, and `attitudes` MUST be **nested JSON objects** containing their respective sub-fields as described. DO NOT use simple strings for these nested objects.

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
- brand_affinities: (Array of strings, relevant brands)

**Example of Nested Structure:**
```json
{{
  "name": "Example Person",
  "age": 35,
  // ... other top-level fields ...
  "demographics": {{
    "ethnicity": "Example Ethnicity",
    "marital_status": "Married",
    "household_composition": "With family"
  }},
  "psychographics": {{
     "values": ["Value 1", "Value 2"],
     "interests": ["Interest 1", "Interest 2"],
     // ... etc
  }},
  // ... other top-level fields ...
}}
```

**Output Requirement:** Return your response ONLY as a JSON array containing exactly {num_participants} valid persona objects adhering strictly to the structure defined above. Ensure all specified fields are present. Do not include any commentary before or after the JSON array.""",
        model="gpt-4o"
    )
    
    print("\nGenerating participant personas...")
    result = await Runner.run(persona_generator_agent, f"Generate {num_participants} detailed personas for topic '{topic}' and audience '{target_audience}'.")
    
    try:
        # Extract JSON from the agent output
        json_match = re.search(r'```json\s*(.*?)\s*```', result.final_output, re.DOTALL)
        if json_match:
            persona_json = json_match.group(1)
        else:
            # If not in code block, try to parse the entire output as JSON
            persona_json = result.final_output
            
        personas = json.loads(persona_json)
        
        if not isinstance(personas, list) or len(personas) < num_participants:
            print(f"Warning: Could not generate {num_participants} personas. Creating generic personas instead.")
            # Create generic personas if parsing fails
            personas = [{"name": f"Generic Person {i}", "age": 30, "occupation": "Professional"} for i in range(num_participants)]
        
        print(f"Generated {len(personas)} detailed personas.")
        
        # Create a timestamp for the personas file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save personas to a file for reference
        os.makedirs("output_data", exist_ok=True)
        with open(f"output_data/personas_{timestamp}.json", "w") as f:
            json.dump(personas, f, indent=2)
            
        return personas[:num_participants]  # Return only the requested number
        
    except json.JSONDecodeError as e:
        print(f"Error parsing personas: {e}")
        print("Creating generic personas instead.")
        # Create generic personas if parsing fails
        return [{"name": f"Generic Person {i}", "age": 30, "occupation": "Professional"} for i in range(num_participants)]

async def create_participant_agents(target_audience: str, num_participants: int, topic: str) -> Tuple[List[ParticipantAgent], List[str]]:
    """Dynamically creates participant agents based on the target audience description."""
    # Generate detailed persona profiles
    personas = await generate_persona_profiles(target_audience, num_participants, topic)
    
    # Create ParticipantAgent instances
    participants = []
    participant_profiles_for_moderator = []
    
    for i, persona in enumerate(personas):
        persona_id = f"Participant_{i+1}"
        participants.append(ParticipantAgent(persona_id, persona, topic))
        
        # Create a summary for the moderator
        persona_summary = f"{persona_id}: {persona.get('name', 'Unknown')}, {persona.get('age', 'Unknown')} - {persona.get('occupation', 'Unknown')}. {persona.get('psychographics', '')}."
        participant_profiles_for_moderator.append(persona_summary)
        
        # Print a snippet of the persona
        name = persona.get('name', 'Unknown')
        age = persona.get('age', 'Unknown')
        occupation = persona.get('occupation', 'Unknown')
        print(f"Created Agent: {persona_id} - {name}, {age}, {occupation}")

    return participants, participant_profiles_for_moderator

async def run_simulation(moderator: ModeratorAgent, participants: List[ParticipantAgent], num_rounds: int) -> List[Tuple[str, str]]:
    """Runs the focus group simulation loop."""
    print("\n--- Starting Focus Group Simulation ---")
    transcript = []
    current_context = f"The discussion topic is: {moderator.topic}"
    
    # Add participant intros to the context
    for participant in participants:
        p_data = participant.persona_data
        intro = f"{participant.persona_id} is {p_data.get('name', 'Unknown')}, {p_data.get('age', 'Unknown')}, {p_data.get('occupation', 'Unknown')}."
        current_context += f"\n{intro}"
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        # Moderator's turn
        print("Moderator is thinking...")
        moderator_prompt = f"""Current context:
{current_context}

Based on the discussion so far, what is your next question or statement for round {round_num} of {num_rounds}? 
Remember to use moderator techniques like probing, reflecting, and ensuring all participants are engaged.
If appropriate for this stage, focus on areas that need more exploration or where there are differences of opinion."""

        if round_num == 1:
            moderator_prompt = f"""Current context:
{current_context}

Start the focus group. Introduce yourself as the moderator, welcome the participants, introduce the topic '{moderator.topic}' and explain the format of the discussion. 
Then ask your opening question to the participants ({', '.join(p.persona_id for p in participants)}).
Your first question should be general and open-ended to start the conversation flowing."""

        if round_num == num_rounds:
            moderator_prompt = f"""Current context:
{current_context}

This is the final round of the discussion. Ask a closing question that helps summarize participants' views or gets their final thoughts on the topic. 
Make sure to give each participant an opportunity to share any final insights they haven't expressed yet."""

        mod_result = await Runner.run(moderator, moderator_prompt)
        moderator_dialogue = mod_result.final_output
        print(f"STREAM: Moderator: {moderator_dialogue}", flush=True)
        transcript.append(("Moderator", moderator_dialogue))
        current_context += f"""
Moderator: {moderator_dialogue}"""

        # Participants' turn (sequential response for simplicity, parallel to be added)
        for participant in participants:
            # print(f"{participant.persona_id} is thinking...") # Keep internal logs separate from stream
            
            # Create a more personalized prompt that references the persona details
            p_data = participant.persona_data
            persona_reminder = f"""
Remember that you are {p_data.get('name', '')}, {p_data.get('age', '')} years old, working as {p_data.get('occupation', '')}.
You have these attitudes: {p_data.get('attitudes', '')}
And these behaviors: {p_data.get('behaviors', '')}
"""
            
            participant_prompt = f"""Current context:
{current_context}

{persona_reminder}

Respond to the moderator and the ongoing discussion from your persona's perspective. 
Be authentic to your character's viewpoint and life experiences.
If other participants have spoken, consider responding to or building upon their points if relevant."""

            part_result = await Runner.run(participant, participant_prompt)
            participant_dialogue = part_result.final_output
            print(f"STREAM: {participant.persona_id}: {participant_dialogue}", flush=True)
            transcript.append((participant.persona_id, participant_dialogue))
            current_context += f"""
{participant.persona_id}: {participant_dialogue}"""
            await asyncio.sleep(0.5) # Small delay to mimic turn-taking and avoid rate limits

    # Moderator conclusion
    print("\n--- Concluding Simulation ---")
    print("Moderator is thinking...")
    mod_result = await Runner.run(moderator, f"""Current context:
{current_context}

Conclude the focus group session. Thank the participants for their time and valuable insights.
Provide a brief summary of the key points discussed and the diverse perspectives shared.
End on a positive and appreciative note.""")
    moderator_dialogue = mod_result.final_output
    print(f"STREAM: Moderator: {moderator_dialogue}", flush=True)
    transcript.append(("Moderator", moderator_dialogue))

    print("\n--- Simulation Finished ---")
    return transcript

async def analyze_transcript(analyst: AnalystAgent, transcript: List[Tuple[str, str]], participant_data: List[Dict]) -> Tuple[str, Dict]:
    """Analyzes the transcript using the AnalystAgent."""
    print("\n--- Analyzing Transcript ---")
    
    # Create a formatted transcript with speaker information
    participant_info = {f"Participant_{i+1}": data for i, data in enumerate(participant_data)}
    formatted_transcript = []
    
    for speaker, dialogue in transcript:
        if speaker.startswith("Participant_"):
            person_data = participant_info.get(speaker, {})
            name = person_data.get("name", "Unknown")
            formatted_line = f"{speaker} ({name}): {dialogue}"
        else:
            formatted_line = f"{speaker}: {dialogue}"
        formatted_transcript.append(formatted_line)
    
    full_transcript = "\n".join(formatted_transcript)
    
    # Create context on participants for the analysis
    participant_context = "Participant Information:\n"
    for pid, data in participant_info.items():
        participant_context += f"{pid}: {data.get('name', 'Unknown')}, {data.get('age', 'Unknown')}, {data.get('occupation', 'Unknown')}\n"
    
    analysis_prompt = f"""Analyze the following focus group transcript:

{participant_context}

TRANSCRIPT:
{full_transcript}

Apply your expert knowledge in qualitative research analysis to identify key themes, sentiments, participant dynamics, and actionable insights.
Remember to include the structured JSON data for visualization at the end of your analysis.
"""
    
    result = await Runner.run(analyst, analysis_prompt)
    analysis = result.final_output
    
    # Calculate additional metrics for transcript
    word_counts = {}
    response_counts = {}
    
    for speaker, dialogue in transcript:
        if speaker not in word_counts:
            word_counts[speaker] = 0
            response_counts[speaker] = 0
        
        word_counts[speaker] += len(dialogue.split())
        response_counts[speaker] += 1
    
    # Extract JSON data for visualization
    print("Extracting visualization data...")
    json_data = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', analysis, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            
            # Add the calculated metrics if not present
            if "engagement_metrics" not in json_data:
                json_data["engagement_metrics"] = {}
                for speaker in word_counts:
                    if speaker != "Moderator":
                        json_data["engagement_metrics"][speaker] = {
                            "word_count": word_counts.get(speaker, 0),
                            "response_count": response_counts.get(speaker, 0),
                            "interaction_score": min(10, int(word_counts.get(speaker, 0) / 100))  # Simple scoring
                        }
            
            # Remove the JSON section from the analysis
            analysis = analysis.replace(json_match.group(0), '')
            print("Data extraction successful.")
        except json.JSONDecodeError:
            print("Warning: Could not parse JSON data from analyst response.")
    else:
        print("Warning: No JSON data found in analyst response.")
        
        # Create basic JSON data from our calculations
        json_data = {
            "engagement_metrics": {},
            "sentiment_breakdown": {"positive": 50, "neutral": 30, "negative": 20},  # Default values
            "key_themes": [{"theme": "General Discussion", "frequency": 1, "description": "Various topics discussed"}],
            "participant_sentiment": {}
        }
        
        for speaker in word_counts:
            if speaker != "Moderator":
                json_data["engagement_metrics"][speaker] = {
                    "word_count": word_counts.get(speaker, 0),
                    "response_count": response_counts.get(speaker, 0),
                    "interaction_score": min(10, int(word_counts.get(speaker, 0) / 100))
                }
                
                # Default sentiment distribution
                json_data["participant_sentiment"][speaker] = {
                    "positive": 50, 
                    "neutral": 30, 
                    "negative": 20,
                    "name": participant_info.get(speaker, {}).get("name", "Unknown")
                }
    
    print("Analysis Complete.")
    return analysis.strip(), json_data # Strip leading/trailing whitespace

def generate_report(analysis: str, parameters: Dict[str, Any], transcript: List[Tuple[str, str]], 
                   visualization_data: Dict[str, Any], participant_data: List[Dict], simulation_id: str):
    """Generates and saves the final report with visualizations."""
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- Generating Report for Simulation ID: {simulation_id} ---")

    # Define the base output directory within the UI's public folder
    # Assumes the script is run from the root of the openai-agents-simulation directory
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "focus-group", simulation_id)
    viz_rel_path_prefix = f"/simulations/focus-group/{simulation_id}/visualizations" # Relative path for markdown links
    viz_abs_dir = os.path.join(base_output_dir, "visualizations")

    # Create directories if they don't exist
    os.makedirs(viz_abs_dir, exist_ok=True)
    print(f"Output directory created/ensured: {base_output_dir}")

    report_content = f"""# Focus Group Simulation Report

## 1. Simulation Parameters
- **Topic:** {parameters['topic']}
- **Target Audience:** {parameters['target_audience']}
- **Number of Participants:** {parameters['num_participants']}
- **Number of Rounds:** {parameters['num_rounds']}
 - **Simulation ID:** {simulation_id}
 - **Date/Time:** {current_timestamp}

## 2. Participant Profiles

"""
    # Add participant profiles to the report
    for i, persona in enumerate(participant_data):
        pid = f"Participant_{i+1}"
        report_content += f"### {pid}: {persona.get('name', 'Unknown')}\n"
        report_content += f"- **Age:** {persona.get('age', 'Unknown')}\n"
        report_content += f"- **Occupation:** {persona.get('occupation', 'Unknown')}\n"
        report_content += f"- **Education:** {persona.get('education', 'Unknown')}\n"
        report_content += f"- **Demographics:** {persona.get('demographics', 'Unknown')}\n"
        report_content += f"- **Psychographics:** {persona.get('psychographics', 'Unknown')}\n\n"
    
    report_content += f"""
## 3. Analysis Results
{analysis}

## 4. Visualizations
*See attached visualization files referenced below*

"""
    
    # Generate visualizations if data is available
    visualization_files = []
    if visualization_data:
        print(f"Generating visualizations in {viz_abs_dir}...")
        
        # 1. Sentiment Breakdown Pie Chart
        if "sentiment_breakdown" in visualization_data:
            sentiment_data = visualization_data["sentiment_breakdown"]
            plt.figure(figsize=(10, 7))
            labels = list(sentiment_data.keys())
            sizes = list(sentiment_data.values())
            colors = ['#66b3ff', '#99ff99', '#ff9999']
            explode = (0.1, 0, 0)  # explode the 1st slice (positive)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Overall Sentiment Distribution', fontsize=16)
            
            pie_file = os.path.join(viz_abs_dir, "sentiment_pie_chart.png")
            plt.savefig(pie_file, dpi=300)
            plt.close()
            visualization_files.append(os.path.basename(pie_file))
            report_content += f"![Sentiment Distribution]({viz_rel_path_prefix}/sentiment_pie_chart.png)\n\n"
        
        # 2. Key Themes Bar Chart
        if "key_themes" in visualization_data:
            themes_data = visualization_data["key_themes"]
            themes = [item["theme"] for item in themes_data]
            frequencies = [item["frequency"] for item in themes_data]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(themes, frequencies, color='skyblue')
            
            # Add data labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}', ha='center', va='bottom')
            
            plt.xlabel('Themes', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title('Key Themes Frequency', fontsize=16)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.tight_layout()
            
            themes_file = os.path.join(viz_abs_dir, "key_themes_chart.png")
            plt.savefig(themes_file, dpi=300)
            plt.close()
            visualization_files.append(os.path.basename(themes_file))
            report_content += f"![Key Themes]({viz_rel_path_prefix}/key_themes_chart.png)\n\n"
        
        # 3. Participant Sentiment Comparison (Stacked Bar Chart)
        if "participant_sentiment" in visualization_data:
            participant_data = visualization_data["participant_sentiment"]
            participants = list(participant_data.keys())
            
            # Get names for display
            participant_names = []
            for p in participants:
                name = participant_data[p].get("name", p)
                participant_names.append(f"{p}\n({name})")
            
            positive_values = [participant_data[p]["positive"] for p in participants]
            neutral_values = [participant_data[p]["neutral"] for p in participants]
            negative_values = [participant_data[p]["negative"] for p in participants]
            
            plt.figure(figsize=(14, 9))
            
            width = 0.8
            plt.bar(participant_names, positive_values, width, label='Positive', color='#99ff99')
            plt.bar(participant_names, neutral_values, width, bottom=positive_values, label='Neutral', color='#66b3ff')
            
            # Calculate the bottom position for negative values
            bottom_negative = [p + n for p, n in zip(positive_values, neutral_values)]
            plt.bar(participant_names, negative_values, width, bottom=bottom_negative, label='Negative', color='#ff9999')
            
            plt.xlabel('Participants', fontsize=14)
            plt.ylabel('Sentiment Distribution (%)', fontsize=14)
            plt.title('Sentiment Distribution by Participant', fontsize=16)
            plt.legend(fontsize=12)
            plt.xticks(fontsize=12)
            plt.tight_layout()
            
            participant_file = os.path.join(viz_abs_dir, "participant_sentiment.png")
            plt.savefig(participant_file, dpi=300)
            plt.close()
            visualization_files.append(os.path.basename(participant_file))
            report_content += f"![Participant Sentiment]({viz_rel_path_prefix}/participant_sentiment.png)\n\n"
            
        # 4. Engagement Metrics Radar Chart
        if "engagement_metrics" in visualization_data:
            engagement_data = visualization_data["engagement_metrics"]
            participants = list(engagement_data.keys())
            
            # Get names for labels
            participant_labels = []
            for p in participants:
                if p in participant_data:
                    name = participant_data[p].get("name", p)
                    participant_labels.append(f"{p}\n({name})")
                else:
                    participant_labels.append(p)
            
            # Metrics to plot
            metrics = ['word_count', 'response_count', 'interaction_score']
            
            # Normalize the data for better visualization
            max_word_count = max([engagement_data[p].get('word_count', 0) for p in participants])
            max_response_count = max([engagement_data[p].get('response_count', 0) for p in participants])
            max_interaction = max([engagement_data[p].get('interaction_score', 0) for p in participants])
            
            # Create a bar chart with grouped bars
            x = np.arange(len(participants))  # the label locations
            width = 0.25  # the width of the bars
            
            fig, ax = plt.subplots(figsize=(14, 8))
            rects1 = ax.bar(x - width, [engagement_data[p].get('word_count', 0)/10 for p in participants], width, label='Word Count (÷10)')
            rects2 = ax.bar(x, [engagement_data[p].get('response_count', 0) for p in participants], width, label='Response Count')
            rects3 = ax.bar(x + width, [engagement_data[p].get('interaction_score', 0) for p in participants], width, label='Interaction Score')
            
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel('Participants', fontsize=14)
            ax.set_ylabel('Metrics', fontsize=14)
            ax.set_title('Participant Engagement Metrics', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(participant_labels)
            ax.legend()
            
            fig.tight_layout()
            
            engagement_file = os.path.join(viz_abs_dir, "engagement_metrics.png")
            plt.savefig(engagement_file, dpi=300)
            plt.close()
            visualization_files.append(os.path.basename(engagement_file))
            report_content += f"![Engagement Metrics]({viz_rel_path_prefix}/engagement_metrics.png)\n\n"
    
    report_content += """
## 5. Full Transcript
"""
    for speaker, dialogue in transcript:
        report_content += f"- **{speaker}:** {dialogue}\n\n"

    # Save the report to a file with timestamp
    report_filename = os.path.join(base_output_dir, "report.md")
    with open(report_filename, "w", encoding='utf-8') as f:
        f.write(report_content)
    print(f"Report saved successfully as: {report_filename}")
    
    # Also save the transcript as a CSV file
    transcript_filename = os.path.join(base_output_dir, "transcript.csv")
    with open(transcript_filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Dialogue"])
        for speaker, dialogue in transcript:
            writer.writerow([speaker, dialogue])
    print(f"Transcript saved successfully as: {transcript_filename}")
    
    if visualization_files:
        print(f"Generated {len(visualization_files)} visualization files:")
        for viz_file in visualization_files:
            print(f" - {viz_file}")


# --- Main Execution ---

async def main():
    """Main function to run the focus group simulation."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with OPENAI_API_KEY=your_key")
        return

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a simulated focus group.")
    parser.add_argument(
        "--topic",
        type=str,
        default="Consumer preferences for sustainable footwear",
        help="The topic for the focus group discussion."
    )
    parser.add_argument(
        "--target_audience",
        type=str,
        default="Urban professionals aged 25-40 interested in sustainability",
        help="Description of the target audience/personas."
    )
    parser.add_argument(
        "--num_participants",
        type=int,
        default=4,
        help="Number of virtual participants."
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of discussion rounds."
    )
    parser.add_argument(
        "--simulation_id",
        type=str,
        required=True, # Make simulation_id mandatory
        help="Unique ID for this simulation run, used for naming output directories."
    )
    args = parser.parse_args()

    # Use parsed arguments directly
    params = {
        "topic": args.topic,
        "target_audience": args.target_audience,
        "num_participants": args.num_participants,
        "num_rounds": args.num_rounds
    }
    simulation_id = args.simulation_id # Get the simulation ID
    print("\n--- Focus Group Parameters ---")
    print(f"Topic: {params['topic']}")
    print(f"Target Audience: {params['target_audience']}")
    print(f"Participants: {params['num_participants']}")
    print(f"Rounds: {params['num_rounds']}")
    print("-----------------------------")

    participants, participant_profiles = await create_participant_agents(
        params["target_audience"], params["num_participants"], params["topic"]
    )

    # Get the raw persona data for the report
    participant_data = [p.persona_data for p in participants]

    # Ensure we have participants before creating the moderator that needs their profiles
    if not participants:
        print("Error: Failed to create participant agents. Exiting.")
        return

    moderator = ModeratorAgent(
        topic=params["topic"],
        participant_profiles=participant_profiles,
        num_rounds=params["num_rounds"]
    )
    analyst = AnalystAgent(topic=params["topic"], target_audience=params["target_audience"])

    transcript = await run_simulation(moderator, participants, params["num_rounds"])

    if transcript:
        analysis, visualization_data = await analyze_transcript(analyst, transcript, participant_data)
        generate_report(analysis, params, transcript, visualization_data, participant_data, simulation_id)
    else:
        print("Simulation did not produce a transcript. Analysis skipped.")

if __name__ == "__main__":
    asyncio.run(main())
