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
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import csv
import numpy as np 
import networkx as nx
from textblob import TextBlob
from pyvis.network import Network
from rich.console import Console
from rich.progress import Progress
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, validator, ValidationError, model_validator, field_validator
import logging
from pathlib import Path
import traceback
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("focus_group")

# Create rich console for enhanced output
console = Console()

# Load environment variables from .env file
load_dotenv()

# --- Pydantic Models for Data Validation ---

class Demographics(BaseModel):
    ethnicity: str = Field(default="Unknown")
    marital_status: str = Field(default="Unknown")
    household_composition: Optional[str] = Field(default=None)
    other_relevant_demographics: Optional[str] = Field(default=None)

class Psychographics(BaseModel):
    values: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    lifestyle_details: Optional[str] = Field(default=None)
    personality_traits: List[str] = Field(default_factory=list)

class Behaviors(BaseModel):
    consumption_patterns: Optional[str] = Field(default=None)
    usage_habits: Optional[str] = Field(default=None)
    other_relevant_behaviors: Optional[str] = Field(default=None)

class Attitudes(BaseModel):
    opinions: Optional[str] = Field(default=None)
    beliefs: Optional[str] = Field(default=None)
    sentiments_towards_topic: Optional[str] = Field(default=None)

class PersonaProfile(BaseModel):
    name: str = Field(default="Unknown")
    age: Union[int, str] = Field(default="Unknown")
    gender: str = Field(default="Unknown")
    occupation: str = Field(default="Unknown")
    education: Optional[str] = Field(default="Unknown")
    location: Optional[str] = Field(default="Unknown")
    income_bracket: Optional[str] = Field(default="Unknown")
    demographics: Demographics = Field(default_factory=Demographics)
    psychographics: Psychographics = Field(default_factory=Psychographics)
    behaviors: Behaviors = Field(default_factory=Behaviors)
    attitudes: Attitudes = Field(default_factory=Attitudes)
    media_consumption: List[str] = Field(default_factory=list)
    motivations: Optional[str] = Field(default=None)
    challenges: Optional[str] = Field(default=None)
    brand_affinities: List[str] = Field(default_factory=list)
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v
    
    def get_summary(self) -> str:
        """Returns a concise summary of the persona for the moderator."""
        return f"{self.name}, {self.age} - {self.occupation}. {self.location or ''}."
    
    def get_full_description(self) -> str:
        """Returns a detailed description of the persona for embodiment."""
        values_str = ", ".join(self.psychographics.values) if self.psychographics.values else "N/A"
        interests_str = ", ".join(self.psychographics.interests) if self.psychographics.interests else "N/A"
        media_str = ", ".join(self.media_consumption) if self.media_consumption else "N/A"
        brands_str = ", ".join(self.brand_affinities) if self.brand_affinities else "N/A"
        
        return f"""
Name: {self.name}, Age: {self.age}, Gender: {self.gender}
Location: {self.location or 'Unknown'}
Occupation: {self.occupation}
Education: {self.education or 'Unknown'}
Income Bracket: {self.income_bracket or 'Unknown'}
Ethnicity: {self.demographics.ethnicity}, Marital Status: {self.demographics.marital_status}
---
Values: {values_str}
Interests: {interests_str}
---
Attitudes/Opinions: {self.attitudes.opinions or 'Unknown'}
Beliefs: {self.attitudes.beliefs or 'Unknown'}
Relevant Behaviors: {self.behaviors.consumption_patterns or 'Unknown'}
---
Motivations: {self.motivations or 'Unknown'}
Challenges/Pain Points: {self.challenges or 'Unknown'}
Media Habits: {media_str}
Brand Affinities: {brands_str}
---
"""

class DialogueEntry(BaseModel):
    speaker_id: str
    speaker_name: str 
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"{self.speaker_name}: {self.content}"

class SentimentScore(BaseModel):
    positive: float
    neutral: float
    negative: float
    
    @model_validator(mode='after')
    def validate_percentages(self):
        total = self.positive + self.neutral + self.negative
        if not 0.99 <= total <= 1.01:  # Allow minor floating point imprecision
            # Normalize to ensure they sum to 1
            factor = 1.0 / total
            self.positive *= factor
            self.neutral *= factor
            self.negative *= factor
        return self

class ThemeData(BaseModel):
    theme: str
    frequency: float
    description: str

class ParticipantMetrics(BaseModel):
    word_count: int
    response_count: int
    interaction_score: float
    
class ParticipantSentiment(BaseModel):
    positive: float
    neutral: float
    negative: float
    name: str

class AnalysisData(BaseModel):
    sentiment_breakdown: SentimentScore
    key_themes: List[ThemeData]
    participant_sentiment: Dict[str, ParticipantSentiment]
    engagement_metrics: Dict[str, ParticipantMetrics]
    
    @classmethod
    def from_json_str(cls, json_str: str) -> 'AnalysisData':
        """Safely parse JSON string to AnalysisData object."""
        try:
            # Find JSON block in the analyst output
            json_match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
            if json_match:
                data_dict = json.loads(json_match.group(1))
                return cls(**data_dict)
            else:
                raise ValueError("No JSON block found in analyst output")
        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing analysis data: {e}")
            # Create default values
            return cls(
                sentiment_breakdown=SentimentScore(positive=0.33, neutral=0.34, negative=0.33),
                key_themes=[ThemeData(theme="Error in analysis", frequency=1.0, description="Could not parse themes")],
                participant_sentiment={},
                engagement_metrics={}
            )

# --- Agent Definitions ---

class ModeratorAgent(Agent):
    """
    Guides the focus group discussion, poses questions, manages flow,
    and ensures all participants contribute.
    """
    def __init__(self, topic: str, participant_profiles: List[PersonaProfile], num_rounds: int = 3):
        super().__init__(
            name="Focus Group Moderator",
            instructions=f"""You are an expert focus group moderator with over 15 years of experience.
Your goal is to facilitate a productive discussion on the topic: '{topic}'.
You need to guide the conversation among participants with the following profiles:
{[p.get_summary() for p in participant_profiles]}

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
Address participants by their real names (e.g., '{participant_profiles[0].name if participant_profiles else "John"}').
Conclude the session by thanking participants and summarizing key insights.

Your output should be ONLY your dialogue as the moderator.""",
            model= "gpt-4o" 
        )
        self.topic = topic
        self.participant_profiles = participant_profiles
        self.num_rounds = num_rounds

class ParticipantAgent(Agent):
    """
    Represents a single focus group participant, responding based on their assigned persona.
    """
    def __init__(self, persona_id: str, persona_profile: PersonaProfile, topic: str):
        persona_description = persona_profile.get_full_description()
        
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
            model="gpt-4o" 
        )
        self.persona_id = persona_id
        self.topic = topic
        self.persona_profile = persona_profile

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

**YOU MUST CALCULATE ALL METRICS BASED ON YOUR ANALYSIS:** You must analyze the sentiment of each statement in the transcript to determine whether it is positive, neutral, or negative. Calculate actual percentages based on your analysis - do not use placeholder or arbitrary values.

The report should include:
1.  **Executive Summary:** A concise overview of the key findings (approx. 150 words).
2.  **Research Background:** Brief context on the study purpose, topic, and target audience.
3.  **Methodology:** Brief description of the focus group simulation approach.
4.  **Participant Overview:** Summary of participant profiles/personas and how they represent the target audience.
5.  **Key Themes:** Identify and elaborate on the major recurring themes, ideas, and opinions expressed. 
    - Calculate theme frequency/prominence based on your analysis.
    - Include representative direct quotes for each theme (with attribution).
    - Note areas of consensus and disagreement.
6.  **Sentiment Analysis:** Assess the overall sentiment towards the topic and specific aspects discussed, based *directly on your analysis of the transcript*.
    - Calculate and report the sentiment distribution (positive, negative, neutral percentages).
    - Analyze how sentiment varies by participant persona characteristics.
    - Identify sentiment trends or shifts during the discussion.
7.  **Participant Dynamics:** Analyze interaction patterns, influence, and engagement levels.
8.  **Actionable Insights & Recommendations:** Translate the findings into strategic insights and actionable recommendations relevant to the focus group's topic.
9.  **Potential Biases/Limitations:** Acknowledge any potential biases observed or limitations of the simulation.
10. **Appendix:** Suggestions for follow-up research.

Structure your output clearly using markdown headings. Ensure the analysis is objective, insightful, and presented professionally, as expected from a top-tier research firm.

AFTER YOUR ANALYSIS, include a JSON-formatted section with structured data derived *from your analysis* for visualization. This must be valid, properly escaped JSON:

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

**Crucially, you must actively analyze the transcript to generate these values - do not use placeholder values.** Calculate word counts from the text, derive sentiment scores by analyzing the content of each statement, and determine theme frequencies based on topic occurrences.""",
            model="gpt-4o" 
        )
        self.topic = topic
        self.target_audience = target_audience

# --- Core Functions ---

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, ValidationError))
)
async def generate_persona_profiles(target_audience: str, num_participants: int, topic: str) -> List[PersonaProfile]:
    """
    Generates detailed persona profiles as structured data for focus group participants.
    Uses retry logic for resilience.
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

**EXTREMELY IMPORTANT:** The JSON output MUST be valid. Ensure all arrays are properly formatted with square brackets, all objects use curly braces, and there are no trailing commas. JSON validity is critical.

**Output Requirement:** Return your response ONLY as a JSON array containing exactly {num_participants} valid persona objects adhering strictly to the structure defined above. Ensure all specified fields are present. Do not include any commentary before or after the JSON array.""",
        model="gpt-4o"
    )
    
    # Replaced console.status with simple console.log to avoid nested live displays
    console.log(f"[bold green]Generating {num_participants} participant personas...[/bold green]")
    
    try:
        result = await Runner.run(persona_generator_agent, f"Generate {num_participants} detailed personas for topic '{topic}' and audience '{target_audience}'.")
        
        # Extract JSON from the agent output
        json_match = re.search(r'```json\s*(.*?)\s*```', result.final_output, re.DOTALL)
        if json_match:
            persona_json = json_match.group(1)
        else:
            # If not in code block, try to parse the entire output as JSON
            persona_json = result.final_output
            
        # Parse JSON into raw dictionaries
        raw_personas = json.loads(persona_json)
        
        # Validate and convert to Pydantic models
        personas = []
        for i, raw_persona in enumerate(raw_personas):
            try:
                persona = PersonaProfile(**raw_persona)
                personas.append(persona)
            except ValidationError as e:
                logger.warning(f"Validation error for persona {i+1}: {e}")
                # Create a fallback persona with available data
                fallback = PersonaProfile(
                    name=raw_persona.get("name", f"Person {i+1}"),
                    age=raw_persona.get("age", 30),
                    gender=raw_persona.get("gender", "Not specified"),
                    occupation=raw_persona.get("occupation", "Professional")
                )
                personas.append(fallback)
        
        if len(personas) < num_participants:
            logger.warning(f"Only generated {len(personas)}/{num_participants} valid personas")
            # Fill in with generic personas if needed
            for i in range(len(personas), num_participants):
                generic = PersonaProfile(
                    name=f"Generic Person {i+1}",
                    age=30 + i,
                    gender="Not specified",
                    occupation="Professional"
                )
                personas.append(generic)
        
        console.log(f"[green]Successfully generated {len(personas)} detailed personas.")
        
        # Create a timestamp for the personas file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save personas to a file for reference
        os.makedirs("output_data", exist_ok=True)
        
        # Save as JSON for reference
        persona_dicts = [p.model_dump() for p in personas]
        with open(f"output_data/personas_{timestamp}.json", "w") as f:
            json.dump(persona_dicts, f, indent=2)
            
        return personas[:num_participants]  # Return only the requested number
        
    except Exception as e:
        logger.error(f"Error in persona generation: {e}")
        logger.error(traceback.format_exc())
        console.log("[bold red]Failed to generate personas. Creating generic personas.")
        
        # Create generic personas as fallback
        return [PersonaProfile(
            name=f"Generic Person {i}",
            age=30 + i,
            gender="Not specified",
            occupation="Professional",
            motivations="To participate in this focus group",
            challenges="Generic challenges"
        ) for i in range(1, num_participants + 1)]

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
        participant_profiles_for_moderator.append(persona.get_summary())
        
        # Log persona creation
        console.log(f"[blue]Created Agent: {persona_id} - {persona.name}, {persona.age}, {persona.occupation}")

    return participants, participant_profiles_for_moderator

async def run_simulation(moderator: ModeratorAgent, participants: List[ParticipantAgent], num_rounds: int) -> List[DialogueEntry]:
    """Runs the focus group simulation loop with enhanced error handling and real-name streaming."""
    console.log("[bold green]\n--- Starting Focus Group Simulation ---[/bold green]")
    # Also send to stream for UI
    print(f"STREAM: --- Starting Focus Group Simulation ---")
    transcript = []
    current_context = f"The discussion topic is: {moderator.topic}"
    
    # Add participant intros to the context with real names
    for participant in participants:
        profile = participant.persona_profile
        intro = f"{participant.persona_id} is {profile.name}, {profile.age}, {profile.occupation}."
        current_context += f"\n{intro}"
    
    for round_num in range(1, num_rounds + 1):
        console.log(f"[bold cyan]\n--- Round {round_num} ---[/bold cyan]")
        # Also send to stream for UI
        print(f"STREAM: --- Round {round_num} ---")

        # Moderator's turn
        console.log("[yellow]Moderator is thinking...[/yellow]")
        print(f"STREAM: Moderator is thinking...")
        moderator_prompt = f"""Current context:
{current_context}

Based on the discussion so far, what is your next question or statement for round {round_num} of {num_rounds}? 
Remember to use moderator techniques like probing, reflecting, and ensuring all participants are engaged.
If appropriate for this stage, focus on areas that need more exploration or where there are differences of opinion."""

        if round_num == 1:
            moderator_prompt = f"""Current context:
{current_context}

Start the focus group. Introduce yourself as the moderator, welcome the participants, introduce the topic '{moderator.topic}' and explain the format of the discussion. 
Then ask your opening question to the participants ({', '.join(p.persona_profile.name for p in participants)}).
Your first question should be general and open-ended to start the conversation flowing."""

        if round_num == num_rounds:
            moderator_prompt = f"""Current context:
{current_context}

This is the final round of the discussion. Ask a closing question that helps summarize participants' views or gets their final thoughts on the topic. 
Make sure to give each participant an opportunity to share any final insights they haven't expressed yet."""

        try:
            mod_result = await Runner.run(moderator, moderator_prompt)
            moderator_dialogue = mod_result.final_output
            dialogue_entry = DialogueEntry(
                speaker_id="Moderator",
                speaker_name="Moderator",
                content=moderator_dialogue
            )
            console.print(f"[bold magenta]Moderator:[/bold magenta] {moderator_dialogue}")
            # Stream to UI
            print(f"STREAM: Moderator: {moderator_dialogue}")
            transcript.append(dialogue_entry)
            current_context += f"\nModerator: {moderator_dialogue}"
        except Exception as e:
            logger.error(f"Error in moderator response: {e}")
            # Create a fallback response
            fallback_msg = f"Let's move forward with our discussion on {moderator.topic}. What are your thoughts on this?"
            dialogue_entry = DialogueEntry(
                speaker_id="Moderator",
                speaker_name="Moderator",
                content=fallback_msg
            )
            console.print(f"[bold red]Error with moderator. Using fallback:[/bold red] {fallback_msg}")
            # Stream fallback to UI
            print(f"STREAM: Moderator: {fallback_msg}")
            transcript.append(dialogue_entry)
            current_context += f"\nModerator: {fallback_msg}"

        # Participants' turn (sequential response for simplicity, parallel to be added)
        for participant in participants:
            profile = participant.persona_profile
            
            # prompt that references the persona details
            persona_reminder = f"""
Remember that you are {profile.name}, {profile.age} years old, working as {profile.occupation}.
Your attitudes about this topic: {profile.attitudes.opinions or 'Not specified'}
Your beliefs: {profile.attitudes.beliefs or 'Not specified'}
Your behaviors related to this topic: {profile.behaviors.consumption_patterns or 'Not specified'}
Your values: {', '.join(profile.psychographics.values) if profile.psychographics.values else 'Not specified'}
Your interests: {', '.join(profile.psychographics.interests) if profile.psychographics.interests else 'Not specified'}
Your challenges: {profile.challenges or 'Not specified'}
Your location: {profile.location or 'Not specified'}
Your education: {profile.education or 'Not specified'}
Your income level: {profile.income_bracket or 'Not specified'}
"""
            
            participant_prompt = f"""Current context:
{current_context}

{persona_reminder}

Respond to the moderator and the ongoing discussion from your persona's perspective. 
Be authentic to your character's viewpoint and life experiences.
If other participants have spoken, consider responding to or building upon their points if relevant."""

            try:
                part_result = await Runner.run(participant, participant_prompt)
                participant_dialogue = part_result.final_output
                dialogue_entry = DialogueEntry(
                    speaker_id=participant.persona_id,
                    speaker_name=profile.name,
                    content=participant_dialogue
                )
                console.print(f"[bold blue]{profile.name} ({participant.persona_id}):[/bold blue] {participant_dialogue}")
                # Stream to UI
                print(f"STREAM: {profile.name} ({participant.persona_id}): {participant_dialogue}")
                transcript.append(dialogue_entry)
                current_context += f"\n{participant.persona_id} ({profile.name}): {participant_dialogue}"
            except Exception as e:
                logger.error(f"Error in participant {participant.persona_id} response: {e}")
                # Create a fallback response
                fallback_msg = f"I'm interested in this topic from my perspective as {profile.occupation}."
                dialogue_entry = DialogueEntry(
                    speaker_id=participant.persona_id,
                    speaker_name=profile.name,
                    content=fallback_msg
                )
                console.print(f"[bold red]Error with {participant.persona_id}. Using fallback:[/bold red] {fallback_msg}")
                # Stream fallback to UI
                print(f"STREAM: {profile.name} ({participant.persona_id}): {fallback_msg}")
                transcript.append(dialogue_entry)
                current_context += f"\n{participant.persona_id} ({profile.name}): {fallback_msg}"
                
            await asyncio.sleep(0.5) # Small delay to mimic turn-taking and avoid rate limits

    # Moderator conclusion
    console.log("[bold green]\n--- Concluding Simulation ---[/bold green]")
    print(f"STREAM: --- Concluding Simulation ---")
    console.log("[yellow]Moderator is thinking...[/yellow]")
    print(f"STREAM: Moderator is thinking...")
    
    try:
        mod_result = await Runner.run(moderator, f"""Current context:
{current_context}

Conclude the focus group session. Thank the participants for their time and valuable insights.
Provide a brief summary of the key points discussed and the diverse perspectives shared.
End on a positive and appreciative note.""")
        moderator_dialogue = mod_result.final_output
        dialogue_entry = DialogueEntry(
            speaker_id="Moderator",
            speaker_name="Moderator",
            content=moderator_dialogue
        )
        console.print(f"[bold magenta]Moderator:[/bold magenta] {moderator_dialogue}")
        # Stream to UI
        print(f"STREAM: Moderator: {moderator_dialogue}")
        transcript.append(dialogue_entry)
    except Exception as e:
        logger.error(f"Error in moderator conclusion: {e}")
        # Create a fallback conclusion
        fallback_msg = "Thank you all for your participation and valuable insights. This concludes our focus group session."
        dialogue_entry = DialogueEntry(
            speaker_id="Moderator",
            speaker_name="Moderator",
            content=fallback_msg
        )
        console.print(f"[bold red]Error with moderator conclusion. Using fallback:[/bold red] {fallback_msg}")
        # Stream fallback to UI
        print(f"STREAM: Moderator: {fallback_msg}")
        transcript.append(dialogue_entry)

    console.log("[bold green]\n--- Simulation Finished ---[/bold green]")
    print(f"STREAM: --- Simulation Finished ---")
    return transcript

async def analyze_transcript(analyst: AnalystAgent, transcript: List[DialogueEntry], participant_data: List[PersonaProfile]) -> Tuple[str, AnalysisData]:
    """Analyzes the transcript using the AnalystAgent."""
    console.log("[bold green]\n--- Analyzing Transcript ---[/bold green]")
    # Stream to UI
    print(f"STREAM: --- Analyzing Transcript ---")
    
    # Create a formatted transcript with speaker information
    participant_info = {f"Participant_{i+1}": data for i, data in enumerate(participant_data)}
    formatted_transcript = []
    
    # Pre-calculate metrics for sentiment analysis
    word_counts = {}
    response_counts = {}
    interaction_map = {}  # For the knowledge graph
    
    for entry in transcript:
        speaker_id = entry.speaker_id
        content = entry.content
        speaker_name = entry.speaker_name
        
        # Update metrics
        if speaker_id not in word_counts:
            word_counts[speaker_id] = 0
            response_counts[speaker_id] = 0
            interaction_map[speaker_id] = set()
            
        word_counts[speaker_id] += len(content.split())
        response_counts[speaker_id] += 1
        
        # Track who is potentially responding to whom for interaction graph
        # This is a simple heuristic - next:  NLP to detect replies
        if speaker_id != "Moderator":
            # 
            for prev_entry in reversed(transcript[-10:]):  # Look at recent entries
                if prev_entry.speaker_id != speaker_id:
                    interaction_map[speaker_id].add(prev_entry.speaker_id)
                    if prev_entry.speaker_id in interaction_map:
                        interaction_map[prev_entry.speaker_id].add(speaker_id)
                    break
        
        # Format for the analyst
        if speaker_id.startswith("Participant_"):
            profile = participant_info.get(speaker_id, None)
            name = profile.name if profile else speaker_name
            formatted_line = f"{speaker_id} ({name}): {content}"
        else:
            formatted_line = f"{speaker_id}: {content}"
        formatted_transcript.append(formatted_line)
    
    full_transcript = "\n".join(formatted_transcript)
    
    # Create context on participants for the analysis
    participant_context = "Participant Information:\n"
    for pid, profile in participant_info.items():
        participant_context += f"{pid}: {profile.name}, {profile.age}, {profile.occupation}\n"
    
    analysis_prompt = f"""Analyze the following focus group transcript:

{participant_context}

TRANSCRIPT:
{full_transcript}

Apply your expert knowledge in qualitative research analysis to identify key themes, sentiments, participant dynamics, and actionable insights.
Remember to include the structured JSON data for visualization at the end of your analysis.

You MUST calculate sentiment scores by analyzing the content of each statement.
"""
    
    # Replace status with regular log to avoid nested live displays
    console.log("[bold yellow]Performing in-depth analysis of focus group transcript...[/bold yellow]")
    # Stream to UI
    print(f"STREAM: Performing in-depth analysis of focus group transcript...")
    
    try:
        result = await Runner.run(analyst, analysis_prompt)
        analysis = result.final_output
        
        # Extract JSON data for visualization and convert to Pydantic model
        logger.info("Extracting visualization data...")
        analysis_data = AnalysisData.from_json_str(analysis)
        
        # If the analyst didn't include engagement metrics, we'll add our calculated ones
        if not analysis_data.engagement_metrics:
            engagement_metrics = {}
            for speaker_id, word_count in word_counts.items():
                if speaker_id != "Moderator":
                    # Calculate interaction score based on responses and word count
                    interaction_score = min(10, int((word_count / 100) + (len(interaction_map.get(speaker_id, set())) * 1.5)))
                    
                    if speaker_id in participant_info:
                        name = participant_info[speaker_id].name
                    else:
                        name = "Unknown Participant"
                        
                    engagement_metrics[speaker_id] = ParticipantMetrics(
                        word_count=word_count,
                        response_count=response_counts.get(speaker_id, 0),
                        interaction_score=interaction_score
                    )
            analysis_data.engagement_metrics = engagement_metrics
            
        # Ensure all participants have sentiment data
        for pid, profile in participant_info.items():
            if pid not in analysis_data.participant_sentiment:
                # Use TextBlob to calculate sentiment as a fallback
                participant_text = " ".join([entry.content for entry in transcript if entry.speaker_id == pid])
                blob = TextBlob(participant_text)
                sentiment_polarity = blob.sentiment.polarity
                
                # Convert polarity to positive/neutral/negative percentages
                if sentiment_polarity > 0.1:
                    pos = 0.5 + (sentiment_polarity * 0.5)  # Scale to 0.5-1.0
                    neu = 1.0 - pos
                    neg = 0.0
                elif sentiment_polarity < -0.1:
                    neg = 0.5 + (abs(sentiment_polarity) * 0.5)  # Scale to 0.5-1.0
                    neu = 1.0 - neg
                    pos = 0.0
                else:
                    pos = (sentiment_polarity + 0.1) * 5  # Scale to 0-0.5
                    neg = (abs(sentiment_polarity - 0.1)) * 5  # Scale to 0-0.5
                    neu = 1.0 - (pos + neg)
                    
                analysis_data.participant_sentiment[pid] = ParticipantSentiment(
                    positive=pos,
                    neutral=neu,
                    negative=neg,
                    name=profile.name
                )
        
        # Remove JSON section from the analysis text
        json_match = re.search(r'```json\s*(.*?)\s*```', analysis, re.DOTALL)
        if json_match:
            analysis = analysis.replace(json_match.group(0), '')
            
        console.log("[green]Analysis Complete.")
        # Stream to UI
        print(f"STREAM: Analysis Complete.")
        return analysis.strip(), analysis_data
        
    except Exception as e:
        logger.error(f"Error in transcript analysis: {e}")
        logger.error(traceback.format_exc())
        
        # Use TextBlob as fallback for simple sentiment analysis
        console.log("[bold red]Error in analysis. Using fallback sentiment analysis.")
        # Stream to UI
        print(f"STREAM: Error in analysis. Using fallback sentiment analysis.")
        
        default_themes = [
            ThemeData(theme="General Discussion", frequency=1.0, description="Various aspects of the topic were discussed"),
        ]
        
        # Calculate basic sentiment using TextBlob
        full_text = " ".join([entry.content for entry in transcript])
        blob = TextBlob(full_text)
        sentiment_polarity = blob.sentiment.polarity
        
        # Convert polarity to positive/neutral/negative percentages
        if sentiment_polarity > 0.1:
            pos = 0.5 + (sentiment_polarity * 0.5)  # Scale to 0.5-1.0
            neu = 1.0 - pos
            neg = 0.0
        elif sentiment_polarity < -0.1:
            neg = 0.5 + (abs(sentiment_polarity) * 0.5)  # Scale to 0.5-1.0
            neu = 1.0 - neg
            pos = 0.0
        else:
            pos = (sentiment_polarity + 0.1) * 5  # Scale to 0-0.5
            neg = (abs(sentiment_polarity - 0.1)) * 5  # Scale to 0-0.5
            neu = 1.0 - (pos + neg)
            
        sentiment_breakdown = SentimentScore(
            positive=pos,
            neutral=neu,
            negative=neg
        )
            
        # Process individual participant sentiment
        participant_sentiment = {}
        for pid, profile in participant_info.items():
            participant_text = " ".join([entry.content for entry in transcript if entry.speaker_id == pid])
            if participant_text:
                blob = TextBlob(participant_text)
                sentiment_polarity = blob.sentiment.polarity
                
                if sentiment_polarity > 0.1:
                    pos = 0.5 + (sentiment_polarity * 0.5)
                    neu = 1.0 - pos
                    neg = 0.0
                elif sentiment_polarity < -0.1:
                    neg = 0.5 + (abs(sentiment_polarity) * 0.5)
                    neu = 1.0 - neg
                    pos = 0.0
                else:
                    pos = (sentiment_polarity + 0.1) * 5
                    neg = (abs(sentiment_polarity - 0.1)) * 5
                    neu = 1.0 - (pos + neg)
            else:
                pos, neu, neg = 0.33, 0.34, 0.33
                
            participant_sentiment[pid] = ParticipantSentiment(
                positive=pos,
                neutral=neu,
                negative=neg,
                name=profile.name
            )
            
        # Create engagement metrics
        engagement_metrics = {}
        for speaker_id, word_count in word_counts.items():
            if speaker_id != "Moderator":
                # Basic interaction score calculation
                interaction_score = min(10, int((word_count / 100) + (len(interaction_map.get(speaker_id, set())) * 1.5)))
                
                engagement_metrics[speaker_id] = ParticipantMetrics(
                    word_count=word_count,
                    response_count=response_counts.get(speaker_id, 0),
                    interaction_score=interaction_score
                )
        
        # Create a fallback analysis data object
        analysis_data = AnalysisData(
            sentiment_breakdown=sentiment_breakdown,
            key_themes=default_themes,
            participant_sentiment=participant_sentiment,
            engagement_metrics=engagement_metrics
        )
        
        fallback_analysis = """
# Focus Group Analysis (Fallback Report)

## Executive Summary
This focus group discussed the topic with varying perspectives. Due to processing limitations, a simplified analysis is provided.

## Key Themes
The discussion covered general aspects of the topic.

## Sentiment Analysis
The overall sentiment was mixed, with participants expressing a range of opinions.

## Participant Dynamics
Different levels of engagement were observed among participants.

## Recommendations
Further analysis is recommended to draw more specific conclusions.
"""
        
        return fallback_analysis, analysis_data

def generate_report(analysis: str, parameters: Dict[str, Any], transcript: List[DialogueEntry], 
                   analysis_data: AnalysisData, participant_profiles: List[PersonaProfile], simulation_id: str):
    """Generates and saves the final report with enhanced visualizations including knowledge graph."""
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.log(f"[bold green]--- Generating Report for Simulation ID: {simulation_id} ---[/bold green]")

    # Define the base output directory within the UI's public folder
    # Assumes the script is run from the root of the openai-agents-simulation directory
    base_output_dir = os.path.join("openai-simulations-ui", "public", "simulations", "focus-group", simulation_id)
    viz_rel_path_prefix = f"/simulations/focus-group/{simulation_id}/visualizations" # Relative path for markdown links
    viz_abs_dir = os.path.join(base_output_dir, "visualizations")

    # Create directories if they don't exist
    os.makedirs(viz_abs_dir, exist_ok=True)
    console.log(f"[blue]Output directory created/ensured: {base_output_dir}")

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
    for i, persona in enumerate(participant_profiles):
        pid = f"Participant_{i+1}"
        report_content += f"### {pid}: {persona.name}\n"
        report_content += f"- **Age:** {persona.age}\n"
        report_content += f"- **Occupation:** {persona.occupation}\n"
        report_content += f"- **Education:** {persona.education or 'Unknown'}\n"
        report_content += f"- **Demographics:** Ethnicity: {persona.demographics.ethnicity}, Marital Status: {persona.demographics.marital_status}\n"
        
        # Format psychographics data
        values_str = ", ".join(persona.psychographics.values) if persona.psychographics.values else "None specified"
        interests_str = ", ".join(persona.psychographics.interests) if persona.psychographics.interests else "None specified"
        report_content += f"- **Values:** {values_str}\n"
        report_content += f"- **Interests:** {interests_str}\n\n"
    
    report_content += f"""
## 3. Analysis Results
{analysis}

## 4. Visualizations
*See attached visualization files referenced below*

"""
    
    # Generate visualizations from the AnalysisData
    visualization_files = []
    console.log(f"[yellow]Generating visualizations in {viz_abs_dir}...[/yellow]")
        
    try:
        # 1. Sentiment Breakdown Pie Chart
        plt.figure(figsize=(10, 7))
        sentiment_data = analysis_data.sentiment_breakdown
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [sentiment_data.positive, sentiment_data.neutral, sentiment_data.negative]
        colors = ['#66b3ff', '#99ff99', '#ff9999']
        explode = (0.1, 0, 0)  # explode the 1st slice (positive)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.title('Overall Sentiment Distribution', fontsize=16)
        
        pie_file = os.path.join(viz_abs_dir, "sentiment_pie_chart.png")
        plt.savefig(pie_file, dpi=300)
        plt.close()
        visualization_files.append(os.path.basename(pie_file))
        report_content += f"![Sentiment Distribution]({viz_rel_path_prefix}/sentiment_pie_chart.png)\n\n"
        
        # 2. Key Themes Bar Chart
        if analysis_data.key_themes:
            plt.figure(figsize=(12, 8))
            themes = [item.theme for item in analysis_data.key_themes]
            frequencies = [item.frequency for item in analysis_data.key_themes]
            
            # 
            sns.set_style("whitegrid")
            ax = sns.barplot(x=frequencies, y=themes, hue=themes, palette="viridis", legend=False)
            
            # Add data labels
            for i, freq in enumerate(frequencies):
                ax.text(freq + 0.05, i, f'{freq:.2f}', va='center')
            
            plt.xlabel('Relative Frequency', fontsize=14)
            plt.ylabel('Themes', fontsize=14)
            plt.title('Key Discussion Themes', fontsize=16)
            plt.tight_layout()
            
            themes_file = os.path.join(viz_abs_dir, "key_themes_chart.png")
            plt.savefig(themes_file, dpi=300)
            plt.close()
            visualization_files.append(os.path.basename(themes_file))
            report_content += f"![Key Themes]({viz_rel_path_prefix}/key_themes_chart.png)\n\n"
        
        # 3. Participant Sentiment Comparison (Stacked Bar Chart)
        if analysis_data.participant_sentiment:
            participant_data = analysis_data.participant_sentiment
            participants = list(participant_data.keys())
            
            # Get names for display
            participant_names = []
            for p in participants:
                name = participant_data[p].name
                participant_names.append(f"{p}\n({name})")
            
            positive_values = [participant_data[p].positive for p in participants]
            neutral_values = [participant_data[p].neutral for p in participants]
            negative_values = [participant_data[p].negative for p in participants]
            
            plt.figure(figsize=(14, 9))
            
            # Create stacked bar chart
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
            
        # 4. Engagement Metrics Bar Chart (improved)
        if analysis_data.engagement_metrics:
            engagement_data = analysis_data.engagement_metrics
            participants = list(engagement_data.keys())
            
            # Get names for labels
            participant_labels = []
            for p in participants:
                if p in analysis_data.participant_sentiment:
                    name = analysis_data.participant_sentiment[p].name
                    participant_labels.append(f"{p}\n({name})")
                else:
                    participant_labels.append(p)
            
            # 
            metrics_df = {
                'Participant': [],
                'Metric': [],
                'Value': [],
                'Scaled Value': []  # For visualization purposes
            }
            
            # Get max values for scaling
            max_word_count = max([engagement_data[p].word_count for p in participants])
            max_response_count = max([engagement_data[p].response_count for p in participants])
            max_interaction = max([engagement_data[p].interaction_score for p in participants])
            
            # Build dataframe for seaborn
            for p in participants:
                # Word count (scaled for visualization)
                metrics_df['Participant'].append(participant_labels[participants.index(p)])
                metrics_df['Metric'].append('Word Count')
                metrics_df['Value'].append(engagement_data[p].word_count)
                metrics_df['Scaled Value'].append(engagement_data[p].word_count / max_word_count * 10)
                
                # Response count
                metrics_df['Participant'].append(participant_labels[participants.index(p)])
                metrics_df['Metric'].append('Response Count')
                metrics_df['Value'].append(engagement_data[p].response_count)
                metrics_df['Scaled Value'].append(engagement_data[p].response_count / max_response_count * 10)
                
                # Interaction score
                metrics_df['Participant'].append(participant_labels[participants.index(p)])
                metrics_df['Metric'].append('Interaction Score')
                metrics_df['Value'].append(engagement_data[p].interaction_score)
                metrics_df['Scaled Value'].append(engagement_data[p].interaction_score / max_interaction * 10)
            
            plt.figure(figsize=(14, 10))
            sns.set_style("whitegrid")
            
            # Plot grouped bar chart with seaborn
            ax = sns.barplot(x='Participant', y='Scaled Value', hue='Metric', 
                          data=metrics_df, palette="deep")
            
            # Add value labels on the bars - Fix the indexing error
            try:
                # Only add labels if there are patches created
                if ax.patches:
                    # Create a mapping between patch index and original values
                    value_index = 0
                    for i, p in enumerate(ax.patches):
                        if value_index < len(metrics_df['Value']):
                            height = p.get_height()
                            orig_value = metrics_df['Value'][value_index]
                            ax.text(p.get_x() + p.get_width()/2., height + 0.1, f'{orig_value}', 
                                  ha="center", fontsize=9)
                            value_index += 1
            except Exception as e:
                logger.warning(f"Could not add value labels to engagement metrics chart: {e}")
            
            plt.title('Participant Engagement Metrics', fontsize=16)
            plt.ylabel('Scaled Value (0-10)', fontsize=14)
            plt.xlabel('Participants', fontsize=14)
            plt.legend(title='Metric', fontsize=12)
            plt.tight_layout()
            
            engagement_file = os.path.join(viz_abs_dir, "engagement_metrics.png")
            plt.savefig(engagement_file, dpi=300)
            plt.close()
            visualization_files.append(os.path.basename(engagement_file))
            report_content += f"![Engagement Metrics]({viz_rel_path_prefix}/engagement_metrics.png)\n\n"
        
        # 5. Knowledge Graph of Interactions
        # Create a network graph visualization of interactions between participants
        interaction_graph = nx.Graph()
        
        # Extract participant names for node labels
        participant_names = {}
        for i, profile in enumerate(participant_profiles):
            pid = f"Participant_{i+1}"
            participant_names[pid] = profile.name
        participant_names["Moderator"] = "Moderator"
        
        # Create a more structured conversation model for better visualization
        # Moderator connects to all participants, participants connect based on reply patterns
        interaction_counts = {}
        
        # First add all nodes
        for pid in participant_names:
            interaction_graph.add_node(pid)
        
        # Track interactions between each pair of speakers
        for i in range(len(transcript) - 1):
            current_speaker = transcript[i].speaker_id
            next_speaker = transcript[i+1].speaker_id
            
            if current_speaker != next_speaker:  # Only count transitions between different speakers
                pair = tuple(sorted([current_speaker, next_speaker]))  # Sort to create unique key
                if pair not in interaction_counts:
                    interaction_counts[pair] = 0
                interaction_counts[pair] += 1
        
        # Add weighted edges based on interaction counts
        for pair, count in interaction_counts.items():
            if count > 0:  # Only add edges with actual interactions
                interaction_graph.add_edge(pair[0], pair[1], weight=count)
        
        # Create an interactive network graph with improved styling
        # Create a circular layout first to position nodes in a circle
        pos = nx.circular_layout(interaction_graph)
        
        # Save positions for each node
        node_positions = {}
        for node, position in pos.items():
            # Scale positions to pixel values (vis.js expects pixel values)
            x, y = position * 500  # Scale factor to make layout larger
            node_positions[node] = (float(x), float(y))
        
        # Create network with larger dimensions for better visibility
        net = Network(height="800px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=False)
        
        # 
        net.barnes_hut(
            gravity=-15000,           # Less negative gravity for more compact layout
            central_gravity=0.5,      # Stronger central gravity to pull nodes together
            spring_length=150,        # Shorter spring length for connections
            spring_strength=0.05,     # Stronger springs for more structured layout
            damping=0.09,
            overlap=0.1               # Allow slight overlap for denser layout
        )
        
        # Create a custom color scheme for node types
        moderator_color = "#EB5757"  # Red for moderator
        sentiment_colors = {
            "positive": "#27AE60",    # Green for positive sentiment
            "neutral": "#2F80ED",     # Blue for neutral sentiment
            "negative": "#F2994A"     # Orange for negative sentiment
        }
        
        # Add nodes with improved styling and fixed positions
        for node in interaction_graph.nodes():
            if node == "Moderator":
                net.add_node(
                    node, 
                    label="Moderator", 
                    title="Focus Group Moderator", 
                    color=moderator_color, 
                    size=35,
                    shape="star",
                    physics=False,  # Disable physics for fixed position
                    x=node_positions[node][0],
                    y=node_positions[node][1]
                )
            else:
                # Determine sentiment-based color
                if node in analysis_data.participant_sentiment:
                    sentiment = analysis_data.participant_sentiment[node]
                    
                    # Choose dominant sentiment
                    dominant = max(
                        ("positive", sentiment.positive), 
                        ("neutral", sentiment.neutral), 
                        ("negative", sentiment.negative), 
                        key=lambda x: x[1]
                    )[0]
                    
                    color = sentiment_colors[dominant]
                    
                    # Get engagement for node size
                    if node in analysis_data.engagement_metrics:
                        # Base size on word count to make more active participants larger
                        word_count = analysis_data.engagement_metrics[node].word_count
                        response_count = analysis_data.engagement_metrics[node].response_count
                        size = 20 + min(30, (word_count / 50)) + (response_count * 2)
                    else:
                        size = 25
                else:
                    color = "#97C2FC"  # Default blue
                
                label = participant_names[node]
                
                # Create detailed title/tooltip
                title = f"{label} ({node})"
                if node in analysis_data.participant_sentiment:
                    sentiment = analysis_data.participant_sentiment[node]
                    title += f"<br>Sentiment: {dominant.capitalize()}"
                
                if node in analysis_data.engagement_metrics:
                    metrics = analysis_data.engagement_metrics[node]
                    title += f"<br>Words: {metrics.word_count}, Responses: {metrics.response_count}"
                
                net.add_node(
                    node, 
                    label=label, 
                    title=title, 
                    color=color, 
                    size=size,
                    shape="dot",
                    physics=False,  # Disable physics for fixed position
                    x=node_positions[node][0],
                    y=node_positions[node][1]
                )
        
        # Add edges with improved styling based on interaction frequency
        max_weight = max(data.get('weight', 1) for _, _, data in interaction_graph.edges(data=True))
        
        for edge in interaction_graph.edges(data=True):
            source, target, data = edge
            weight = data.get('weight', 1)
            
            # Scale edge width based on relative interaction frequency
            # Min width 1, max width 10
            scaled_width = 1 + ((weight / max_weight) * 9)
            
            # Determine the total number of interactions
            interaction_count = data.get('weight', 1)
            
            # Create detailed edge title
            title = f"{participant_names[source]} ⟷ {participant_names[target]}: {interaction_count} interactions"
            
            # Add edge with custom styling
            net.add_edge(
                source, 
                target, 
                value=scaled_width, 
                title=title,
                color={"color": "#555555", "opacity": 0.8},  # Gray with slight transparency
                arrowStrikethrough=False,  # Better arrow appearance
                smooth={"type": "continuous"}  # Smoother curves
            )
        
        # Add network options for better visualization
        net.set_options("""
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": {
              "enabled": true
            }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            }
          },
          "physics": {
            "stabilization": {
              "iterations": 200,
              "fit": true
            }
          }
        }
        """)
        
        # Generate network
        knowledge_graph_file = os.path.join(viz_abs_dir, "knowledge_graph.html")
        net.save_graph(knowledge_graph_file)
        visualization_files.append(os.path.basename(knowledge_graph_file))
        report_content += f"[Knowledge Graph of Interactions]({viz_rel_path_prefix}/knowledge_graph.html) (Interactive visualization)\n\n"
        
        # Also generate a static PNG version for embedding in reports
        try:
            # Create a static version using matplotlib
            plt.figure(figsize=(12, 10))
            
            # Create a nice layout for the graph
            pos = nx.spring_layout(interaction_graph, k=0.5, iterations=50)
            
            # Get edge weights for line thickness
            edge_weights = [data.get('weight', 1) * 2 for _, _, data in interaction_graph.edges(data=True)]
            
            # Create node groups for coloring
            moderator_nodes = [n for n in interaction_graph.nodes() if n == "Moderator"]
            participant_nodes = [n for n in interaction_graph.nodes() if n != "Moderator"]
            
            # Draw the nodes with different colors based on role
            nx.draw_networkx_nodes(interaction_graph, pos, 
                                  nodelist=moderator_nodes, 
                                  node_color="#EB5757", 
                                  node_size=800,
                                  alpha=0.9)
            
            # Draw participant nodes with sentiment-based colors
            participant_colors = []
            for node in participant_nodes:
                if node in analysis_data.participant_sentiment:
                    sentiment = analysis_data.participant_sentiment[node]
                    # Simple RGB blend based on sentiment
                    r = int(255 * sentiment.negative)
                    g = int(255 * sentiment.positive)
                    b = int(255 * sentiment.neutral)
                    participant_colors.append(f"#{r:02x}{g:02x}{b:02x}")
                else:
                    participant_colors.append("#97C2FC")  # Default blue
            
            nx.draw_networkx_nodes(interaction_graph, pos, 
                                  nodelist=participant_nodes, 
                                  node_color=participant_colors,
                                  node_size=600,
                                  alpha=0.8)
            
            # Draw the edges with weight-based thickness
            nx.draw_networkx_edges(interaction_graph, pos, 
                                  width=edge_weights,
                                  alpha=0.7, 
                                  edge_color='gray')
            
            # Add labels with participant names
            nx.draw_networkx_labels(interaction_graph, pos, 
                                   labels={n: participant_names.get(n, n) for n in interaction_graph.nodes()},
                                   font_size=10, 
                                   font_weight='bold')
            
            plt.title("Focus Group Interaction Network", fontsize=16)
            plt.axis('off')  # Turn off axis
            
            # Save static image
            static_graph_file = os.path.join(viz_abs_dir, "knowledge_graph.png")
            plt.savefig(static_graph_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(os.path.basename(static_graph_file))
            report_content += f"![Knowledge Graph (Static Version)]({viz_rel_path_prefix}/knowledge_graph.png)\n\n"
        except Exception as e:
            logger.warning(f"Could not create static knowledge graph image: {e}")
            # Continue even if static image fails
        
        # 6. Sentiment Analysis Over Time Visualization
        # time-series visualization of sentiment
        speaker_order = []
        sentiments = []
        speaker_names = []
        
        # Calculate sentiment for each dialogue entry
        for entry in transcript:
            speaker_order.append(entry.speaker_id)
            speaker_names.append(entry.speaker_name)
            
            # Calculate sentiment with TextBlob
            blob = TextBlob(entry.content)
            sentiments.append(blob.sentiment.polarity)
        
        # Create a time series visualization
        plt.figure(figsize=(15, 8))
        
        # Create a colormap for different speakers
        unique_speakers = list(set(speaker_order))
        cmap = plt.cm.get_cmap('tab10', len(unique_speakers))
        speaker_colors = {speaker: cmap(i) for i, speaker in enumerate(unique_speakers)}
        
        # Plot points with different colors for different speakers
        for i, (speaker, sentiment) in enumerate(zip(speaker_order, sentiments)):
            plt.scatter(i, sentiment, color=speaker_colors[speaker], s=100, 
                      label=participant_names[speaker] if speaker not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # Connect points from the same speaker with a dotted line
            if i > 0 and speaker == speaker_order[i-1]:
                plt.plot([i-1, i], [sentiments[i-1], sentiment], 'k--', alpha=0.3)
        
        # Add labels and annotations
        for i, (speaker, sentiment, name) in enumerate(zip(speaker_order, sentiments, speaker_names)):
            # Add speaker name for every 5th point to avoid crowding
            if i % 5 == 0:
                plt.annotate(name, (i, sentiment), 
                           textcoords="offset points", 
                           xytext=(0, 10), 
                           ha='center',
                           fontsize=8)
        
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.ylim(-1.1, 1.1)
        plt.title('Sentiment Progression Throughout the Focus Group', fontsize=16)
        plt.xlabel('Sequential Dialogue Order', fontsize=14)
        plt.ylabel('Sentiment Polarity (-1 to 1)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Create a custom legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=speaker_colors[speaker], 
                                   label=participant_names[speaker], markersize=10) 
                                 for speaker in unique_speakers]
        plt.legend(handles=legend_elements, title="Participants", loc='upper center', 
                 bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
        
        plt.tight_layout()
        
        sentiment_time_file = os.path.join(viz_abs_dir, "sentiment_progression.png")
        plt.savefig(sentiment_time_file, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files.append(os.path.basename(sentiment_time_file))
        report_content += f"![Sentiment Progression]({viz_rel_path_prefix}/sentiment_progression.png)\n\n"
        
        console.log(f"[green]Generated {len(visualization_files)} visualization files.")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        logger.error(traceback.format_exc())
        report_content += "**Error generating some visualizations. See console for details.**\n\n"
    
    report_content += """
## 5. Full Transcript
"""
    for entry in transcript:
        report_content += f"- **{entry.speaker_name} ({entry.speaker_id}):** {entry.content}\n\n"

    # Save the report to a file with timestamp
    report_filename = os.path.join(base_output_dir, "report.md")
    with open(report_filename, "w", encoding='utf-8') as f:
        f.write(report_content)
    console.log(f"[green]Report saved successfully as: {report_filename}")
    
    # Also save the transcript as a CSV file
    transcript_filename = os.path.join(base_output_dir, "transcript.csv")
    with open(transcript_filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker ID", "Speaker Name", "Dialogue", "Timestamp"])
        for entry in transcript:
            writer.writerow([entry.speaker_id, entry.speaker_name, entry.content, entry.timestamp])
    console.log(f"[green]Transcript saved successfully as: {transcript_filename}")

    return visualization_files

# --- Main Execution ---

async def main():
    """Main function to run the focus group simulation."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            console.log("[bold red]Error: OPENAI_API_KEY environment variable not set.")
            console.log("Please create a .env file with OPENAI_API_KEY=your_key")
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
        
        console.log("\n[bold cyan]--- Focus Group Parameters ---[/bold cyan]")
        console.log(f"Topic: {params['topic']}")
        console.log(f"Target Audience: {params['target_audience']}")
        console.log(f"Participants: {params['num_participants']}")
        console.log(f"Rounds: {params['num_rounds']}")
        console.log("[bold cyan]-----------------------------[/bold cyan]")

        # 
        console.log("[bold green]Setting up simulation...[/bold green]")
        participants, participant_profiles = await create_participant_agents(
            params["target_audience"], params["num_participants"], params["topic"]
        )

        # Get the persona profiles for reporting
        participant_profiles = [p.persona_profile for p in participants]

        # Ensure we have participants before creating the moderator that needs their profiles
        if not participants:
            console.log("[bold red]Error: Failed to create participant agents. Exiting.")
            return

        moderator = ModeratorAgent(
            topic=params["topic"],
            participant_profiles=participant_profiles,
            num_rounds=params["num_rounds"]
        )
        analyst = AnalystAgent(topic=params["topic"], target_audience=params["target_audience"])

        # Run the simulation
        transcript = await run_simulation(moderator, participants, params["num_rounds"])

        if transcript:
            console.log("[bold green]Simulation completed successfully. Analyzing results...[/bold green]")
            analysis, analysis_data = await analyze_transcript(analyst, transcript, participant_profiles)
            
            console.log("[bold green]Analysis completed. Generating report and visualizations...[/bold green]")
            visualization_files = generate_report(analysis, params, transcript, analysis_data, participant_profiles, simulation_id)
            
            console.log("[bold green]Focus group simulation completed successfully![/bold green]")
            console.log(f"Report and {len(visualization_files)} visualizations saved to: openai-simulations-ui/public/simulations/focus-group/{simulation_id}/")
        else:
            console.log("[bold red]Simulation did not produce a transcript. Analysis skipped.")
            
    except Exception as e:
        logger.error(f"Uncaught exception in main: {e}")
        logger.error(traceback.format_exc())
        console.log(f"[bold red]Error running focus group simulation: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())