"""
Mock Agent implementation for testing survey simulation functions.
"""

from typing import Dict, Any, List

class Agent:
    """Mock version of Agent class for testing."""
    
    def __init__(self, name: str, instructions: str, model: str = "gpt-4o"):
        self.name = name
        self.instructions = instructions
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """
        Generate a mock response based on the agent type.
        
        Args:
            prompt: The prompt to send to the agent
            
        Returns:
            A mock response for testing
        """
        print(f"Mock Agent '{self.name}' processing prompt... (model: {self.model})")
        
        # Simple mock responses for testing different agent types
        if self.name == "Survey Designer":
            return '''```json
{
  "survey_title": "Electric Vehicle Adoption Survey",
  "introduction": "Thank you for participating in our survey about electric vehicles. Your feedback will help us understand barriers to EV adoption.",
  "sections": [
    {
      "section_name": "Demographics",
      "questions": [
        {
          "question_id": "Q1",
          "question_text": "What is your age?",
          "question_type": "multiple_choice",
          "options": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
          "required": true
        },
        {
          "question_id": "Q2",
          "question_text": "What is your gender?",
          "question_type": "multiple_choice",
          "options": ["Male", "Female", "Non-binary", "Prefer not to say"],
          "required": true
        }
      ]
    },
    {
      "section_name": "Current Vehicle Usage",
      "questions": [
        {
          "question_id": "Q3",
          "question_text": "How many vehicles do you currently own?",
          "question_type": "multiple_choice",
          "options": ["0", "1", "2", "3+"],
          "required": true
        },
        {
          "question_id": "Q4",
          "question_text": "What type of vehicle do you primarily drive?",
          "question_type": "multiple_choice",
          "options": ["Sedan", "SUV", "Truck", "Van", "Compact", "Other"],
          "required": true
        }
      ]
    },
    {
      "section_name": "EV Awareness",
      "questions": [
        {
          "question_id": "Q5",
          "question_text": "How familiar are you with electric vehicles?",
          "question_type": "likert_scale",
          "options": ["Not at all familiar", "Slightly familiar", "Moderately familiar", "Very familiar", "Extremely familiar"],
          "required": true
        },
        {
          "question_id": "Q6",
          "question_text": "Have you ever driven an electric vehicle?",
          "question_type": "multiple_choice",
          "options": ["Yes", "No"],
          "required": true
        }
      ]
    }
  ]
}```'''
        elif self.name == "Respondent Generator":
            return '''```json
[
  {
    "respondent_id": "R001",
    "age": 34,
    "gender": "Male",
    "ethnicity": "Caucasian",
    "location": "Boston, MA",
    "occupation": "Software Engineer",
    "education": "Master's Degree",
    "income_bracket": "$100,000-$149,999",
    "marital_status": "Married",
    "household_composition": {
      "household_size": 3,
      "has_children": true,
      "number_of_children": 1,
      "child_ages": [3],
      "home_ownership": "Homeowner"
    },
    "psychographics": {
      "interests": ["Technology", "Outdoor activities", "Home automation", "Sustainability"],
      "personality_traits": ["Analytical", "Early adopter"],
      "technology_adoption": ["Early adopter"],
      "shopping_behavior": ["Researcher", "Brand loyal"],
      "media_preferences": ["YouTube", "LinkedIn", "Podcasts"],
      "values": "Environmentally conscious"
    },
    "response_style": "thoughtful",
    "decision_making": "Analytical",
    "brand_preferences": {
      "technology": "Apple",
      "grocery": "Whole Foods",
      "clothing": "Sustainable brands"
    }
  },
  {
    "respondent_id": "R002",
    "age": 42,
    "gender": "Female",
    "ethnicity": "African American",
    "location": "Chicago, IL",
    "occupation": "Marketing Director",
    "education": "Bachelor's Degree",
    "income_bracket": "$75,000-$99,999",
    "marital_status": "Divorced",
    "household_composition": {
      "household_size": 3,
      "has_children": true,
      "number_of_children": 2,
      "child_ages": [10, 12],
      "home_ownership": "Homeowner"
    },
    "psychographics": {
      "interests": ["Fitness", "Reading", "Travel", "Fashion"],
      "personality_traits": ["Extroverted", "Conscientious"],
      "technology_adoption": ["Early majority"],
      "shopping_behavior": ["Price-conscious", "Deal hunter"],
      "media_preferences": ["Instagram", "Facebook", "Traditional TV"],
      "values": "Family-oriented"
    },
    "response_style": "rushed",
    "decision_making": "Emotional",
    "brand_preferences": {
      "technology": "Samsung",
      "grocery": "Target",
      "clothing": "Department stores"
    }
  }
]```'''
        elif self.name in ["Survey Respondent", "Respondent"]:
            return '''```json
[
  {
    "question_id": "Q1",
    "response": "25-34"
  },
  {
    "question_id": "Q2",
    "response": "Male"
  },
  {
    "question_id": "Q3",
    "response": "1"
  },
  {
    "question_id": "Q4",
    "response": "Sedan"
  },
  {
    "question_id": "Q5",
    "response": "Moderately familiar"
  },
  {
    "question_id": "Q6",
    "response": "No"
  }
]```'''
        elif self.name in ["Survey Analyst", "IDI Analyst", "Focus Group Analyst"]:
            return '''This is an analysis response from the analyst agent.

Here are some key findings from the survey:
1. Most respondents have some familiarity with EVs but have not driven one
2. Cost concerns are the primary barrier to adoption
3. Environmental benefits are the most appealing aspect

```json
{
  "sentiment_breakdown": {
    "positive": 55,
    "neutral": 35,
    "negative": 10
  },
  "key_themes": [
    {"theme": "Cost Concerns", "frequency": 8, "sentiment": "negative", "description": "High upfront cost is a barrier"},
    {"theme": "Environmental Benefits", "frequency": 6, "sentiment": "positive", "description": "Respondents value eco-friendly aspects"},
    {"theme": "Charging Infrastructure", "frequency": 4, "sentiment": "neutral", "description": "Questions about charging availability"}
  ]
}```'''
        else:
            return f"This is a mock response from the {self.name} agent. In a real implementation, this would be the output from the OpenAI API."

class SurveyDesignerAgent(Agent):
    """Mock implementation of SurveyDesignerAgent."""
    def __init__(self, topic: str, research_objectives: str, target_audience: str):
        super().__init__(
            name="Survey Designer",
            instructions=f"Design a survey about {topic}",
            model="gpt-4o"
        )
        self.topic = topic
        self.research_objectives = research_objectives
        self.target_audience = target_audience

class RespondentGeneratorAgent(Agent):
    """Mock implementation of RespondentGeneratorAgent."""
    def __init__(self, target_audience: str, num_respondents: int = 5):
        super().__init__(
            name="Respondent Generator",
            instructions=f"Generate {num_respondents} respondents for {target_audience}",
            model="gpt-4o"
        )
        self.target_audience = target_audience
        self.num_respondents = num_respondents

class SurveyResponseAgent(Agent):
    """Mock implementation of SurveyResponseAgent."""
    def __init__(self, survey_data: Dict[str, Any], respondent_data: Dict[str, Any], topic: str):
        super().__init__(
            name="Survey Respondent",
            instructions=f"Respond to survey about {topic}",
            model="gpt-4o"
        )
        self.survey_data = survey_data
        self.respondent_data = respondent_data
        self.topic = topic

class SurveyAnalystAgent(Agent):
    """Mock implementation of SurveyAnalystAgent."""
    def __init__(self, topic: str, research_objectives: str, target_audience: str):
        super().__init__(
            name="Survey Analyst",
            instructions=f"Analyze survey data about {topic}",
            model="gpt-4o"
        )
        self.topic = topic
        self.research_objectives = research_objectives
        self.target_audience = target_audience

# Mock Runner class
class Runner:
    """Mock implementation of Runner for testing."""
    
    @staticmethod
    async def run(agent, prompt, context=None):
        """
        Mock run method that returns a result object with the agent's response.
        """
        response = agent.generate(prompt)
        
        # Create a simple result object
        class Result:
            def __init__(self, output):
                self.final_output = output
                
            def final_output_as(self, output_type):
                # This would normally convert to the output type
                return {"result": self.final_output}
        
        return Result(response) 