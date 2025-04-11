# In-Depth Interview (IDI) Simulation

An advanced In-Depth Interview (IDI) simulation tool powered by OpenAI's GPT models, designed to produce professional-grade qualitative research insights and reports.

## Overview

This tool simulates a complete one-on-one in-depth interview. It dynamically generates a detailed respondent persona based on a target audience, simulates a conversation between an expert AI interviewer and the AI respondent, analyzes the resulting transcript for themes and sentiment, and generates a comprehensive report with visualizations.

## Features

-   **Dynamic Persona Generation**: Creates a single, rich, and realistic respondent persona (demographics, psychographics, behaviors, attitudes) tailored to the target audience and topic.
-   **Expert AI Interviewer**: Simulates an interviewer using qualitative research best practices (rapport building, probing, active listening, funnel questioning).
-   **Authentic AI Respondent**: The respondent agent answers consistently based on its detailed, assigned persona.
-   **Natural Conversation Flow**: Generates realistic back-and-forth dialogue, including introductions, probing follow-ups, and conclusions.
-   **In-depth Qualitative Analysis**: An AI analyst agent performs thematic analysis, sentiment analysis (overall and per turn), and identifies key quotes and insights from the transcript.
-   **Professional Reporting**: Generates a detailed Markdown report including an executive summary, methodology, respondent profile, key themes, sentiment analysis, actionable insights, and the full transcript.
-   **Data Visualizations**: Creates charts (pie, bar, line) visualizing sentiment breakdown, key themes, question types, and sentiment flow using Matplotlib.
-   **Structured Output**: Saves the persona, transcript (CSV), report, and visualizations in a dedicated simulation folder.

## Prerequisites

-   Python 3.8+
-   OpenAI API Key

## Installation

1.  Clone the repository.
2.  Install the required dependencies. Assuming `openai-agents` is a custom library:
    ```bash
    pip install python-dotenv matplotlib numpy
    # Ensure the 'agents' module/library is accessible in your Python path
    ```
3.  Create a `.env` file in the project root directory (or where the script can find it) with your OpenAI API key:
    ```
    OPENAI_API_KEY="your_api_key_here"
    ```

## Usage

Run the IDI simulation script from the command line, providing the necessary arguments:

```bash
python idi_simulation/idi_simulation.py \
    --topic "Your Interview Topic" \
    --target_audience "Description of Your Target Audience" \
    --num_questions 8 \
    --simulation_id "unique_simulation_run_id"
```

**Command-Line Arguments:**

-   `--topic` (string, optional, default: `"Mobile banking app user experience"`): The central subject of the interview.
-   `--target_audience` (string, optional, default: `"Urban professionals aged 25-40"`): A description of the desired respondent profile.
-   `--num_questions` (integer, optional, default: `8`): The target number of main questions the interviewer should aim to ask during the core interview phase (excluding intro/outro).
-   `--simulation_id` (string, **required**): A unique identifier for this specific simulation run. This ID determines the output folder name.

## Simulation Pipeline

1.  **Generate Persona**: An agent creates a detailed `persona.json` file based on the target audience and topic.
2.  **Run Interview**: `InterviewerAgent` and `RespondentAgent` engage in dialogue for the specified number of questions/rounds. The conversation is streamed to the console.
3.  **Analyze Transcript**: `AnalystAgent` processes the full transcript, performing thematic and sentiment analysis, and extracts structured data for visualizations.
4.  **Generate Report & Visualizations**: Compiles the analysis narrative, persona details, full transcript (`transcript.csv`), and generated charts (PNGs in `visualizations/`) into a final `report.md`.

## Output Structure

All outputs for a given run are saved within a specific directory structure:

```
openai-simulations-ui/
└── public/
    └── simulations/
        └── idi/
            └── {simulation_id}/        <-- Your unique simulation ID
                ├── persona.json
                ├── transcript.csv
                ├── visualizations/
                │   ├── sentiment_pie_chart.png
                │   ├── key_themes_

## Technical Notes

-   Uses different OpenAI models (configurable per agent) for various tasks (e.g., `gpt-4o` for persona generation, `o3-mini` for interview dialogue and analysis).
-   Leverages `asyncio` for running agent interactions.
-   Uses `matplotlib` and `numpy` for generating plots based on structured data extracted by the analyst.
-   Includes robust persona generation with JSON validation and fallback to generic persona on error.
-   The analyst extracts structured JSON from its narrative output for reliable visualization generation.

## Customization

You can modify the agent definitions in the script to adjust:
- The interview structure and flow
- Persona generation parameters
- Analysis focus areas
- Visualization styles

## Limitations

-   The simulated respondent is an LLM agent; insights are derived from its persona-based generation, not real human experience.
-   Output quality is highly dependent on the quality and detail of the `target_audience` description.
-   The interview flow depends on the AI agent interactions and may not perfectly replicate a human-led IDI.
-   Simulation involves API calls and associated costs.

