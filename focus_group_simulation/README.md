# Focus Group Simulation

An advanced focus group simulation tool powered by OpenAI's GPT models, designed to produce professional-grade qualitative market research insights and reports.

## Overview

This tool simulates a complete focus group discussion. It dynamically generates multiple diverse participant personas based on a target audience, facilitates a discussion led by an AI moderator, analyzes the transcript for themes, sentiment, and participant dynamics, and generates a comprehensive report with visualizations.

## Features

-   **Dynamic Persona Generation**: Creates multiple distinct, detailed participant personas (demographics, psychographics, behaviors, attitudes) based on the target audience description.
-   **Expert AI Moderator**: Simulates a moderator guiding the discussion using professional techniques (probing, managing flow, encouraging interaction, synthesizing).
-   **Authentic AI Participants**: Each participant agent responds consistently based on their unique assigned persona, interacting with the moderator and each other.
-   **Realistic Discussion Flow**: Generates multi-turn, back-and-forth dialogue across specified discussion rounds.
-   **In-depth Qualitative Analysis**: An AI analyst agent performs thematic analysis, sentiment analysis (overall and per participant), identifies key quotes, and analyzes participant engagement/dynamics.
-   **Professional Reporting**: Generates a detailed Markdown report including an executive summary, methodology, participant profiles, key themes, sentiment analysis, participant dynamics, actionable insights, and the full transcript.
-   **Data Visualizations**: Creates charts (pie, bar, stacked bar) visualizing overall sentiment, key themes, sentiment per participant, and engagement metrics using Matplotlib.
-   **Structured Output**: Saves the transcript (CSV), report, and visualizations in a dedicated simulation folder. Persona files are saved temporarily for reference during the run.

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

Run the focus group simulation script from the command line, providing the necessary arguments:

```bash
python focus_group_simulation/focus_group_simulation.py \
    --topic "Your Focus Group Topic" \
    --target_audience "Description of Your Target Audience" \
    --num_participants 6 \
    --num_rounds 4 \
    --simulation_id "unique_simulation_run_id"
```

**Command-Line Arguments:**

-   `--topic` (string, optional, default: `"Consumer preferences for sustainable footwear"`): The central subject of the discussion.
-   `--target_audience` (string, optional, default: `"Urban professionals aged 25-40 interested in sustainability"`): A description of the desired participant profiles.
-   `--num_participants` (integer, optional, default: `4`): The number of virtual participants to include in the group.
-   `--num_rounds` (integer, optional, default: `3`): The number of core discussion rounds (moderator initiates each round).
-   `--simulation_id` (string, **required**): A unique identifier for this specific simulation run. This ID determines the output folder name.

## Simulation Pipeline

1.  **Generate Personas**: An agent creates detailed JSON profiles for the specified number of participants. *Note: These JSON files (`personas_{timestamp}.json`) are currently saved to an `output_data/` directory relative to the script's execution path for reference during the run, not within the final simulation ID folder.*
2.  **Run Simulation**: `ModeratorAgent` leads the discussion, prompting responses from `ParticipantAgent` instances over the specified rounds. The conversation is streamed to the console.
3.  **Analyze Transcript**: `AnalystAgent` processes the full transcript, performing thematic/sentiment analysis, analyzing dynamics, and extracting structured data for visualizations.
4.  **Generate Report & Visualizations**: Compiles the analysis narrative, participant profiles, full transcript (`transcript.csv`), and generated charts (PNGs in `visualizations/`) into a final `report.md`.

## Output Structure

Most outputs for a given run are saved within a specific directory structure:

```
openai-simulations-ui/
└── public/
    └── simulations/
        └── focus-group/
            └── {simulation_id}/        <-- Your unique simulation ID
                ├── transcript.csv
                ├── visualizations/
                │   ├── sentiment_pie_chart.png
                │   ├── key_themes_chart.png
                │   ├── participant_sentiment.png
                │   └── engagement_metrics.png
                └── report.md
```

## Technical Notes

-   Uses different OpenAI models (configurable per agent, e.g., `gpt-4o` for analysis/personas, `o3-mini` for discussion).
-   Leverages `asyncio` for running agent interactions.
-   Uses `matplotlib` and `numpy` for generating plots based on structured data extracted by the analyst agent.
-   Includes persona generation with JSON validation and fallback.
-   The analyst extracts structured JSON from its narrative output for reliable visualization generation.
-   Participant responses are currently sequential within each round for simplicity.

## Limitations

-   Simulated participants are LLM agents; insights reflect persona-based generation, not genuine group dynamics or individual human experiences.
-   Output quality depends heavily on the detail and clarity of the `target_audience` description.
-   The simulation captures basic turn-taking but may not fully replicate the complex social dynamics of a real focus group.
-   Simulation involves API calls and associated costs, increasing with more participants and rounds.

## Customization

You can modify the agent definitions in the script to adjust:
- The number and type of discussion rounds
- Persona generation parameters
- Analysis focus areas
- Visualization styles

