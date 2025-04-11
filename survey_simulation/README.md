# Survey Simulation

An advanced survey simulation tool powered by OpenAI's GPT models, designed to produce professional-grade quantitative research insights and reports.

## Overview

This tool orchestrates a multi-agent system to simulate a complete survey pipeline. It designs a survey based on research objectives, generates diverse respondent personas based on a target audience, simulates realistic responses from those personas, processes the raw data, performs a detailed analysis, generates visualizations, and compiles a final report.

## Features

-   **AI-Powered Survey Design**: Generates comprehensive survey instruments (questions, sections, types) tailored to research goals.
-   **Dynamic Respondent Generation**: Creates detailed, diverse respondent profiles (demographics, psychographics, behaviors) based on target audience descriptions.
-   **Realistic Response Simulation**: Captures varied response styles (e.g., thoughtful, rushed, biased) for nuanced data.
-   **Automated Data Processing**: Structures raw JSON responses into an aggregated format.
-   **In-depth Analysis**: An AI analyst agent identifies key findings, segment differences, and provides an executive summary.
-   **Data Visualizations**: Automatically generates charts (histograms, bar charts) for demographic distributions and question responses using Matplotlib and Seaborn.
-   **Comprehensive Reporting**: Produces a Markdown report summarizing the methodology, findings, analysis, and visualizations.
-   **Structured Output**: Saves all generated artifacts (survey, personas, responses, data, analysis, visualizations, report) in a dedicated simulation folder.

## Prerequisites

-   Python 3.8+
-   OpenAI API Key

## Installation

1.  Clone the repository.
2.  Install the required dependencies. Assuming `openai-agents` is a custom library within the project structure:
    ```bash
    pip install python-dotenv matplotlib numpy pandas seaborn
    # Ensure the 'agents' module/library is accessible in your Python path
    ```
3.  Create a `.env` file in the project root directory (or where the script can find it) with your OpenAI API key:
    ```
    OPENAI_API_KEY="your_api_key_here"
    ```

## Usage

Run the survey simulation script from the command line, providing the necessary arguments:

```bash
python survey_simulation/survey_simulation.py \
    --topic "Your Survey Topic" \
    --objectives "Your Research Objectives" \
    --audience "Description of Your Target Audience" \
    --respondents 50 \
    --simulation_id "unique_simulation_run_id"
```

**Command-Line Arguments:**

-   `--topic` (string, **required**): The central subject of the survey (e.g., "Consumer preferences for electric vehicles").
-   `--objectives` (string, **required**): A description of what the survey aims to discover (e.g., "Understand barriers to EV adoption among urban professionals").
-   `--audience` (string, **required**): A detailed description of the target respondent group (e.g., "Urban car owners aged 25-50 with household income above $75k").
-   `--respondents` (integer, optional, default: `50`): The number of virtual respondents to simulate.
-   `--simulation_id` (string, **required**): A unique identifier for this specific simulation run. This ID determines the output folder name.

## Simulation Pipeline

The script executes the following steps sequentially:

1.  **Design Survey**: `SurveyDesignerAgent` creates `survey_questions.json`.
2.  **Generate Respondents**: `RespondentGeneratorAgent` creates `respondents.json`.
3.  **Collect Responses**: `SurveyResponseAgent` instances (one per respondent) generate responses, saved in `survey_responses_raw.json`.
4.  **Process Data**: Raw responses are aggregated into `survey_data_processed.json`.
5.  **Analyze Results**: `SurveyAnalystAgent` produces `survey_analysis.json` containing narrative analysis and structured visualization data.
6.  **Generate Visualizations**: Creates PNG charts (e.g., `age_distribution.png`, `q_q1_responses.png`) within the `visualizations/` subdirectory.
7.  **Generate Final Report**: Compiles all findings into `report.md`.
8.  **Save Parameters**: Records simulation inputs and status in `parameters.json`.

## Output Structure

All outputs for a given run are saved within a specific directory structure:

```
openai-simulations-ui/
└── public/
    └── simulations/
        └── survey/
            └── {simulation_id}/        <-- Your unique simulation ID
                ├── survey_questions.json
                ├── respondents.json
                ├── survey_responses_raw.json
                ├── survey_data_processed.json
                ├── survey_analysis.json
                ├── visualizations/
                │   ├── age_distribution.png
                │   ├── gender_distribution.png
                │   ├── q_q1_responses.png
               │    └── ... (other question charts)
               ├── report.md
               └── parameters.json

## Technical Notes

-   Uses different OpenAI models (configurable per agent) for various tasks (e.g., `gpt-4o` for analysis, `o3-mini` for persona generation).
-   Leverages `asyncio` for concurrent respondent simulation.
-   Uses `pandas` and `numpy` for data handling and potential calculations within analysis/visualization.
-   Employs `matplotlib` and `seaborn` for generating plots.
-   Includes basic error handling and fallback mechanisms for agent failures (e.g., generating generic personas or a fallback survey).

## Customization

You can modify the script to adjust:
- The types of questions generated
- The analysis techniques applied
- The visualization styles
- The report structure and content

## Limitations

-   Simulated data is not a replacement for real-world survey data. Respondent behavior is based on LLM capabilities and assigned personas/styles.
-   The quality of the output heavily depends on the clarity and detail of the input `topic`, `objectives`, and `audience` descriptions.
-   Simulation involves API calls and associated costs.

