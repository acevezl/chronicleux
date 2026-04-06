# EPIC: Intelligent Analysis of Diary Study Data

## Epic Description
Enable UX researchers, specifically Evaluators, to transform longitudinal qualitative data into timely, actionable UX insights using NLP and LLM techniques, while preserving qualitative richness.

## Problem Statement
Diary studies generate high-value longitudinal data, but analysis is slow, manual, and cognitively demanding. This limits their adoption and delays UX decision-making. Researchers often struggle to extract sentiment trends, recurring themes, and meaningful patterns at scale.

## Goal
Reduce the time and effort required to analyze diary study entries by augmenting qualitative analysis with automated NLP and LLM-driven insights.

## Success Criteria
- Evaluators can run automated analysis on a study with minimal setup.
- The platform is able to evaluate:
    - Sentiment (per entry + aggregated trends)
    - Key themes/topics
    - Notable issues/frictions
- Results are interpretable and reviewable (not a black box)
- Time-to-insight is significantly reduced vs manual review

## User Stories

### 1. Study Design and Setup (i.e., Study Protocol Definition)

#### 1.1 As an Evaluator, I want to create diary studies so that I can use this technique in my UX research initiative. [Complete]

#### 1.2 As an Evaluator, I want to define the study protocol and key attributes so that analysis is grounded in research intent, participation is time bound, and study information is categorizable and retrievable. [Complete]

List of protocol attributes:
* Study title | `CharField`
* Study description | `TextField`
* Goal | `TextField` | Evaluator-facing
* Context | `TextField` 
* Hypotheses | `TextField`
* Participant instructions | `TextField`
* Tags | `CharField`
* Owner | `User` | Creator of study is owner by default
* Status | `CharField` | {PLANNING, COLLECTING, ANALYZING, COMPLETED}
* Start and End dates | `DatetimeField`
* Entry frequency | `CharField` | {DAILY, WEEKLY, EVENT_BASED, FREEFORM}
* Created at | `DatetimeField`
* Updated at | `DatetimeField`

#### 1.3 As an Evaluator, I want to define diary entry attributes and relevant prompts so that diary entries are standardized, consistent, guided, and relevant. [Complete]

List of diary entry attributes
* Study | `Study`
* Participant | `Participant`
* Sentiment Self Report (Likert Scale) | `CharField`
* Issue Encountered? | `BooleanField`
* Content | `TextField`
* Created at | `DateTimeField`


### 2. Participant Management

#### 2.1 As an Evaluator, I want to assign participants to my studies, so that I can control who joins the study.

#### 2.2 As an Evaluator, I want to remove or replace participants so that I can adapt the study if needed

#### 2.3 As a participant, I want to sign up to the platform so that I do not have to rely or wait for an Evaluator to grant me access.

#### [Future] As an Evaluator, I want to import participants in bulk (e.g., CSV) so that I can onboard large groups efficiently

#### [Future] As an Evaluator, I want to view participant status (active, inactive, dropped) so that I can manage engagement

### 3. Data Collection Monitoring

#### 3.1 As an Evaluator, I want to track incoming diary entries so that I know participation is happening

#### 3.2 As an Evaluator, I want to send manual and automatic reminders to participants so that they are reminded to complete their entries on time

#### 3.3 As an Evaluator, I want monitor submission frequency per participant so that I can detect drop-offs and missing entries

#### 3.4 As an Evaluator, I want to monitor overall study progress so that I can spot early signals or issues, follow up with participants, assess meaningfulness of entries, and know when enough data has been collected

### 4. Data Ingestion & Import

#### 4.1 As an Evaluator, I want to import external diary entries (CSV, JSON) so that I can analyze data collected outside the platform. [Complete]

### 5. Study Lifecycle Management

#### 5.1 As an Evaluator, I want to move a study through its status so that I can start and stop diary entry collection and move on to analysis and completion.

- Study workflow:
    - {PLANNING -> COLLECTING -> ANALYZING -> COMPLETED}

    - Planning: Study is being created, Evaluator is defining and entering protocol details.
    - Collecting: Participants performing their tasks and writing diary entries.
    - Analyzing: Data collection is over, Evaluator is analyzing data.
    - Completed: Analysis is over and conclusions have been extracted.

### 6. Analysis

#### 6.1 As an Evaluator, I want to run analysis on a study so that I can efficiently synthesize large volumes of diary data without manual review.

#### 6.2 As an Evaluator, I want to see sentiment trends over time so that I can understand how experiences evolve

#### 6.3 As an Evaluator, I want the system to highlight recurring issues so that I can prioritize improvements

#### 6.4 As an Evaluator, I want summaries of participant experiences so that I can quickly grasp key insights

#### 6.5 As an Evaluator, I want to explore themes across entries so that I can identify patterns

#### 6.6 As an Evaluator, I want to compare self-reported sentiment with inferred sentiment so that I can detect discrepancies

### 7. Natural Language Processing Pipelines

#### 7.1 As an Evaluator, I want to select among available analytical pipelines and configure their parameters so that automated analysis can be adapted to the methodological needs and research objectives of each diary study.

Pipelines are defined as configurable analytical processes that may include:

- Sentiment analysis
- Theme or topic extraction
- Issue detection
- Summarization
- Comparison between self-reported and inferred sentiment
- [Future]Different prompt templates or model configurations


### 8. Insight Validation (Human-in-the-loop)

#### 8.1 As an Evaluator, I want to review AI-generated insights so that I can verify their accuracy

#### 8.2 As an Evaluator, I want to edit or refine generated summaries so that they reflect my interpretation

#### 8.3 As an Evaluator, I want to drill down from aggregated insights to raw entries so that I can validate evidence

#### [Future] As an Evaluator, I want to flag incorrect insights so that the system improves over time

### 9. Reporting & Output

#### 9.1 As an Evaluator, I want to export insights so that I can share findings with stakeholders

#### 9.2 As an Evaluator, I want structured summaries of findings so that I can support decision-making

#### 9.3 As an Evaluator, I want visualizations of trends so that insights are easy to communicate

### 10. User Interface and Experience Improvements

#### 10.1 As a user, I want to see the participant details when I click on a system username so that I can see the user details

#### 10.2 As a user, I want to sort any and all tables by their headers, so that I can organize the information in the screen.