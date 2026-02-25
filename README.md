\# Evolutionary System-Prompt Optimization for LLM Agents in Competitive Market Simulations



This repository contains the code and experimental framework used to evaluate and optimize Large Language Model (LLM) agents in competitive economic simulations.



\## Overview



We implement an Agent-Based Model (ABM) where LLM-driven firms compete against other LLM agents and heuristic rule-based firms.  

LLM behavior is optimized using a binary Genetic Algorithm (GA) applied to system prompts.



---



\## Repository Structure



src/llm\_abm\_ga/ # Core package (environment, agents, GA, LLM clients)

scripts/ # Run experiments (run\_ga.py, run\_market.py)

notebooks/ # Interactive experiments and analysis

results/ # Logs, fitness values, optimized genomes



---



\## Installation



```bash

git clone <repository-url>

cd llm\_abm\_ga\_project

pip install -e .



---



\## API Configuration



Create a .env file in the root directory:



OPENAI\_API\_KEY="your-openai-api-key"

GEMINI\_API\_KEY="your-gemini-api-key"



LLM\_PROVIDER="openai"        # or "gemini"

LLM\_MODEL="gpt-4o-mini"      # e.g. gpt-3.5-turbo, gpt-5-mini, gemini-2.0-flash

MAX\_TOKENS=30



---



\## Usage



Run the Genetic Algorithm: python scripts/run\_ga.py

Run a Market Simulation: python scripts/run\_market.py

Alternatively, simulations can be executed using the template notebooks available in: llm\_abm\_ga\_project/notebooks/

Results are saved in the /results/ directory.



