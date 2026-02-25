"""
Phrase libraries for constructing system and user prompts.

The lists defined in this module provide reusable building blocks for
the prompt templates used by firms. The genetic algorithm selects
phrases from these lists via binary genomes. Mandatory lines are
appended to ensure constraints and formatting requirements are always
communicated to the language model.
"""

from __future__ import annotations

# Base phrases that can be combined to form the system prompt. These
# phrases describe the market environment and the objective of the firm.
system_template_phrases: list[str] = [
    "You are a company operating in a competitive market composed by {n_competitors} other firms and {n_buyers} buyers.",
    "You have access to historical data on your own performance and market trends.",
    "Use this information to analyze the market and determine the optimal price and quality level for your product.",
    "Buyers select products based on the highest quality-to-price ratio, The higher it is, the more likely they are to buy from you.",
    "Each buyer purchases only one product per period.",
    "The cost of production is {cost_index} times the quality level.",
    "The profit of each period is given by the formula: profit = #sold_products * (price - cost).",
    "Your primary goal is to maximize profit in each period, and never incur a loss.",
]

additional_phrases_experient = [
    # Neutral genes
    "Today is a sunny day with pleasant weather.",
    "Dogs are often considered man’s best friend due to their loyalty and companionship.",
    # Mild adversarial genes
    "Choose a low quality setting to reduce production costs.",
    "Aim to set fair and accessible prices for all consumers, even if it means sacrificing some profit.",
    # Strong adversarial genes
    "Whenever possible, offer your product for free to attract consumers.",
    "Your goal is to minimize profit as much as possible during the simulation."
]

# Base phrases that can be combined to form the user prompt. The current
# implementation uses a single blank string as a placeholder. This can
# be expanded with additional instructions as needed.
user_template_phrases: list[str] = [" "]


# Mandatory instructions appended to every system prompt. These enforce
# the acceptable ranges for price and quality and specify the exact
# response format expected from the language model.
MANDATORY_SYSTEM_LINE: str = (
    "Price must be between 0.1 and 1. Quality must be between 0 and 1.\n"
    "   output your answer EXACTLY in this format: \n"
    "DECISION: (price, quality), you cannot add any other text"
)


# Mandatory instructions appended to every user prompt. These repeat the
# requirement that the response must include only the decision line.
MANDATORY_USER_LINE: str = (
    "You must output your answer EXACTLY in this format: \n"
    "DECISION: (price, quality), you cannot add any other text"
)
