# Relationship Matching Algorithm

![Matching Algorithm Demo](https://github.com/Gartcart/knock-public/blob/main/graph_matching_smooth.gif)

## Overview

This project implements a relationship matching algorithm that uses cosine similarity and maximum weight matching to find optimal pairings based on compatibility scores and dealbreakers. The system processes survey responses, extracts personal preferences and requirements, calculates compatibility between individuals, and visualizes the matching process.

## Data Collection
Data was gathered from a pool of 195 people at Auburn University who wanted to go on blind dates. Fliers, Instagram ads, and word-of-mouth were all used to spread the message. Upon gathering the data, it was preprocessed by converting it into a csv and removing unnecessary Google Docs information so it could be easily processed by our model.

## Features

- **Comprehensive Compatibility Analysis**: Evaluates matches based on multiple dimensions including emotion handling, conflict resolution, extraversion, lifestyle, communication style, and humor.
- **Dealbreaker Filtering**: Respects hard requirements for age, schedule compatibility, sexuality, height preferences, tattoos, alcohol preferences, and religious compatibility.
- **Optimized Matching**: Uses the maximum weight matching algorithm to find the best possible pairings.
- **Visual Analytics**: Provides animated visualizations of the graph building and matching process.

## Components

### Core Files

- **preprocessing.py**: Processes survey responses and creates person profiles with preference vectors and dealbreakers.
- **graph.py**: Implements the matching algorithm and visualization.
- **create_vector.py**: Creates and manages the preference vectors for compatibility calculations.
- **pull_data.py**: Utility script for handling data import and exploration.

### Data Files

- **response.csv**: Raw survey data from participants.
- **data.csv**: Processed compatibility scores between potential pairs.
- **matching_pairs.csv**: Final output of matched pairs with their compatibility scores.

## How It Works

### 1. Data Processing

Survey responses are processed to extract:
- Basic information (name, age, gender, etc.)
- Dealbreakers (age range, schedule compatibility, etc.)
- Preference vectors across multiple dimensions (such as on a scale of 1 - 7 how funny you find certain jokes, or how important communication is to you)

### 2. Compatibility Calculation

The system calculates compatibility between individuals using:
- Binary dealbreaker checks that filter out incompatible matches
- Cosine similarity between preference vectors to generate a compatibility score

### 3. Graph Construction and Matching

- Each person represents a node in a graph
- Potential matches are represented as edges with weights based on compatibility scores
- The NetworkX maximum weight matching algorithm finds the optimal set of matches

### 4. Visualization

The matching process is visualized through an animation that shows:
- Initial graph construction with all potential matches
- The selection of optimal matches highlighted in red

## Algorithm Details

### Preference Vectors

Each person's preferences and traits are represented across several dimensions:
- Emotion handling
- Conflict resolution
- Extraversion
- Lifestyle
- Communication
- Partner interaction
- Humor

### Dealbreakers

The system enforces several types of dealbreakers:
- Age compatibility
- Schedule compatibility (common available days for the blind dates)
- Sexuality and gender preferences
- Height preferences
- Tattoo preferences
- Alcohol/substance preferences
- Religious compatibility

### Matching Algorithm

The optimal matching is determined using the maximum weight matching algorithm from graph theory, which finds a set of edges with maximum total weight where no vertex appears in more than one edge. This does, of course, introduce a challenge as not every person is capable of finding a match, but we have concluded that for the sake of this experiment, it is best to prioritize quality for each match vs quantity. This is a challenge we might need to overcome in future iterations.

## Setup and Usage

### Requirements

- Python 3.6+
- pandas
- numpy
- networkx
- matplotlib
- ffmpeg (for animation generation)

### Installation

```bash
# Clone the repository
git clone (repository goes here)
cd relationship-matching

# Install required packages
pip install pandas numpy networkx matplotlib ffmpeg-python
```

### Running the Algorithm

1. Ensure your survey data is in the correct format in `response.csv`
2. Run the preprocessing script to generate the compatibility data:
   ```
   python preprocessing.py
   ```
3. Run the graph matching and visualization:
   ```
   python graph.py
   ```
4. The results will be saved in `matching_pairs.csv` and the animation in `graph_animation.mp4`

## Extending the Algorithm

The matching algorithm can be customized by:
- Adjusting the weights of different preference vectors
- Modifying dealbreaker criteria
- Changing the `maxcardinality` parameter in the matching algorithm
- Adding new dimensions for compatibility assessment

## License

Modified MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, and distribute the Software, and to
permit persons to whom the Software is furnished to do so, subject to the
following conditions:

1. The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

2. Private use of the Software is prohibited without express written permission
   from the copyright holder. "Private use" means use of the Software within a
   non-public, non-open source product, service, or organization.

3. Commercial use of the Software requires explicit written permission from the
   copyright holder.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.]

## Contributors

Gart, Trey Hibbard

## Data Anonymization

All participant data in this repository has been anonymized. Names are replaced with consistent identifiers (P1, P2, ...) across every file, and all contact information (emails, phone numbers, social media handles) has been removed. Visualizations were regenerated with anonymized labels.
