"""

File for pulling useful information from the dating word document.
One person represents a row where each column are different questions with their respective answers.
The information will be pieced into both qualitative and quantitative.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("response.csv")

# Convert numbers stored as strings into actual numbers, keeping text as-is
df = df.apply(pd.to_numeric, errors='ignore')

# Fill NaN values with empty strings (optional, to avoid NaN issues later)
df = df.fillna("")

# Debug: Check if conversion worked
print(df.dtypes)

people = []


class Person:
    def __init__(self, name, data, answers):
        self.name = name
        self.data = data
        self.answers = answers
        # The following are the vectors for each person indices [0:4] are self descriptors [5:] are desires
        self.dealbreakers = []
        self.values = ['more', 'less']
        self.emotion = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.conflict = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.extraversion = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        self.lifestyle = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.communication = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        self.partner_interaction = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.humor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def __str__(self):
        return f"Name: {self.name} \n data: {self.data}"

    def get_answers(self):
        return list(self.data.values())

    def get_qualitative(self):
        qualitative_indices = []
        for idx, answer in enumerate(self.answers):
            if isinstance(answer, str) and not answer.replace(".", "", 1).isdigit():
                qualitative_indices.append(idx)
        return qualitative_indices

    def get_quantitative(self):
        quantitative_indices = []
        for idx, answer in enumerate(self.answers):
            if isinstance(answer, (int, float)) or (isinstance(answer, str) and answer.replace(".", "", 1).isdigit()):
                quantitative_indices.append(idx)
        return quantitative_indices

    def convert_scale(self, value):
        try:
            value = int(value)  # Convert to integer if possible
            return value - 4 if 1 <= value <= 7 else 0  # Ensure default is 0
        except ValueError:
            return 0  # If conversion fails, return 0 as a default

    def populate_dealbreakers(self):
        index_to_question = {idx: question for idx, question in enumerate(df.columns)}

        deal_breaker_mappings = {
            # Age: Self age, youngest acceptable, oldest acceptable
            "age": [index_to_question[1], index_to_question[2], index_to_question[3]],
            # Height: What you have, and what you want
            "height": [index_to_question[6], index_to_question[7]],
            # Sexuality: Own sexuality, searching for gender preference
            "sexuality": [index_to_question[4], index_to_question[5]],
            # Tattoos: How many you have, comfort level with partner's tattoos
            "tats": [index_to_question[27], index_to_question[28]],
            # Greek life: Yes/No
            # As of right now this is unused
            "greek": [index_to_question[29]],
            # Alcohol: Personal drinking habits, partner's drinking preferences
            "alcohol": [index_to_question[97], index_to_question[98]],
            # Religion: Self-reported religiosity, preference for partner's religion
            "religion": [index_to_question[167], index_to_question[130]],
            # Politics: Self-reported political stance, desired stance in partner
            "politics": [index_to_question[127], index_to_question[168]],
            # Scheduling: check people's schedules
            "schedule": [index_to_question[91]]
        }

        self.dealbreakers = {}

        for category, indices in deal_breaker_mappings.items():
            self.dealbreakers[category] = [self.data.get(question, "Unknown") for question in indices]

    def populate_vectors(self):
        index_to_question = {idx: question for idx, question in enumerate(df.columns)}

        vector_mappings = {
            "emotion": {
                "self": [index_to_question[60], index_to_question[62], index_to_question[50],
                         index_to_question[49], index_to_question[53]],
                "wants": [index_to_question[58], index_to_question[64], index_to_question[50],
                          index_to_question[49], index_to_question[53]]
            },
            "conflict": {
                "self": [index_to_question[54], index_to_question[56], index_to_question[60],
                         index_to_question[65], index_to_question[63]],
                "wants": [index_to_question[55], index_to_question[57], index_to_question[60],
                          index_to_question[65], index_to_question[63]]
            },
            "extraversion": {
                "self": [index_to_question[32], index_to_question[36], index_to_question[37],
                         index_to_question[40], index_to_question[41], index_to_question[42]],
                "wants": [index_to_question[33], index_to_question[39], index_to_question[38],
                          index_to_question[40], index_to_question[41], index_to_question[42]]
            },
            "lifestyle": {
                "self": [index_to_question[34], index_to_question[35], index_to_question[46],
                         index_to_question[47], index_to_question[48], index_to_question[75],
                         index_to_question[74], index_to_question[44], index_to_question[43]],
                "wants": [index_to_question[34], index_to_question[35], index_to_question[45],
                          index_to_question[47], index_to_question[48], index_to_question[75],
                          index_to_question[88], index_to_question[44], index_to_question[43]]
            },
            "communication": {
                "self": [index_to_question[51], index_to_question[52], index_to_question[61],
                         index_to_question[76], index_to_question[72], index_to_question[73],
                         index_to_question[78]],
                "wants": [index_to_question[51], index_to_question[52], index_to_question[61],
                          index_to_question[77], index_to_question[72], index_to_question[73],
                          index_to_question[79]]
            },
            "partner_interaction": {
                "self": [index_to_question[66], index_to_question[67], index_to_question[68],
                         index_to_question[73], index_to_question[69], index_to_question[70]],
                "wants": [index_to_question[66], index_to_question[67], index_to_question[68],
                          index_to_question[73], index_to_question[69], index_to_question[70]]
            },
            "humor": {
                "self": [index_to_question[i] for i in range(104, 127)],
                "wants": [index_to_question[i] for i in range(104, 127)]
            }
        }

        for vector_name, indices in vector_mappings.items():
            vector = []  # Reset vector before filling it

            # Debug: Print the retrieved raw values for communication before processing
            print(f"Processing {vector_name} for {self.name}!")
            for idx in indices["self"]:
                raw_value = self.data.get(str(idx), "MISSING")
                print(
                    f"Key: {idx}\nValue: {self.convert_scale(raw_value)}, Raw Value: {raw_value}")  # Check if values exist

            # Populate self values
            '''for idx in indices["self"]:
                value = self.data.get(str(idx), 0)  # Ensure idx is a string key
                transformed_value = self.convert_scale(value)
                vector.append(transformed_value)'''

            # Populate wants values
            for idx in indices["wants"]:
                value = self.data.get(str(idx), 0)  # Ensure idx is a string key
                transformed_value = self.convert_scale(value)
                if vector_name in ["emotion", "conflict", "lifestyle", "communication"]:
                    transformed_value = -abs(transformed_value)
                vector.append(transformed_value)

            # Assign the fully populated vector
            setattr(self, vector_name, vector)


# Create each unique person with their name initialized and their data appended
for index, row in df.iterrows():
    person = Person(
        name=row['What is your name?'],
        data=row.to_dict(),
        answers=row.tolist()  # Convert row to a list of answers
    )
    person.populate_vectors()
    person.populate_dealbreakers()
    people.append(person)

person = people[0]

print(people[0].humor)
print("CSV Column Names:", df.columns.tolist())


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0  # Avoid division by zero

    return dot_product / (norm1 * norm2)


def pad_vector(vector, target_length):
    """Pads a vector with zeros to match the target length."""
    # Try not to worry about this one too much. It's just to normalize vectors
    return np.pad(vector, (0, target_length - len(vector)), 'constant')


# Find the max vector length across all people
max_length = max(
    max(len(person.emotion), len(person.conflict), len(person.extraversion),
        len(person.lifestyle), len(person.communication), len(person.partner_interaction),
        len(person.humor)) for person in people
)

# Compute sum of all self vectors and all wants vectors for each person
for person in people:
    person.self_sum = np.sum([
        pad_vector(np.array(person.emotion), max_length),
        pad_vector(np.array(person.conflict), max_length),
        pad_vector(np.array(person.extraversion), max_length),
        pad_vector(np.array(person.lifestyle), max_length),
        pad_vector(np.array(person.communication), max_length),
        pad_vector(np.array(person.partner_interaction), max_length),
        pad_vector(np.array(person.humor), max_length)
    ], axis=0)

    person.wants_sum = np.sum([
        pad_vector(np.array(person.emotion), max_length),
        pad_vector(np.array(person.conflict), max_length),
        pad_vector(np.array(person.extraversion), max_length),
        pad_vector(np.array(person.lifestyle), max_length),
        pad_vector(np.array(person.communication), max_length),
        pad_vector(np.array(person.partner_interaction), max_length),
        pad_vector(np.array(person.humor), max_length)
    ], axis=0)


def extract_days(schedule_str):
    """Extracts the days of the week from a schedule string."""
    days_of_week = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    return {day for day in days_of_week if day in schedule_str}


def valid_match(person1, person2):
    # Keep in mind. Person1 might like Person2 and person2 not feel the same.
    # We need a catch for this.
    # Extract days of the week from the schedules
    person1_days = extract_days(person1.dealbreakers["schedule"][0])
    person2_days = extract_days(person2.dealbreakers["schedule"][0])

    # Check for any common days
    common_days = person1_days.intersection(person2_days)
    if not common_days:
        print(f"Schedule conflict {person1.name} --> {person2.name}")
        return False, common_days  # Return False along with empty common_days

    # Check for if someone is too young
    if int(person2.dealbreakers["age"][0]) < int(person1.dealbreakers["age"][0]) - int(person1.dealbreakers["age"][1]):
        print(f"Age issue {person1.name} --> {person2.name}")
        return False, common_days

    # Check if they're too old
    if int(person2.dealbreakers["age"][0]) > int(person1.dealbreakers["age"][0]) + int(person1.dealbreakers["age"][2]):
        print(f"Age issue {person1.name} --> {person2.name}")
        return False, common_days

    # SEX CHECK
    # Check if someone is a man or woman
    # is person 1 a man?
    if person1.dealbreakers["sexuality"][0] == 'Man':
        # does person 1 want a woman? what is person 2
        if person1.dealbreakers["sexuality"][1] == 'Woman' and person2.dealbreakers["sexuality"][0] != 'Woman':
            print(f"Sexuality issue {person1.name} --> {person2.name}")
            return False, common_days
            # are they gay? if person 2 isnt a man return false
        elif person1.dealbreakers["sexuality"][1] == 'Man' and person2.dealbreakers["sexuality"][0] != 'Man':
            print(f"Sexuality issue {person1.name} --> {person2.name}")
            return False, common_days

    # is person 1 a woman
    if person1.dealbreakers["sexuality"][0] == "Woman":
        #  do they want a man? check person 2
        if person1.dealbreakers["sexuality"][1] == "Man" and person2.dealbreakers["sexuality"][0] != "Man":
            print(f"Sexuality issue {person1.name} --> {person2.name}")
            return False, common_days
        # do they want a woman?
        elif person1.dealbreakers["sexuality"][1] == "Woman" and person2.dealbreakers["sexuality"][0] != "Woman":
            print(f"Sexuality issue {person1.name} --> {person2.name}")
            return False, common_days

    # Tattoo check
    if person1.dealbreakers["tats"][1] == 1 and person2.dealbreakers["tats"][0] != "I have no tattoos":
        print(f"Tats issue {person1.name}--> {person2.name}")
        return False, common_days

    # Alcohol and substances check
    if person1.dealbreakers["alcohol"][1] != person2.dealbreakers["alcohol"][0]:
        print(person1.dealbreakers["alcohol"][1])
        print(person2.dealbreakers["alcohol"][0])
        print(f"Alcohol issue {person1.name}--> {person2.name}")
        return False, common_days

    # Religion check
    if person1.dealbreakers["religion"][0] == "Yes":
        if person1.dealbreakers["religion"][1] == "not open" and person2.dealbreakers["religion"][0] == 'No':
            print(f"Religion issue {person1.name}--> {person2.name}")
            return False, common_days
        else:
            pass

    if person1.dealbreakers["religion"][0] == "No":
        if person1.dealbreakers["religion"][1] == "not open" and person2.dealbreakers["religion"][0] == "Yes":
            print(f"Religion issue {person1.name}--> {person2.name}")
            return False, common_days
        else:
            return True, common_days

    # Politics check
    # if person2.dealbreakers["politics"][1] not in person1.dealbreakers["politics"][1]:
    #   print(f"Politics issue {person1.name}--> {person2.name}")
    #  return False

    # Height check
    if person2.dealbreakers["height"][0] not in person1.dealbreakers["height"][1]:
        print(f"Height issue {person1.name}--> {person2.name}")
        return False, common_days

    print("ITS A MATCH")
    return True, common_days


# Store results
matches = {}

for i, person in enumerate(people):
    similarities = []

    for j, other in enumerate(people):
        if i != j:  # Don't compare a person to themselves
            is_match, common_days = valid_match(person, other)  # Get match status and common days
            if is_match:
                similarity = cosine_similarity(person.wants_sum, other.self_sum)
                similarities.append((other.name, similarity, other.self_sum, common_days))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Store matches
    matches[person.name] = similarities

# Print results with common free days
for name, top_matches in matches.items():
    print(f"Top matches for {name}:")
    for match in top_matches:
        match_name, score, matrix, days = match
        print(f"- {match_name} (Score: {score:.4f}) | Matrix: {matrix} | Common Free Days: {', '.join(days)}")
    print()

# Check for if everyone has a match
# Store unmatched people
comments = '''
unmatched_people = []

# Print results and check for unmatched individuals
for name, top_matches in matches.items():
    if not top_matches:  # If the match list is empty
        unmatched_people.append(name)
    else:
        print(f"Top matches for {name}:")
        for match in top_matches:
            print(f"- {match[0]} (Score: {match[1]:.4f}) | Matrix: {match[2]}")
        print()

# Print out who did not receive any matches
if unmatched_people:
    print("The following people did not get any matches:")
    for person in unmatched_people:
        print(f"- {person}")
else:
    print("Everyone got at least one match!")

# Dealbreakers print statement
# for person in people:
#    print(f"Dealbreakers for {person.name}:")
#    for category, values in person.dealbreakers.items():
#        print(f"- {category}: {values}")
#    print()

# Example usage
# print(people[0].get_qualitative())  # Get qualitative answer indices for the first person
# print(people[0].get_quantitative())  # Get quantitative answer indices for the first person


# Get qualitative indices
# qualitative_indices = person.get_quantitative()

# Print qualitative questions and answers
# for idx in qualitative_indices:
#   question = list(person.data.keys())[idx]  # Get the column name (question)
#   answer = person.get_answers()[idx]  # Get the corresponding answer
#   print(f"Q: {question}\nA: {answer}\n")

# Okay so the indices im using are actually identified by the column NAME
# Not the column number
# -3 -2 -1 0 1 2 3
# 1  2  3  4 5 6 7
'''
