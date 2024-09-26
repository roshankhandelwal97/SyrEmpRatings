import pandas as pd

# Load your dataset
data = pd.read_csv('ratings/enhanced_data.csv')

def calculate_rating(row):
    # Extracting relevant columns
    minutes_closed = row['Minutes_to_closed']
    expected_minutes = row['CV_Expected_Minutes_to_Close']
    sla_hours = row['Sla_in_hours']
    
    # Calculate efficiency and responsiveness
    efficiency = expected_minutes / minutes_closed if minutes_closed > 0 else 0
    responsiveness = minutes_closed / (sla_hours * 60) if sla_hours > 0 else 0
    
    # Normalize efficiency and responsiveness to be between 0 and 1
    efficiency_score = min(efficiency, 1.0)
    responsiveness_score = 1 - min(responsiveness, 1.0)  # Lower is better
    
    # Combined weighted score (assuming equal weighting)
    total_score = (efficiency_score + responsiveness_score) / 2
    
    # Convert to a scale of 1 to 5
    final_rating = total_score * 5

    # Round to the nearest half
    final_rating = round(final_rating * 2) / 2
    
    return final_rating

# Apply the rating function to the dataframe
data['Calculated_Rating'] = data.apply(calculate_rating, axis=1)

# Save the updated dataframe to a new CSV file
data.to_csv('ratings/ratings.csv', index=False)
