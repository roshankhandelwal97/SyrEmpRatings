import pandas as pd

def group_ratings_by_agency():
    # Load the dataset with ratings
    data = pd.read_csv('ratings/ratings.csv')

    # Group by 'Agency_Name' and calculate the average rating
    agency_ratings = data.groupby('Agency_Name')['Calculated_Rating'].mean().reset_index(name='Average_Rating')
    
    # Optionally, save this to a CSV if needed for debugging or further use
    agency_ratings.to_csv('ratings/agency_ratings.csv', index=False)
    
    return agency_ratings

if __name__ == "__main__":
    print(group_ratings_by_agency())
