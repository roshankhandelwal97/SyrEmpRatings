from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
import pandas as pd
from collections import Counter
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import datetime
from geopy.geocoders import Nominatim 
from .model import predict


def round_to_nearest_half(num):
    return round(num * 2) / 2


def agency_ratings_view(request):
    data = pd.read_csv('ratings/ratings.csv')

    # Exclude 'Feedback to the City' from the dataset
    data = data[data['Agency_Name'] != 'Feedback to the City']

    agency_data = data.groupby('Agency_Name')['Calculated_Rating'].mean().reset_index(name='Average_Rating')

    # Message to ignore in category analysis
    messages_to_ignore = [
        "To report an illegally parked vehicle, please call the Syracuse Police Ordinance at 315-448-8650. If this is an emergency, please call 911. Do NOT submit requests to Cityline.",
        "To report an abandoned vehicle, please call the Syracuse Police Ordinance at 315-448-8650. If this is an emergency, please call 911. Do NOT submit requests to Cityline."
    ]

    # Adding number of reviews and top categories
    enriched_agency_data = []
    for _, row in agency_data.iterrows():
        agency_name = row['Agency_Name']
        specific_data = data[data['Agency_Name'] == agency_name]

        # Filter out the specific unwanted message from the categories
        filtered_categories = specific_data[~specific_data['Category'].isin(messages_to_ignore)]

        num_reviews = filtered_categories.shape[0]
        top_categories = Counter(filtered_categories['Category']).most_common(3)
        top_categories = [cat[0] for cat in top_categories]  # Extract category names

        enriched_agency_data.append({
            "Agency_Name": agency_name,
            "Average_Rating": row['Average_Rating'],
            "Reviews": num_reviews,
            "Top_Categories": top_categories
        })

    return JsonResponse(enriched_agency_data, safe=False)




def detailed_ratings_view(request):
    # Load the dataset with ratings
    data = pd.read_csv('ratings/ratings.csv')

    # Group by 'Agency_Name' and 'Assignee_Name' and calculate the average rating, then round
    detailed_ratings = data.groupby(['Agency_Name', 'Assignee_name'])['Calculated_Rating'].mean().reset_index(name='Average_Rating')
    detailed_ratings['Average_Rating'] = detailed_ratings['Average_Rating'].apply(round_to_nearest_half)

    # Restructure the data into the specified nested format
    structured_data = {}
    for _, row in detailed_ratings.iterrows():
        agency = row['Agency_Name']
        if agency not in structured_data:
            structured_data[agency] = []
        structured_data[agency].append({
            "Assignee_name": row['Assignee_name'],
            "Average_Rating": row['Average_Rating']
        })
    
    # Prepare the final list of dictionaries for JSON response
    response_data = [{"Agency_Name": key, "Assignees": value} for key, value in structured_data.items()]
    
    return JsonResponse(response_data, safe=False)

def agency_names_view(request):
    # Load the dataset
    data = pd.read_csv('ratings/ratings.csv')

    # Extract unique agency names, excluding 'Feedback to the City'
    unique_agencies = data[data['Agency_Name'] != 'Feedback to the City']['Agency_Name'].unique()
    unique_agencies = sorted(unique_agencies)  # Optionally sort the names alphabetically

    # Return these names as a JSON list
    return JsonResponse({'agency_names': unique_agencies}, safe=False)

def fetch_categories_by_agency(request):
    agency_name = request.GET.get('agency_name')
    if not agency_name:
        return JsonResponse({'error': 'Agency name is required'}, status=400)

    # Load the dataset
    data = pd.read_csv('ratings/ratings.csv')

    # Filter data by agency and get unique categories
    categories = data[data['Agency_Name'] == agency_name]['Category'].unique()
    categories = sorted(categories)  # Sort the categories alphabetically

    return JsonResponse({'categories': list(categories)}, safe=False)



@csrf_exempt
def submit_issue_request(request):
    if request.method == 'POST':
        # Extract form data from POST request
        data = json.loads(request.body.decode('utf-8'))
        
        ratings_data = pd.read_csv('ratings/ratings.csv')

        # Generate Date, Month, Hour from the server's current datetime
        current_datetime = datetime.now()
        data['Day_of_week'] = current_datetime.weekday()  # Monday is 0 and Sunday is 6
        data['Month'] = current_datetime.month
        data['Hour_of_day'] = current_datetime.hour

        # Geocode to get Latitude and Longitude (ensure 'Street_Address' is provided in data)
        street_address = data.get('Street_Address')
        if street_address:
            full_address = f"{street_address}, Syracuse, NY, USA"
            geolocator = Nominatim(user_agent="YourAppName (your.email@example.com)")
            location = geolocator.geocode(full_address)
            if location:
                data['Lat'] = location.latitude
                data['Lng'] = location.longitude

        filtered_data = ratings_data[(ratings_data['Agency_Name'] == data['Agency_Name']) & (ratings_data['Category'] == data['Category'])]
        if not filtered_data.empty:
            sla_value = filtered_data['Sla_in_hours'].iloc[0]
            data['SLA'] = int(sla_value)  # Convert SLA to integer
        else:
            data['SLA'] = 0 

        # Pass data to ML model
        print(data)
        prediction = predict(data)
        days, hours = divmod(prediction, 1440)
        hours = hours // 60  # Convert remaining minutes to hours

        response_message = f"{int(days)} days, {int(hours)} hours"
        return JsonResponse({'predicted_resolution_time': response_message})

    else:
        return JsonResponse({'error': 'This endpoint supports only POST requests.'}, status=405)
    
    