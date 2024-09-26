from django.urls import path
from .views import agency_ratings_view, detailed_ratings_view, submit_issue_request, agency_names_view, fetch_categories_by_agency

urlpatterns = [
    path('agency_ratings/', agency_ratings_view, name='agency_ratings'),
    path('detailed_ratings/', detailed_ratings_view, name='detailed_ratings'),  # New URL for detailed ratings
    path('submit_issue/', submit_issue_request, name='submit_issue'),
    path('agency_names/', agency_names_view, name='agency_names'),
    path('categories_by_agency/', fetch_categories_by_agency, name='categories_by_agency'),

]
