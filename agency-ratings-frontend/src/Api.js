import axios from 'axios';

const baseURL = 'http://127.0.0.1:8000/api/';

export const fetchAgencyRatings = async () => {
  try {
    const response = await axios.get(`${baseURL}agency_ratings/`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch agency ratings:', error);
    return [];
  }
};

export const fetchDetailedRatings = async (agencyName) => {
  try {
    const response = await axios.get(`${baseURL}detailed_ratings/?agencyName=${encodeURIComponent(agencyName)}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch detailed ratings:', error);
    return [];
  }
};

export const submitIssue = async (data) => {
  try {
    const response = await axios.post(`${baseURL}submit_issue/`, data);
    return response.data;
  } catch (error) {
    console.error('Error submitting issue:', error);
    return null;
  }
};

export const fetchAgencyNames = async () => {
  try {
      const response = await axios.get('http://127.0.0.1:8000/api/agency_names/');
      return response.data;  // Adjust according to your actual API response
  } catch (error) {
      console.error('Failed to fetch agency names:', error);
      return [];  // Return empty array on error
  }
};


export const fetchCategoriesByAgency = async (agencyName) => {
    try {
        const response = await axios.get(`http://127.0.0.1:8000/api/categories_by_agency/?agency_name=${encodeURIComponent(agencyName)}`);
        return response.data.categories;  // Adjust according to your actual API response
    } catch (error) {
        console.error('Failed to fetch categories:', error);
        return [];
    }
};

