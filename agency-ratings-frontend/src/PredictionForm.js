import React, { useState, useEffect } from 'react';
import { submitIssue, fetchAgencyNames, fetchCategoriesByAgency } from './Api';
import './styles/PredictionForm.css'; // Import the CSS



function PredictionForm() {
    const [address, setAddress] = useState('');
    const [agency, setAgency] = useState('');
    const [agencies, setAgencies] = useState([]);
    const [category, setCategory] = useState('');
    const [categories, setCategories] = useState([]);
    const [prediction, setPrediction] = useState('');

    useEffect(() => {
        const loadAgencies = async () => {
            const data = await fetchAgencyNames();
            setAgencies(data.agency_names);
        };
        loadAgencies();
    }, []);

    useEffect(() => {
        const loadCategories = async () => {
            if (agency) {
                const categoryData = await fetchCategoriesByAgency(agency);
                setCategories(categoryData);
            }
        };
        loadCategories();
    }, [agency]); // Fetch categories when 'agency' changes

    const handleSubmit = async (event) => {
        event.preventDefault();
        const result = await submitIssue({
            Street_Address: address,
            Agency_Name: agency,
            Category: category
        });
        setPrediction(result.predicted_resolution_time);
    };

    return (
        <div className="prediction-page">
            <div className="form-container">
                <h1><strong style={{ fontWeight: 'bold', color: '#0a7ca4', fontSize: '1.5em' }}>Predict </strong> Resolution Time</h1>
                <form onSubmit={handleSubmit}>
                    <input type="text" value={address} onChange={(e) => setAddress(e.target.value)} placeholder="Street Address" />
                    <select value={agency} onChange={(e) => setAgency(e.target.value)}>
                        <option value="">Select an Agency</option>
                        {agencies.map((name, index) => <option key={index} value={name}>{name}</option>)}
                    </select>
                    <select value={category} onChange={(e) => setCategory(e.target.value)}>
                        <option value="">Select a Category</option>
                        {categories.map((cat, index) => <option key={index} value={cat}>{cat}</option>)}
                    </select>
                    <button type="submit">Predict</button>
                </form>
                {prediction && <p className="prediction-label"><br/>Predicted Resolution Time <span className="prediction-result">{prediction}</span></p>}
            </div>
            <div className="content-container">
                <h2>How the Prediction Model Works</h2>
                <p>The QuickWork model is trained on the Syracuse cityline requests datasets<br/><br/>

                The QuickWork model leverages Python's data manipulation libraries and machine learning techniques, specifically a Random Forest algorithm, to predict the time required to close service requests. <br/><br/>

                Random Forest: a robust ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the average prediction of the individual trees. This approach helps improve accuracy and control over-fitting, making it suitable for complex datasets with multiple input features that may have nonlinear relationships.<br/><br/>

                This QuickWork model not only aids in immediate decision-making but also provides a scalable framework for ongoing municipal operational improvements.</p>
                <img src={`${process.env.PUBLIC_URL}/PredictionImage.jpeg`} alt="Comparison Chart" style={{ width: '100%', marginTop: '20px' }} />
            </div>
        </div>
    );
}

export default PredictionForm;
