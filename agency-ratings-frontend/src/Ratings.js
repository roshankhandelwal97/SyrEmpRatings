import React, { useEffect, useState } from 'react';
import { fetchAgencyRatings } from './Api';
import ReactStars from 'react-rating-stars-component';
import { useNavigate } from 'react-router-dom';
import { Card } from 'react-bootstrap'; 
import './styles/Ratings.css';  // Make sure to import your CSS file

function Ratings() {
    const [agencies, setAgencies] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        const loadData = async () => {
            const data = await fetchAgencyRatings();
            setAgencies(data);
        };
        loadData();
    }, []);

    return (
        <div className="home">
            <h1>Trusted Department<strong style={{ fontWeight: 'bold', color: '#0a7ca4', fontSize: '1.5em' }}> Ratings </strong></h1>
            <div className="card-container"> {/* Added for layout styling */}
                {agencies.map((agency, index) => (
                    <Card key={index} className="agency-card" onClick={() => navigate(`/department/${encodeURIComponent(agency.Agency_Name)}`)}>
                        <Card.Body>
                            <Card.Title>{agency.Agency_Name}</Card.Title>
                            <div className="rating-container">
                                <ReactStars
                                count={5}
                                value={agency.Average_Rating}
                                size={24}
                                activeColor="#ffd700"
                                edit={false}
                                />
                                <span className="rating-text">Average Rating {agency.Average_Rating.toFixed(1)} | {agency.Reviews} reviews</span>
                            </div>
                        </Card.Body>
                        <Card.Footer>
                            {agency.Top_Categories.join(', ')}
                        </Card.Footer>
                    </Card>
                ))}
            </div>
        </div>
    );
}

export default Ratings;
