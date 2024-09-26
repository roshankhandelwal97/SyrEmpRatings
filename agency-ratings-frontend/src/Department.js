import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { fetchDetailedRatings } from './Api';
import ReactStars from 'react-rating-stars-component';
import { Card } from 'react-bootstrap';
import './styles/Department.css'; // Import the CSS

function Department() {
    const { agencyName } = useParams();
    const [assignees, setAssignees] = useState([]);
    const [selectedAssignee, setSelectedAssignee] = useState(null);

    useEffect(() => {
        const loadData = async () => {
            const data = await fetchDetailedRatings(decodeURIComponent(agencyName));
            if (data && data.length > 0 && data[0].Assignees.length > 0) {
                setAssignees(data[0].Assignees);
                setSelectedAssignee(data[0].Assignees[0]); // Set the first assignee as selected
            }
        };
        loadData();
    }, [agencyName]);

    const handleSelectAssignee = (assignee) => {
        setSelectedAssignee(assignee);
    };

    return (
        <div className="Department">
            <div className={`sidebar ${selectedAssignee ? 'show' : ''}`}>
                {selectedAssignee && (
                    <>
                        <h2>{selectedAssignee.Assignee_name}</h2>
                        <p>Phone: (123) 456-7890</p>
                        <p>Email: syrCitylineDemo@syr.edu</p>
                    </>
                )}
            </div>
            <div className="department-main">
                <h1><strong style={{ fontWeight: 'bold', color: '#0a7ca4', fontSize: '1.5em' }}>Ratings </strong> for {decodeURIComponent(agencyName)}</h1>
                <div className="department-cards">
                    {assignees.map((assignee, index) => (
                        <Card
                            key={index}
                            className={`assignee-card ${selectedAssignee && selectedAssignee.Assignee_name === assignee.Assignee_name ? 'selected' : ''}`}
                            onClick={() => handleSelectAssignee(assignee)}
                        >
                            <Card.Body>
                                <Card.Title>{assignee.Assignee_name}</Card.Title>
                                <ReactStars
                                    count={5}
                                    value={assignee.Average_Rating}
                                    size={24}
                                    activeColor="#ffd700"
                                    isHalf={true}
                                    edit={false}
                                />
                            </Card.Body>
                        </Card>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default Department;
