import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from 'react-bootstrap';

function CategoryCard({ title, description, link }) {
  const navigate = useNavigate();

  return (
    <Card onClick={() => navigate(link)} className="my-3 p-3 clickable-card">
      <Card.Body>
        <Card.Title>{title}</Card.Title>
        <Card.Text dangerouslySetInnerHTML={{ __html: description }}></Card.Text>
      </Card.Body>
    </Card>
  );
}

export default CategoryCard;
