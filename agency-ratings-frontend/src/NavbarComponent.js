// NavbarComponent.js
import React from 'react';
import { Navbar, Nav } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './styles/styles.css';  // Assuming the CSS file is named styles.css and is in the same folde


function NavbarComponent() {
    return (
      <Navbar bg="light" expand="lg" style={{ backgroundColor: '#FFD700', boxShadow: '0 2px 5px rgba(0, 0, 0, 0.1)' }}>
        <Navbar.Brand href="/">SYRCitylineDemo</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ml-auto">
            <Nav.Link href="/ratings">View Ratings</Nav.Link>
            <Nav.Link href="/predict">Predict Resolution Time</Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Navbar>
    );
  }
  

export default NavbarComponent;
