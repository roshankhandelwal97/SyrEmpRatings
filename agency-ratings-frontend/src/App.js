import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './Home';
import Ratings from './Ratings';
import Department from './Department';
import PredictionForm from './PredictionForm';
import 'bootstrap/dist/css/bootstrap.min.css';
import NavbarComponent from './NavbarComponent';  // Import the Navbar component
import './styles/styles.css'; // Ensure the CSS is imported to apply styles

function App() {
  return (
    <Router>
      <NavbarComponent />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/ratings" element={<Ratings />} />
        <Route path="/predict" element={<PredictionForm />} />
        <Route path="/department/:agencyName" element={<Department />} />
      </Routes>
    </Router>
  );
}

export default App;
