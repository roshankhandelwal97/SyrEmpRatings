import React from 'react';
import CategoryCard from './CategoryCard';
import './styles/home.css'; // Ensure the CSS is imported to apply styles

function HomePage() {
  return (
    <div className="home-container">
      <h1>Try out the <strong style={{ fontWeight: 'bold', color: '#0a7ca4', fontSize: '1.5em' }}>QuickWork </strong>model!</h1>
      <div className="cards-container">
        <CategoryCard
          title="View Ratings"
          description={`<br/>Build Public Trust Through Transparency<br/>
          On a single page, you can discover the most hardworking agencies and employees in the city.<br/>
          
          
          <span class="pseudo-button">Trusted Hardworkers</span><br/>
          
          Residents of Syracuse can now see the employees handling their requests and contact the agencies or employees directly.<br/><br/>
          
          The rating is calculated by comparing the actual time taken to the expected time and SLA, rewarding quicker completions with higher scores. <br/>
          Efficiency and responsiveness are combined to produce a score out of 5, where faster and more efficient task completions result in higher ratings.`}
          link="/ratings"
        />
        <CategoryCard
        title="Predict Resolution Time"
        description={`<br/><strong style={{ fontWeight: 'bold', fontSize: '2em' }}>Wondering When Your Request Will Be Completed? </strong><br/>Discover the efficiency of our  – your fast track to insights!<br/><br/>
        <span class="pseudo-button">QuickWork </span><br/>

        How It Works:<br/><br/>
        
        Just enter your Street Name and select the relevant Agency Name and Category from our drop-down menu.<br/>
        It's that simple! Instantly find out the approximate days for the action to be taken.<br/><br/>
        No more waiting in the dark – put our QUICK WORK MODEL to work for you today!`}
        link="/predict"
        />
      </div>
    </div>
  );
}

export default HomePage;
