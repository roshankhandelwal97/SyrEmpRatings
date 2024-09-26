# SyrEmpRatings

## Overview
The project involves a React frontend and a Django backend, incorporating a machine learning prediction model to estimate service resolution times and rate government department performances. The frontend includes pages for home navigation, department-specific ratings, and a form for predicting resolution times, allowing users to interact seamlessly. The backend manages data interactions and model integration, facilitating efficient data processing and response handling.

Roshan Khandelwal was responsible for the backend and frontend integration with the machine learning model developed by Mandar Angchekar. This synergy between frontend, backend, and machine learning components ensures a robust and scalable application, capable of handling complex data interactions and delivering precise model predictions.

The machine learning model, built on Python, uses Random Forest algorithms to predict service request completion times based on historical data. This prediction capability is integrated into the React application, allowing users to obtain instant predictions by entering specific details in the provided form. Additionally, the application features a dynamic rating system where departments are rated based on performance metrics, enhancing transparency and accountability.

## Features
- **Effectiveness of Predictions**: The integration of machine learning into municipal service operations has significantly improved prediction accuracy, providing residents with reliable estimates of service completion times.
- **User Engagement**: By allowing residents to access detailed ratings and predicted times, the platform fosters greater community engagement and trust in local government operations.

## Technologies
- **Frontend**: React.js, JavaScript, HTML5, CSS3
- **Backend**: Django, Python
- **Data Analysis**: Jupyter Notebook
- **Database**: SQLite

## Project Structure
- `agency-ratings-frontend/`: Contains all React frontend code.
- `backend/`: Django application with REST API.
- `ratings/`: Django app handling rating logic.
- `Data Analysis and Model.ipynb`: Jupyter notebook for analyzing data.
- `manage.py`: Django script for project management tasks.
- `Data Analysis and Model (1).py`: Model and Prediction Code

## Installation
1. Clone the repository

    `git clone https://github.com/roshankhandelwal97/SyrEmpRatings.git`

3. Install dependencies:

   `cd agency-ratings-frontend npm install cd ../backend pip install -r requirements.txt`


## Usage
Navigate to `localhost:3000` in your web browser to view the frontend. The backend API can be accessed at `localhost:8000`.

## Contributing
Contributions are welcome! Please read the contribution guidelines in `CONTRIBUTING.md` before submitting pull requests.

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Contact
- **Roshan Khandelwal** - [GitHub](https://github.com/mandarangchekar) | [LinkedIn](https://www.linkedin.com/in/mandar-angchekar/) 
- **Roshan Khandelwal** - [GitHub](https://github.com/roshankhandelwal97) | [LinkedIn](https://www.linkedin.com/in/rokhande/) 
- **Project Link** - [SyrEmpRatings](https://github.com/roshankhandelwal97/SyrEmpRatings)

## Acknowledgments
- Syracuse University
- All contributors and testers who have helped shape this project.



