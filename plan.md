# DishDecode Flutter App Development Plan

This document outlines the plan to create a Flutter application for DishDecode, which will serve as the frontend for the existing backend.

## Backend

The backend is a LangGraph graph that will be deployed as a REST API. The Flutter app will communicate with this API.

## Frontend (Flutter)

The Flutter application will be built in the `android` directory.

### 1. Project Setup

*   Initialize a new Flutter project in the `android` directory.
*   Set up the basic project structure, including folders for screens, services, and models.

### 2. User Interface (UI)

*   Create a UI that is similar to the Streamlit prototype.
*   The main screen will have two options:
    *   Take a picture with the camera.
    *   Upload an image from the gallery.
*   Implement a loading indicator to show while the image is being processed.
*   Design a view to display the results, including the dish name, description, and images.

### 3. Image Handling

*   Integrate the `camera` and `image_picker` packages to handle image capture and selection.
*   Implement logic to compress and send the image to the backend API.

### 4. API Integration

*   Use the `http` package to make API calls to the backend.
*   Create data models to represent the API request and response.
*   Implement error handling for API calls.

### 5. State Management

*   Use a state management solution (like Provider or BLoC) to manage the application state, including the image, loading status, and results.

### 6. Displaying Results

*   Create a UI to display the list of recommended dishes.
*   For each dish, display the Korean and English names, description, and a carousel of images.
