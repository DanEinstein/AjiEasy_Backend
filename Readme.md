# AjiEasy Backend 🚀

This repository contains the backend API for AjiEasy, an AI-powered employment preparation service. It's built with **FastAPI** and **Python**, providing a secure, token-based API for user management and dynamic generation of interview questions using **Google's Gemini AI**.

This backend is designed to be hosted on a Python-friendly platform (like Render or Railway) and consumed by a separate frontend (e.g., Vercel).

## 🛠️ Tech Stack

* **Framework:** FastAPI
* **Database:** SQLAlchemy with SQLite
* **AI:** Google Gemini Pro
* **Authentication:** JWT (via `python-jose`) & OAuth2PasswordBearer
* **Password Hashing:** Passlib with `bcrypt`
* **Configuration:** Pydantic-Settings (loads from `.env`)
* **Validation:** Pydantic
* **Server:** Uvicorn

## ✨ Key Features

* **Secure User Authentication:** Full registration and login flow using JWT access tokens.
* **Protected Endpoints:** Core features are protected and require a valid token.
* **Dynamic AI Question Generation:** The main `/generate-questions` endpoint takes a topic name.
* **Smart Service Caching:**
    * If the topic exists in the database, its description is used to generate questions.
    * If the topic does **not** exist, the AI is first used to generate a *description* for that topic, which is then saved to the database.
    * This "smart caching" makes future requests for the same topic much faster and builds your database over time.
* **Service Management:** A protected endpoint (`/services/`) allows authorized users to manually add new interview topics.

---

## 🛑 Prerequisite: Create `requirements.txt`

Before you push to GitHub, you **must** create a `requirements.txt` file so that your host (and other developers) know which packages to install.

Run this command in your `Backend` terminal (while your `venv` is active):

```bash
pip freeze > requirements.txt
📦 Setup and Local Installation
To run this project locally, follow these steps:

Clone the repository:

Bash

git clone  https://github.com/DanEinstein/ajieasy-backend.git
cd ajieasy-backend
Create and activate a virtual environment:

Bash

# Windows
python -m venv venv
.\venv\Scripts\activate
Install the dependencies:

Bash

pip install -r requirements.txt
Create your .env file: Create a file named .env in the root of the Backend folder. Copy the structure from .env.example (or use the one below) and add your secret keys.

Code snippet

# .env
GEMINI_API_KEY="YOUR_GOOGLE_AI_API_KEY"
DATABASE_URL="sqlite:///./ajieasy.db"
SECRET_KEY="YOUR_SUPER_SECRET_RANDOM_STRING_FOR_JWT"
Run the application:

Bash

uvicorn main:app --reload
Access the API: Your API is now live at http://127.0.0.1:8000. You can access the interactive documentation at http://127.0.0.1:8000/docs.

📂 Project Structure
Here is an overview of the file structure and the purpose of each file:

ajieasy-backend/
│
├── .gitignore        # Specifies files for Git to ignore (like .env, venv)
├── .env              # Stores all secret keys and environment variables (IGNORED)
├── ai_service.py     # Core logic for interacting with the Google Gemini API.
│                     # Handles question generation and new topic description generation.
├── auth.py           # Handles all user authentication, password hashing,
│                     # and JWT token creation/verification.
├── config.py         # Uses Pydantic-Settings to load and validate variables
│                     # from the .env file.
├── database.py       # Defines all SQLAlchemy models (User, AiService)
│                     # and handles database session creation.
├── main.py           # The main FastAPI application. Defines all API
│                     # endpoints (routes) and ties all modules together.
├── schemas.py        # Defines all Pydantic models used for API data
│                     # validation (request bodies and responses).
└── requirements.txt  # A list of all Python packages required for the project.
📖 API Endpoints
All endpoints are available to test at http://127.0.0.1:8000/docs.

Authentication
POST /register/

Action: Creates a new user.

Body: { "name": "string", "email": "user@example.com", "password": "string" }

Response: Public user data (no password).

POST /token

Action: Logs in a user and returns a JWT token.

Body: username (your email) and password (form-data).

Response: { "access_token": "...", "token_type": "bearer" }

User
GET /users/me

Action: (Protected) Returns the details of the currently logged-in user.

Response: Public user data.

AI Services
POST /services/

Action: (Protected) Manually adds a new interview topic to the database.

Body: { "name": "Topic Name", "description": "A brief description." }

Response: The newly created service object.

POST /generate-questions/

Action: (Protected) The main feature. Generates AI questions for a topic.

Body: { "service_name": "Python Basics" }

Response: { "questions": "1. What is... \n2. Explain..." }