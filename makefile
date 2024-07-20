# Define your frontend and backend repository URLs
FRONTEND_REPO_URL = https://github.com/fatcatt013/tennis-betting-app
BACKEND_REPO_URL = https://github.com/fatcatt013/tennis-betting-backend

# Define the directories where the repos will be cloned
FRONTEND_DIR = frontend
BACKEND_DIR = backend

.PHONY: all init pull-frontend pull-backend

# Default target
all: init pull-frontend pull-backend

# Initial setup target
init:
	@echo "Cloning repositories..."
	git clone $(FRONTEND_REPO_URL) $(FRONTEND_DIR) || (cd $(FRONTEND_DIR) && git pull)
	git clone $(BACKEND_REPO_URL) $(BACKEND_DIR) || (cd $(BACKEND_DIR) && git pull)

# Pull updates for the frontend repository
pull-frontend:
	@echo "Pulling updates for frontend..."
	cd $(FRONTEND_DIR) && git pull

# Pull updates for the backend repository
pull-backend:
	@echo "Pulling updates for backend..."
	cd $(BACKEND_DIR) && git pull