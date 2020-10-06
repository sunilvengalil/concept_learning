# concept_learning

https://docs.google.com/document/d/1QvV-zOu4XU5Eku-SPAycugGWpUOrfSncg-6N8QX-o0E/edit#

## Application Setup

### Step 1 - Build Front-end docker image for pulling latest codebase
```bash
cd app/frontend/

docker build -t clearn-frontend . 
```

### Step 2 - Run the images
```bash
cd app
docker-compose up -d
```

### Step 3 - Initialize DB Schema
```bash
cd app/backend/app/scripts

# Caution, this removes the tables
sh reset_db.sh 

sh init_db.sh
```

### Step 4 - Insert Raw Images
```bash
cd app/backend/app/scripts

python3  insert_raw_images.py --path <absolute path of the base experiment>
```

### Step 5 - Use browser for testing
* [Application - http://localhost](http://localhost/main/dashboard)
* [Backend Docs - http://localhost/docs](http://localhost/docs)