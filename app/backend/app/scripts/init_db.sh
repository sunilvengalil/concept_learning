#!/bin/bash

docker cp ../schema.sql app_db_1:/
docker exec -it app_db_1 psql  -h localhost -p 5432 -U postgres -d app --no-password -f /schema.sql