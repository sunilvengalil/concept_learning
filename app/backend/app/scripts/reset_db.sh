#!/bin/bash

docker cp ../schema.sql app_db_1:/
docker exec -it app_db_1 psql  -h localhost -p 5432 -U postgres -d app --no-password -c "DROP table annotated_images;"
docker exec -it app_db_1 psql  -h localhost -p 5432 -U postgres -d app --no-password -c "DROP table raw_images;"
docker exec -it app_db_1 psql  -h localhost -p 5432 -U postgres -d app --no-password -c "DROP table filter_options;"
docker exec -it app_db_1 psql  -h localhost -p 5432 -U postgres -d app --no-password -c 'alter table "user" drop column "totalAnnotations";'




